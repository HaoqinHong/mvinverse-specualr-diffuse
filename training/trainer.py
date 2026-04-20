# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules, expand_frozen_names
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.optimizer import construct_optimizers


def linear_to_srgb_tensor(linear: torch.Tensor) -> torch.Tensor:
    """
    Linear RGB -> sRGB (approx. gamma 2.2)
    """
    # 训练内部很多材质图采用 linear 空间；
    # 保存 PNG 或做可视化时转到更接近人眼显示的 sRGB 空间。
    linear = torch.clamp(linear, 0.0, 1.0)
    return linear ** (1 / 2.2)


def compute_masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> Optional[torch.Tensor]:
    """
    Compute PSNR over valid pixels only.

    Args:
        pred: [B, N, H, W, C]
        target: [B, N, H, W, C]
        mask: [B, N, H, W]
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    valid = mask.unsqueeze(-1)
    valid_count = valid.sum()
    if valid_count.item() == 0:
        return None

    denom = valid.sum().clamp_min(1) * pred.shape[-1]
    mse = ((pred - target) ** 2 * valid).sum() / denom
    psnr = -10.0 * torch.log10(mse.clamp_min(eps))
    return psnr

class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """
    # 一个 epoch / step 的主流程可以概括为：
    # 1. 初始化分布式、模型、loss、优化器、数据集
    # 2. 训练时从 dataset 动态采样多视图 batch
    # 3. forward -> loss -> backward -> optimizer step
    # 4. 周期性保存 checkpoint、跑验证、写 TensorBoard
    #
    # 这份 Trainer 明显是为“动态多视图训练”设计的：
    # batch["images"] 的常见形状是 [B, N, C, H, W]，
    # B 是场景数，N 是每个场景采到的视图数。

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # 保存 Hydra 配置对象，后续 instantiate / 日志 / checkpoint 都会用到。
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # 训练超参数与运行态标量。
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # where 用于 scheduler，表示“当前整体训练进度”在 [0, 1] 的哪个位置。
        self.where = 0.0
        self.best_psnr = float("-inf")
        self.best_psnr_epoch = -1

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # 日志目录和 logger 必须尽早初始化，后面很多地方都会写日志。
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # 先建核心组件，再建 dataloader。
        self._setup_components()
        self._setup_dataloaders()

        # 模型先搬到 device，再构造 optimizer，避免参数对象不一致。
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # optimizer 必须在 model.to(device) 之后构造。
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # 优先使用显式指定的 resume_checkpoint_path，否则尝试从 save_dir 自动恢复。
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # 最后一步再包 DDP，这样前面的权重加载和冻结逻辑都作用在原始模块上。
        self._setup_ddp_distributed_training(distributed, device)
        
        # 所有 rank 对齐后再正式开始训练/验证。
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        # 打印完整环境，调 Docker / CUDA / NCCL / Hydra 问题时很有帮助。
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # 这些 backend 开关主要在“速度”和“可复现性”之间做平衡。
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # 初始化 DDP 通信进程组；后面所有 rank 间同步都依赖这里。
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoint = load_file(ckpt_path)
        else:
            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        
        # strict=False 时，允许“部分加载”：
        # 能对上的权重加载进来，对不上的通过 missing / unexpected 报告出来。
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. \n Missing keys: {missing or 'None'}.\n Unexpected keys: {unexpected or 'None'}.")

        # 如果 checkpoint 里带 optimizer，就尽量一并恢复训练状态。
        try:
            if "optimizer" in checkpoint:
                logging.info(f"Loading optimizer state dict (rank {self.rank})")
                if len(self.optims) == 1:
                    self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])
                else:
                    for idx, optim in enumerate(self.optims):
                        optim.optimizer.load_state_dict(checkpoint["optimizer"][idx])
        except ValueError as e:
            print("[Warning]: Error in loading optimizer's paramter group, abort loading")
            pass

        # epoch / steps / time_elapsed 让训练能够尽可能从中断点接着跑。
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
        self.best_psnr = checkpoint.get("best_psnr", self.best_psnr)
        self.best_psnr_epoch = checkpoint.get("best_psnr_epoch", self.best_psnr_epoch)

        # 混合精度训练时，GradScaler 状态也需要恢复，否则 loss scale 会重新热身。
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        del checkpoint
        torch.cuda.empty_cache()

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # 所有主要组件都通过 Hydra instantiate 构建。
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # 先建模型，再按配置冻结指定模块。
        # 对迁移学习很关键，例如只冻结已经有预训练权重的 encoder。
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

            patterns = expand_frozen_names(self.optim_conf.frozen_module_names)

            self.model = freeze_modules(
                self.model,
                patterns=patterns
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # register token 是否冻结单独给了一个开关，方便针对 Pi3/Pi3X 迁移实验。
        if getattr(self.optim_conf, "freeze_register", False):
            self.model.register_token.requires_grad = False

        # 只在 rank 0 写模型摘要，避免多卡重复写文件。
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            # 这里先实例化 dataset，而不是 dataloader。
            # 真正的 loader 会在每个 epoch 动态创建，因为该项目支持动态多视图采样。
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        # DDP 的这些参数会直接影响通信性能和显存占用。
        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        # 包装后，self.model(...) 就会自动触发梯度同步。
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        # 这里只保存恢复训练真正需要的内容：
        # 模型权重、优化器状态、AMP scaler、epoch/step、累计训练时长。
        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
            "best_psnr": self.best_psnr,
            "best_psnr_epoch": self.best_psnr_epoch,
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # DDPCheckpointSaver 内部会处理“只让合适的 rank 落盘”这件事。
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        saver.save_checkpoint(
            model=model_to_save,
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def _compute_validation_psnr(self, predictions: Mapping, batch: Mapping) -> Optional[torch.Tensor]:
        """
        Compute a mean PSNR over all available material properties in the validation batch.
        """
        psnr_values = []
        prop_names = ["albedo", "metallic", "roughness", "diffuse", "specular", "glossiness", "normal", "shading"]

        for prop_name in prop_names:
            mask_name = f"mask_{prop_name}"
            if prop_name not in predictions or prop_name not in batch or mask_name not in batch:
                continue

            pred = predictions[prop_name]
            target = batch[prop_name].permute(0, 1, 3, 4, 2).contiguous()
            mask = batch[mask_name]

            if prop_name == "normal":
                pred = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
                target = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
            else:
                pred = pred.clamp(0.0, 1.0)
                target = target.clamp(0.0, 1.0)

            psnr = compute_masked_psnr(pred, target, mask)
            if psnr is not None and torch.isfinite(psnr):
                psnr_values.append(psnr)

        if not psnr_values:
            return None

        return torch.stack(psnr_values).mean()

    def _maybe_save_best_checkpoint(self, val_metrics: Optional[Dict[str, float]]) -> None:
        """
        Save best.pt when validation PSNR improves.
        """
        if not val_metrics or "psnr" not in val_metrics:
            return

        current_psnr = float(val_metrics["psnr"])
        if current_psnr <= self.best_psnr:
            return

        self.best_psnr = current_psnr
        self.best_psnr_epoch = int(self.epoch)

        logging.info(
            f"New best PSNR: {self.best_psnr:.4f} at epoch {self.best_psnr_epoch}. Saving best checkpoint."
        )
        self.save_checkpoint(self.epoch, checkpoint_names=["best"])

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            val_metrics = self.run_val()
            self._maybe_save_best_checkpoint(val_metrics)
        elif self.mode == "val":
            val_metrics = self.run_val()
            self._maybe_save_best_checkpoint(val_metrics)

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            # 每个 epoch 调一次 seed，既保证可复现，也让动态采样每轮有变化。
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            # 这个项目的 loader 是“按 epoch 动态构建”的，不是一次建好用到底。
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # 每个 epoch 后至少会更新一次 checkpoint。
            self.save_checkpoint(self.epoch)

            # 动态多视图 batch 很吃显存，这里及时清理缓存。
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 按配置决定是否在当前 epoch 后做验证。
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                val_metrics = self.run_val()
                self._maybe_save_best_checkpoint(val_metrics)
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return None

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        val_metrics = self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return val_metrics


    @torch.no_grad()
    def val_epoch(self, val_loader):
        # AverageMeter / ProgressMeter 是这个项目统一的日志统计工具。
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        psnr_meter = AverageMeter("PSNR", self.device, ":.4f")
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                psnr_meter,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        limit_val_batches = (
            len(val_loader)
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        # 每轮验证单独建一个目录，便于直接翻图看训练效果。
        epoch_val_dir = os.path.join(self.checkpoint_conf.val_dir, f"epoch{self.epoch}")
        os.makedirs(epoch_val_dir, exist_ok=True)

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            
            # 统计 dataloader 取 batch 的耗时。
            data_time.update(time.time() - end)
            
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = torch.bfloat16 if self.optim_conf.amp.amp_dtype == "bfloat16" else torch.float16
            
            # 验证阶段只 forward，不 backward。
            with torch.cuda.amp.autocast(
                enabled=self.optim_conf.amp.enabled,
                dtype=amp_type,
            ):
                val_predictions = self._val_step(batch, self.model, phase, loss_meters)

            batch_psnr = self._compute_validation_psnr(val_predictions, batch)
            if batch_psnr is not None:
                psnr_meter.update(batch_psnr.item(), batch["images"].shape[0])
                if self.rank == 0 and self.steps[phase] % self.logging_conf.log_freq == 0:
                    self.tb_writer.log("Metrics/val/psnr", batch_psnr.item(), self.steps[phase])

            # 这里会把输入图、GT、预测结果、mask 全部保存成 PNG。
            # 对材质任务非常有用，因为很多时候“看图”比只看 loss 更快发现问题。
            from torchvision.utils import save_image
            orig_ids = batch["ids"]
            for ii, scene_cam in enumerate(batch["seq_name"]):
                scene_cam_val_dir = os.path.join(epoch_val_dir, scene_cam.replace('/', '_'))
                os.makedirs(scene_cam_val_dir, exist_ok=True)
                # orig_ids 记录当前采到的是哪些视角，方便回溯对应原始样本。
                orig_id = "_".join([str(xx) for xx in orig_ids[ii].tolist()])
                image = batch["images"][ii] # N, C, H, W
                image = linear_to_srgb_tensor(image)
                save_image(image,  os.path.join(scene_cam_val_dir, f"{orig_id}_image.png"))

                PROPERTIES_TO_VALIDATE = [
                    "albedo", "metallic", "roughness", "diffuse", "specular", "glossiness", "normal", "shading"
                ]
                for prop_name in PROPERTIES_TO_VALIDATE:
                    if prop_name in batch and prop_name in val_predictions:
                        gt_tensor = batch[prop_name][ii]
                        pred_tensor = val_predictions[prop_name][ii].permute(0, 3, 1, 2)

                        # 可视化时把不同物理量转到更适合显示的范围：
                        # 颜色类量 -> sRGB；法线 -> 从 [-1,1] 映射到 [0,1]。
                        if prop_name in ["albedo", "metallic", "roughness", "diffuse", "specular", "glossiness", "shading"]:
                            gt_tensor = linear_to_srgb_tensor(gt_tensor)
                            pred_tensor = linear_to_srgb_tensor(pred_tensor)
                        elif prop_name == "normal":
                            gt_tensor = (gt_tensor + 1) / 2.0
                            pred_tensor = (pred_tensor + 1) / 2.0
                        
                        gt_save_path = os.path.join(scene_cam_val_dir, f"{orig_id}_gt_{prop_name}.png")
                        pred_save_path = os.path.join(scene_cam_val_dir, f"{orig_id}_pred_{prop_name}.png")
                        
                        save_image(gt_tensor, gt_save_path)
                        save_image(pred_tensor, pred_save_path)

                    mask_name = f"mask_{prop_name}"
                    if mask_name in batch:
                        save_image(batch[mask_name][ii].to(float).unsqueeze(1), os.path.join(scene_cam_val_dir, f"{orig_id}_{mask_name}.png"))

            # 统计一个 iteration 从开始到现在的完整耗时。
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        val_metrics = {}
        if psnr_meter.count > 0:
            val_metrics["psnr"] = psnr_meter.avg
            if self.rank == 0:
                logging.info(f"Validation PSNR: {psnr_meter.avg:.4f} dB")

        return val_metrics

    def train_epoch(self, train_loader):        
        # 训练阶段除了 loss，还会额外记录梯度裁剪相关的 grad norm。
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        if self.gradient_clipper:
            for config in self.gradient_clipper.configs: 
                param_names = ",".join(config['module_names'])
                loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        limit_train_batches = (
            len(train_loader)
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # 先把 clipping hook / 配置准备好，真正 step 前再执行。
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break

            # 统计 dataloader 取 batch 的耗时。
            data_time.update(time.time() - end)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # 梯度累积时，会把大 batch 切成多个小 chunk 依次 forward/backward。
            chunked_batches = chunk_batch_for_accum_steps(batch, self.accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # where 表示整体训练进度，用于驱动 scheduler。
            self.where = float(self.epoch + float(data_iter) / limit_train_batches) / self.max_epochs
            
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
                    
            # 把 lr / wd 等 scheduler 调整后的值写进 TensorBoard。
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = f"{i}_{j}_" if len(self.optims) > 1 or len(optim.optimizer.param_groups) > 1 else ""
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

            # 在 scaler.step 之前先 unscale，再做梯度裁剪和监控。
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # 这里才是真正的参数更新时刻。
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # 记录时间和显存峰值。
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        # 梯度累积的核心思想：
        # 一个原始 batch 被拆成多个 chunk，逐个 backward，把梯度累到参数上，
        # 最后只做一次 optimizer.step()，从而模拟更大的 batch。
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)
        amp_type = torch.bfloat16 if self.optim_conf.amp.amp_dtype == "bfloat16" else torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            # DDP 下，除了最后一个 chunk，前面的 backward 不需要立刻做梯度同步，
            # 用 no_sync() 能明显减少通信开销。
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                # 每个 chunk 只承担总 loss 的一部分，所以要除以 accum_steps。
                loss = loss_dict["objective"] / accum_steps
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                # 遇到 NaN / Inf 直接中断本轮，避免把梯度写坏。
                if not math.isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, attempting to stop training")
                    return

                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # 模型只吃图像输入；其它监督量都在 batch 中留给 loss 使用。
        y_hat = model(imgs=batch["images"])
        loss_dict = self.loss(y_hat, batch)
        # 合并预测、loss、原始 batch，后续日志和可视化统一从 log_data 里取。
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _val_step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        """
        # 验证也会算 loss 并记录标量，只是不做 backward。
        y_hat = model(imgs=batch["images"])
        loss_dict = self.loss(y_hat, batch)
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return y_hat

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['images'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                # AverageMeter 负责平滑显示当前 epoch 的均值；
                # TensorBoard 则保留逐 step 的原始曲线。
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase]["keys_to_log"]
            
            # 这里只取 batch 的第一个样本做可视化，再把多个 key 竖着拼起来。
            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0], 
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1).cpu()

            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)

            self.tb_writer.log_visuals(
                f"Visuals/{phase}", visuals_to_log.numpy(), step, self.logging_conf.video_logging_fps
            )


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.
    """
    # 这个工具函数很关键：它支持递归地把一个复杂 batch 拆成多个子 batch，
    # 无论里面是 Tensor、dict、list，都会尽量按 batch 维度切开。
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
