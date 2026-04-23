# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from hydra.utils import instantiate
import torch
import random
import numpy as np
import bisect
from torch.utils.data import Dataset, ConcatDataset
from .dataset_util import *


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]

class ComposedIntrinsicDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Mode Settings ---
        # Whether the dataset is being used for training 
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, albedo, metallic, roughness, normal, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        def _to_chw_tensor(key, channel_last=True):
            if (
                key not in batch
                or batch[key] is None
                or len(batch[key]) == 0
                or any(item is None for item in batch[key])
            ):
                return None
            tensor = torch.from_numpy(np.stack(batch[key]).astype(np.float32)).contiguous()
            if channel_last:
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor.to(torch.get_default_dtype())

        def _to_mask_tensor(key):
            if (
                key not in batch
                or batch[key] is None
                or len(batch[key]) == 0
                or any(item is None for item in batch[key])
            ):
                return None
            return torch.from_numpy(np.stack(batch[key]).astype(bool)).contiguous()

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        images = images.permute(0,3,1,2).to(torch.get_default_dtype())

        albedo = _to_chw_tensor("albedo")
        mask_albedo = _to_mask_tensor("mask_albedo")
        metallic = _to_chw_tensor("metallic")
        mask_metallic = _to_mask_tensor("mask_metallic")
        roughness = _to_chw_tensor("roughness")
        mask_roughness = _to_mask_tensor("mask_roughness")
        diffuse = _to_chw_tensor("diffuse")
        mask_diffuse = _to_mask_tensor("mask_diffuse")
        specular = _to_chw_tensor("specular")
        mask_specular = _to_mask_tensor("mask_specular")
        glossiness = _to_chw_tensor("glossiness")
        mask_glossiness = _to_mask_tensor("mask_glossiness")
        normal = _to_chw_tensor("normal")
        normal_view = _to_chw_tensor("normal_view")
        mask_normal = _to_mask_tensor("mask_normal")
        view = _to_chw_tensor("view")
        shading = _to_chw_tensor("shading")
        mask_shading = _to_mask_tensor("mask_shading")
        camera_intrinsics = _to_chw_tensor("camera_intrinsics", channel_last=False)
        camera_c2w = _to_chw_tensor("camera_c2w", channel_last=False)
        camera_w2c = _to_chw_tensor("camera_w2c", channel_last=False)
        camera_position = _to_chw_tensor("camera_position", channel_last=False)

        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence

        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
            "albedo": albedo,
            "metallic": metallic,
            "roughness": roughness,
            "diffuse": diffuse,
            "specular": specular,
            "glossiness": glossiness,
            "normal": normal,
            "normal_view": normal_view,
            "view": view,
            "shading": shading,
            "camera_intrinsics": camera_intrinsics,
            "camera_c2w": camera_c2w,
            "camera_w2c": camera_w2c,
            "camera_position": camera_position,
            "mask_albedo": mask_albedo,
            "mask_metallic": mask_metallic,
            "mask_roughness": mask_roughness,
            "mask_diffuse": mask_diffuse,
            "mask_specular": mask_specular,
            "mask_glossiness": mask_glossiness,
            "mask_normal": mask_normal,
            "mask_shading": mask_shading,
        }

        sample = {k: v for k, v in sample.items() if v is not None}
        return sample
