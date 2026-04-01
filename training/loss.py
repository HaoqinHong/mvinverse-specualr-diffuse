
import torch
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Albedo loss
    - Metallic loss 
    - Roughness loss
    - Normal loss 
    """
    # def __init__(self, camera=None, depth=None, point=None, track=None, albedo=None, metallic=None, roughness=None, normal=None, **kwargs):
    def __init__(self, albedo=None, metallic=None, roughness=None, normal=None, shading=None, **kwargs):
        super().__init__()
        ## Loss configuration dictionaries for each task
        self.albedo = albedo
        self.metallic = metallic
        self.roughness = roughness
        self.normal = normal
        self.shading = shading

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        if "albedo" in predictions:
            albedo_loss_dict = compute_albedo_loss(predictions, batch, **self.albedo)
            albedo_loss = \
                self.albedo["reg_weight"] * albedo_loss_dict["loss_reg_albedo"] + \
                self.albedo["grad_weight"] * albedo_loss_dict["loss_grad_albedo"]
            total_loss = total_loss + albedo_loss
            loss_dict.update(albedo_loss_dict)

        if "metallic" in predictions:
            metallic_loss_dict = compute_metallic_loss(predictions, batch, **self.metallic)
            metallic_loss = \
                self.metallic["reg_weight"] * metallic_loss_dict["loss_reg_metallic"] + \
                self.metallic["grad_weight"] * metallic_loss_dict["loss_grad_metallic"]
            total_loss = total_loss + metallic_loss
            loss_dict.update(metallic_loss_dict)
        
        if "roughness" in predictions:
            roughness_loss_dict = compute_roughness_loss(predictions, batch, **self.roughness)
            roughness_loss = \
                self.roughness["reg_weight"] * roughness_loss_dict["loss_reg_roughness"] + \
                self.roughness["grad_weight"] * roughness_loss_dict["loss_grad_roughness"]
            total_loss = total_loss + roughness_loss
            loss_dict.update(roughness_loss_dict)

        if "normal" in predictions:
            normal_loss_dict = compute_normal_loss(predictions, batch, **self.normal)
            normal_loss = \
                self.normal["weight"] * normal_loss_dict["loss_negcos_normal"]
            total_loss = total_loss + normal_loss
            loss_dict.update(normal_loss_dict)

        # if "shading" in predictions:
        if "shading" in predictions and "shading" in batch:
            shading_loss_dict = compute_shading_loss(predictions, batch, **self.shading)
            shading_loss = \
                self.shading["reg_weight"] * shading_loss_dict["loss_reg_shading"] + \
                self.shading["grad_weight"] * shading_loss_dict["loss_grad_shading"]
            total_loss = total_loss + shading_loss
            loss_dict.update(shading_loss_dict)

        loss_dict["objective"] = total_loss
        return loss_dict


def compute_albedo_loss(predictions, batch, **kwargs):
    """
    Compute albedo loss.
    
    Args:
        predictions: Dict containing 'albedo'
        batch: Dict containing ground truth 'albedo'
        gradient_loss_fn: Type of gradient loss to apply
    """
    pred_albedo = predictions['albedo']  # [B, N, H, W, C]
    mask_albedo = batch["mask_albedo"]

    B, N, H, W, C = pred_albedo.shape
    # if dataset is multi illum dataset, compute pseudo gt using the mean
    gt_albedo = []
    for i, seq_name in enumerate(batch["seq_name"]):
        if "multi_illum" in seq_name:
            gt = torch.mean(pred_albedo[i], dim=0, keepdim=True).detach() # [1, H, W, C]
            gt = gt.expand((N, H, W, C))
        else:
            gt = batch['albedo'][i].permute(0, 2, 3, 1).contiguous() # [N, H, W, C]
        gt_albedo.append(gt)
    gt_albedo = torch.stack(gt_albedo, dim=0)

    if mask_albedo.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_albedo).mean()
        loss_dict = {
            f"loss_reg_albedo": dummy_loss,
            f"loss_grad_albedo": dummy_loss,
        }
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    b_scale = kwargs["b_scale"] if "b_scale" in kwargs else True
    scales = kwargs["scales"] if "scales" in kwargs else 4
    loss_reg, loss_grad = material_regression_loss(pred_albedo, gt_albedo, mask_albedo, b_scale=b_scale, scales=scales)

    loss_dict = {
        f"loss_reg_albedo": loss_reg,
        f"loss_grad_albedo": loss_grad,
    }

    return loss_dict

def compute_metallic_loss(predictions, batch, **kwargs):
    """
    Compute metallic loss.
    
    Args:
        predictions: Dict containing 'metallic'
        batch: Dict containing ground truth 'metallic'
        gradient_loss_fn: Type of gradient loss to apply
    """
    pred_metallic = predictions['metallic']
    gt_metallic = batch['metallic'].permute(0, 1, 3, 4, 2).contiguous()
    mask_metallic = batch["mask_metallic"]

    if mask_metallic.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_metallic).mean()
        loss_dict = {
            f"loss_reg_metallic": dummy_loss,
            f"loss_grad_metallic": dummy_loss,
        }
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_reg, loss_grad = material_regression_loss(pred_metallic, gt_metallic, mask_metallic)
    
    loss_dict = {
        f"loss_reg_metallic": loss_reg,
        f"loss_grad_metallic": loss_grad,
    }

    return loss_dict

def compute_roughness_loss(predictions, batch, **kwargs):
    """
    Compute roughness loss.
    
    Args:
        predictions: Dict containing 'roughness'
        batch: Dict containing ground truth 'roughness'
        gradient_loss_fn: Type of gradient loss to apply
    """
    pred_roughness = predictions['roughness']
    gt_roughness = batch['roughness'].permute(0, 1, 3, 4, 2).contiguous()
    mask_roughness = batch["mask_roughness"]
   
    if mask_roughness.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_roughness).mean()
        loss_dict = {
            f"loss_reg_roughness": dummy_loss,
            f"loss_grad_roughness": dummy_loss,
        }
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_reg, loss_grad = material_regression_loss(pred_roughness, gt_roughness, mask_roughness)

    loss_dict = {
        f"loss_reg_roughness": loss_reg,
        f"loss_grad_roughness": loss_grad,
    }

    return loss_dict

def compute_normal_loss(predictions, batch, **kwargs):
    """
    Compute normal loss.
    
    Args:
        predictions: Dict containing 'normal'
        batch: Dict containing ground truth 'normal'
        gradient_loss_fn: Type of gradient loss to apply
    """
    pred_normal = predictions['normal'] 
    gt_normal = batch['normal'].permute(0, 1, 3, 4, 2).contiguous() # [B, N, H, W, 3]
    mask_normal = batch["mask_normal"]

    # normalize (l2 norm)
    pred_normal = F.normalize(pred_normal, p=2, dim=-1, eps=1e-6)

    gt_normal = torch.nan_to_num(gt_normal, nan=0.0)
    gt_normal = F.normalize(gt_normal, p=2, dim=-1, eps=1e-6)

    # negative cosine simlarity
    dot_product = torch.sum(pred_normal * gt_normal, dim=-1, keepdim=False)
    loss_map = 1 - dot_product

    # gt normal might contain Nan values
    nan_mask = torch.isnan(gt_normal).any(dim=-1)
    # mask invalid pixels
    if mask_normal.dtype != torch.bool:
        mask_normal = mask_normal.bool()
    mask_normal = mask_normal & ~nan_mask

    loss_negcos = torch.sum(loss_map[mask_normal])
    num_valid_pixels = torch.sum(mask_normal)

    if num_valid_pixels.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_normal).mean()
        loss_dict = {
            f"loss_negcos_normal": dummy_loss,
        }
        return loss_dict

    loss_negcos = loss_negcos / num_valid_pixels
    loss_dict = {
        f"loss_negcos_normal": loss_negcos,
    }
    return loss_dict

def compute_shading_loss(predictions, batch, **kwargs):
    """
    Compute shading loss.
    
    Args:
        predictions: Dict containing 'shading'
        batch: Dict containing ground truth 'shading'
        gradient_loss_fn: Type of gradient loss to apply
    """
    pred_shading = predictions['shading']  # [B, N, H, W, C]
    mask_shading = batch["mask_shading"]

    B, N, H, W, C = pred_shading.shape
    # if dataset is multi illum dataset, compute pseudo gt using the mean
    gt_shading = []
    for i, seq_name in enumerate(batch["seq_name"]):
        if "multi_illum" in seq_name:
            gt = torch.mean(pred_shading[i], dim=0, keepdim=True).detach() # [1, H, W, C]
            gt = gt.expand((N, H, W, C))
        else:
            gt = batch['shading'][i].permute(0, 2, 3, 1).contiguous() # [N, H, W, C]
        gt_shading.append(gt)
    gt_shading = torch.stack(gt_shading, dim=0)

    if mask_shading.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_shading).mean()
        loss_dict = {
            f"loss_reg_shading": dummy_loss,
            f"loss_grad_shading": dummy_loss,
        }
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    b_scale = kwargs["b_scale"] if "b_scale" in kwargs else True
    scales = kwargs["scales"] if "scales" in kwargs else 4
    loss_reg, loss_grad = material_regression_loss(pred_shading, gt_shading, mask_shading, b_scale=b_scale, scales=scales)

    loss_dict = {
        f"loss_reg_shading": loss_reg,
        f"loss_grad_shading": loss_grad,
    }

    return loss_dict

def compute_scale(pred, gt, mask, eps=1e-5):
    """
    Args:
        pred: (B, C, H, W)
        gt: (B, C, H, W)
        mask: (B, H, W)
    
    Returns:
        s: (B, C)
    """
    mask_expanded = mask.unsqueeze(1) # (B, 1, H, W)

    pred_masked = pred * mask_expanded
    gt_masked = gt * mask_expanded
    
    nume = torch.sum(pred_masked * gt_masked, dim=(-2, -1))
    denom = torch.sum(pred_masked * pred_masked, dim=(-2, -1))

    s = nume / (denom + eps)
    
    return s.detach()

def material_regression_loss(pred, gt, mask, b_scale=False, scale_min=0.5, scale_max=2.0, scales=4):
    """
    material loss function with mse error and multi-scale gradient loss.
    
    Computes:
    1. ||pred - gt||**2 / N (MSE Loss)
    2. Multi-scale Gradient Loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
    
    Returns:
        loss_reg: l2 regression loss
        loss_grad: multi-scale gradient loss
    """
    bb, ss, hh, ww, nc = pred.shape

    if b_scale:
        pred_permuted = pred.reshape(bb * ss, hh, ww, nc).permute(0, 3, 1, 2)
        gt_permuted = gt.reshape(bb * ss, hh, ww, nc).permute(0, 3, 1, 2)
        mask_reshaped = mask.reshape(bb * ss, hh, ww)
        
        with torch.no_grad():
            computed_scale = compute_scale(pred_permuted, gt_permuted, mask_reshaped)

        valid_scale_mask = (computed_scale > scale_min) & (computed_scale < scale_max) # [bb*ss, nc]
        # replace invalid scale with zeros
        scale = torch.where(valid_scale_mask, computed_scale, torch.ones_like(computed_scale))

        scale = scale.view(bb * ss, nc, 1, 1)
        pred_scaled_permuted = scale * pred_permuted
        pred = pred_scaled_permuted.permute(0, 2, 3, 1).reshape(bb, ss, hh, ww, nc)

    loss_reg = F.mse_loss(pred[mask], gt[mask])
    # loss_reg = F.l1_loss(pred[mask], gt[mask])

    # --- Start of inlined Multi-Scale Gradient Loss calculation ---
    loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            scales=scales,
        )

    pred = pred.reshape(bb * ss, hh, ww, nc).permute(0, 3, 1, 2)
    gt = gt.reshape(bb * ss, hh, ww, nc).permute(0, 3, 1, 2)

    # --- End of inlined Multi-Scale Gradient Loss calculation ---
    return loss_reg, loss_grad

def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total

def gradient_loss_multi_scale_wrapper_bilinear(prediction, target, mask, scales=4, gradient_loss_fn=None):
    """
    Multi-scale gradient loss using Bilinear Interpolation for downsampling.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0

    # PyTorch's interpolation functions require (B, C, H, W) format.
    pred_chw = prediction.permute(0, 3, 1, 2)
    target_chw = target.permute(0, 3, 1, 2)
    
    # Unsqueeze mask and conf to add a channel dimension: (B, H, W) -> (B, 1, H, W)
    mask_bchw = mask.unsqueeze(1)

    for scale in range(scales):
        if scale == 0:
            # At the original scale, no downsampling is needed
            pred_scale = prediction
            target_scale = target
            mask_scale = mask
        else:
            # Calculate the scale factor for downsampling
            scale_factor = 1.0 / pow(2, scale)

            # Downsample using bilinear interpolation
            pred_downsampled_chw = F.interpolate(pred_chw, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            target_downsampled_chw = F.interpolate(target_chw, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            mask_downsampled_bchw = F.interpolate(mask_bchw.float(), scale_factor=scale_factor, mode='bilinear', align_corners=False)
            
            # Permute dimensions back to (B, H, W, C)
            pred_scale = pred_downsampled_chw.permute(0, 2, 3, 1)
            target_scale = target_downsampled_chw.permute(0, 2, 3, 1)

            # Squeeze the channel dimension back out for mask and conf
            mask_scale = mask_downsampled_bchw.squeeze(1)
            
        total += gradient_loss_fn(
            pred_scale,
            target_scale,
            mask_scale,
        )

    total = total / scales
    return total


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss

def gradient_loss_mse(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L2 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.square(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.square(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss