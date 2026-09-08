import torch
import torch.nn.functional as F

from marigoldv2.core.registry import register
from marigoldv2.validation.util import get_nested_key


def _as_scalar(value, name):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] not in {1, 3}:
        raise ValueError(
            f"{name} must be [B,1,H,W] or [B,3,H,W], got {tuple(value.shape)}"
        )
    return value[:, :1]


@register("loss")
class MaskedRGBAlbedoL1Loss:
    """Masked per-channel L1 for linear RGB albedo."""

    def __init__(
        self,
        weight,
        pred_key="out/albedo_pred",
        gt_key="albedo_gt",
        mask_key="valid_mask",
        loss_name="masked_rgb_albedo_l1",
        **_,
    ):
        self.weight = float(weight)
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = None if mask_key is None else str(mask_key)
        self.loss_name = str(loss_name)

    def __call__(self, batch):
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        if pred is None or gt is None:
            raise KeyError(
                f"MaskedRGBAlbedoL1Loss requires '{self.pred_key}' and '{self.gt_key}'"
            )
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if gt.ndim == 3:
            gt = gt.unsqueeze(0)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"Prediction must be [B,3,H,W], got {tuple(pred.shape)}")
        if gt.ndim != 4 or gt.shape[1] != 3:
            raise ValueError(f"Ground truth must be [B,3,H,W], got {tuple(gt.shape)}")
        pred = pred.float()
        gt = gt.to(pred.device).float()
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
        valid = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
        if self.mask_key is not None:
            mask = get_nested_key(batch, self.mask_key)
            if mask is None:
                raise KeyError(f"Missing mask '{self.mask_key}'")
            if mask.ndim == 4:
                mask = mask[:, 0]
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError(f"Expected mask [B,H,W], got {tuple(mask.shape)}")
            valid &= mask.to(device=pred.device, dtype=torch.bool)
        error = (pred - gt).abs()
        pixel_error = error.mean(dim=1)
        loss = (
            pixel_error[valid].mean()
            if valid.any()
            else torch.nan_to_num(pred).sum() * 0.0
        )
        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss
        return batch


@register("loss")
class MaskedGrayscaleAlbedoL1Loss:
    """Masked scalar L1 using the same connected-empty behavior as normals."""

    def __init__(
        self,
        weight,
        pred_key="out/albedo_pred",
        gt_key="albedo_gt",
        mask_key="valid_mask",
        loss_name="masked_grayscale_albedo_l1",
        **_,
    ):
        self.weight = float(weight)
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = None if mask_key is None else str(mask_key)
        self.loss_name = str(loss_name)

    def __call__(self, batch):
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        if pred is None or gt is None:
            raise KeyError(
                f"MaskedGrayscaleAlbedoL1Loss requires '{self.pred_key}' and '{self.gt_key}'"
            )
        pred = _as_scalar(pred, "prediction").float()
        gt = _as_scalar(gt, "ground truth").to(pred.device).float()
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
        valid = torch.isfinite(pred[:, 0]) & torch.isfinite(gt[:, 0])
        if self.mask_key is not None:
            mask = get_nested_key(batch, self.mask_key)
            if mask is None:
                raise KeyError(f"Missing mask '{self.mask_key}'")
            if mask.ndim == 4:
                mask = mask[:, 0]
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError(f"Expected mask [B,H,W], got {tuple(mask.shape)}")
            valid &= mask.to(device=pred.device, dtype=torch.bool)
        error = (pred[:, 0] - gt[:, 0]).abs()
        loss = (
            error[valid].mean() if valid.any() else torch.nan_to_num(pred).sum() * 0.0
        )
        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss
        return batch
