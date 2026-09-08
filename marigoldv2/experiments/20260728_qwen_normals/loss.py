import torch
import torch.nn.functional as F

from marigoldv2.core.registry import register
from marigoldv2.validation.util import get_nested_key


def _as_batched_normals(value, name):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,H,W], got {tuple(value.shape)}")
    return value


@register("loss")
class AngularNormalLoss:
    """Masked mean angular error in radians with stable float32 gradients."""

    def __init__(
        self,
        weight,
        pred_key="out/normal_pred",
        gt_key="normal_gt",
        mask_key="valid_mask",
        eps=1e-6,
        loss_name="angular_normal_loss",
        **_,
    ):
        self.weight = float(weight)
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = None if mask_key is None else str(mask_key)
        self.eps = float(eps)
        self.loss_name = str(loss_name)

    def __call__(self, batch):
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        if pred is None or gt is None:
            raise KeyError(
                f"AngularNormalLoss requires '{self.pred_key}' and '{self.gt_key}'"
            )

        pred = _as_batched_normals(pred, "prediction").float()
        gt = _as_batched_normals(gt, "ground truth").to(pred.device).float()
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )

        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        gt_norm = F.normalize(gt, p=2, dim=1, eps=self.eps)
        valid = torch.isfinite(pred).all(dim=1)
        valid &= torch.isfinite(gt).all(dim=1)
        valid &= torch.linalg.vector_norm(gt, dim=1) > self.eps

        if self.mask_key is not None:
            mask = get_nested_key(batch, self.mask_key)
            if mask is None:
                raise KeyError(f"AngularNormalLoss: missing mask '{self.mask_key}'")
            if mask.ndim == 4:
                mask = mask[:, 0]
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError(f"Expected mask [B,H,W], got {tuple(mask.shape)}")
            valid &= mask.to(device=pred.device, dtype=torch.bool)

        dot = (pred_norm * gt_norm).sum(dim=1)
        dot = dot.clamp(-1.0 + self.eps, 1.0 - self.eps)
        angles = torch.acos(dot)
        loss = (
            angles[valid].mean() if valid.any() else torch.nan_to_num(pred).sum() * 0.0
        )

        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss
        return batch
