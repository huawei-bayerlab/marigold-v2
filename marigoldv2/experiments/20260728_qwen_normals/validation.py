from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from marigoldv2.core.registry import REGISTRY, register
from marigoldv2.validation.util import get_nested_key

NORMAL_METRIC_NAMES = (
    "normal_mean_angular_error_deg",
    "normal_median_angular_error_deg",
    "normal_rmse_angular_error_deg",
    "normal_within_5_deg",
    "normal_within_7_5_deg",
    "normal_within_11_25_deg",
    "normal_within_22_5_deg",
    "normal_within_30_deg",
)


def normal_angular_errors_degrees(pred, gt, mask=None, eps=1e-6):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError("Normal tensors must be [B,3,H,W] or [3,H,W]")
    if pred.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError("Normal tensors must have three channels")

    pred = pred.float()
    gt = gt.to(pred.device).float()
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(
            pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    pred = F.normalize(pred, p=2, dim=1, eps=eps)
    gt_normalized = F.normalize(gt, p=2, dim=1, eps=eps)

    valid = torch.isfinite(pred).all(dim=1)
    valid &= torch.isfinite(gt).all(dim=1)
    valid &= torch.linalg.vector_norm(gt, dim=1) > eps
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        valid &= mask.to(device=pred.device, dtype=torch.bool)

    dot = (pred * gt_normalized).sum(dim=1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(dot))
    return angles, valid


def normal_metric_values(pred, gt, mask=None):
    angles, valid = normal_angular_errors_degrees(pred, gt, mask)
    selected = angles[valid]
    if selected.numel() == 0:
        return None
    values = {
        "normal_mean_angular_error_deg": selected.mean(),
        "normal_median_angular_error_deg": selected.quantile(0.5),
        "normal_rmse_angular_error_deg": selected.square().mean().sqrt(),
        "normal_within_5_deg": (selected < 5.0).float().mean() * 100.0,
        "normal_within_7_5_deg": (selected < 7.5).float().mean() * 100.0,
        "normal_within_11_25_deg": (selected < 11.25).float().mean() * 100.0,
        "normal_within_22_5_deg": (selected < 22.5).float().mean() * 100.0,
        "normal_within_30_deg": (selected < 30.0).float().mean() * 100.0,
    }
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def _batch_item(value, index):
    if torch.is_tensor(value):
        return value[index] if value.ndim >= 4 else value
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _batch_size(value):
    return int(value.shape[0]) if torch.is_tensor(value) and value.ndim == 4 else 1


def _base_name(batch, index, file_path_key):
    annotation = batch.get("annotation", {})
    path_value = annotation.get(file_path_key) if isinstance(annotation, dict) else None
    if isinstance(path_value, (list, tuple)):
        path_value = path_value[index]
    if path_value is None:
        return f"sample_{index:04d}"
    scene = None
    if isinstance(annotation, dict):
        scene = annotation.get("scene")
        if isinstance(scene, (list, tuple)):
            scene = scene[index]
    stem = Path(str(path_value)).stem
    return f"{scene}_{stem}" if scene else stem


def _output_dir(batch, folder_name):
    cfg = REGISTRY["cfg"]
    dataset_name = batch["dataset_disp_name"]
    if "override_vis_dir" in cfg.get("paths", {}):
        root = Path(cfg.paths.override_vis_dir) / dataset_name
    else:
        root = Path(cfg["vis_out_dir"])
    output = root / folder_name
    output.mkdir(parents=True, exist_ok=True)
    return output


def _record(batch, folder_name, path, index):
    saved = batch.setdefault("_saved_output_paths", {})
    saved.setdefault(folder_name, []).append(
        {"batch_index": int(index), "path": str(path)}
    )


def _normal_rgb(normal, mask=None):
    normal = normal.detach().cpu().float()
    normal = F.normalize(normal.unsqueeze(0), p=2, dim=1, eps=1e-6)[0]
    rgb = ((normal.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255)
    rgb = rgb.round().astype(np.uint8)
    if mask is not None:
        mask_np = mask.detach().cpu().bool()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        rgb[~mask_np.numpy()] = 0
    return rgb


def _rgb_image(rgb):
    rgb = rgb.detach().cpu().float()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.min() < 0:
        rgb = (rgb + 1.0) * 0.5
    return (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)


def _error_rgb(pred, gt, mask, max_degrees):
    angles, valid = normal_angular_errors_degrees(
        pred.unsqueeze(0), gt.unsqueeze(0), mask.unsqueeze(0)
    )
    scaled = angles[0].clamp(0, max_degrees) / max_degrees * 255.0
    heat_bgr = cv2.applyColorMap(
        scaled.detach().cpu().round().to(torch.uint8).numpy(), cv2.COLORMAP_TURBO
    )
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb[~valid[0].detach().cpu().numpy()] = 0
    return heat_rgb


@register("validation_steps")
class ComputeNormalMetrics:
    def __init__(
        self,
        pred_key="out/normal_pred",
        gt_key="normal_gt",
        mask_key="valid_mask",
        **_,
    ):
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = None if mask_key is None else str(mask_key)

    def __call__(self, batch):
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        mask = None if self.mask_key is None else get_nested_key(batch, self.mask_key)
        if pred is None or gt is None:
            raise KeyError(
                "ComputeNormalMetrics could not resolve prediction or target"
            )

        tracker = batch["metric_tracker"]
        for index in range(_batch_size(pred)):
            values = normal_metric_values(
                _batch_item(pred, index),
                _batch_item(gt, index),
                None if mask is None else _batch_item(mask, index),
            )
            if values is None:
                continue
            for name, value in values.items():
                tracker.update(name, value)
        return batch


@register("validation_steps")
class VisualizeSurfaceNormals:
    def __init__(
        self,
        batch_key="out/normal_pred",
        mask_key="valid_mask",
        folder_name="normal_prediction",
        file_path_key="rgb_path",
        **_,
    ):
        self.batch_key = str(batch_key)
        self.mask_key = None if mask_key is None else str(mask_key)
        self.folder_name = str(folder_name)
        self.file_path_key = str(file_path_key)

    def __call__(self, batch):
        value = get_nested_key(batch, self.batch_key)
        mask = None if self.mask_key is None else get_nested_key(batch, self.mask_key)
        output_dir = _output_dir(batch, self.folder_name)
        for index in range(_batch_size(value)):
            image = _normal_rgb(
                _batch_item(value, index),
                None if mask is None else _batch_item(mask, index),
            )
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            _record(batch, self.folder_name, path, index)
        return batch


@register("validation_steps")
class VisualizeNormalAngularError:
    def __init__(
        self,
        pred_key="out/normal_pred",
        gt_key="normal_gt",
        mask_key="valid_mask",
        folder_name="normal_angular_error",
        file_path_key="rgb_path",
        max_degrees=90.0,
        **_,
    ):
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = str(mask_key)
        self.folder_name = str(folder_name)
        self.file_path_key = str(file_path_key)
        self.max_degrees = float(max_degrees)

    def __call__(self, batch):
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        mask = get_nested_key(batch, self.mask_key)
        output_dir = _output_dir(batch, self.folder_name)
        for index in range(_batch_size(pred)):
            image = _error_rgb(
                _batch_item(pred, index),
                _batch_item(gt, index),
                _batch_item(mask, index),
                self.max_degrees,
            )
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            _record(batch, self.folder_name, path, index)
        return batch


def _title_panel(image_rgb, title):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    canvas = np.full(
        (image_bgr.shape[0] + 28, image_bgr.shape[1], 3), 255, dtype=np.uint8
    )
    canvas[28:] = image_bgr
    cv2.putText(
        canvas,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


@register("validation_steps")
class VisualizeNormalComparison:
    def __init__(
        self,
        rgb_key="rgb_norm",
        pred_key="out/normal_pred",
        gt_key="normal_gt",
        mask_key="valid_mask",
        folder_name="normal_comparison",
        file_path_key="rgb_path",
        max_degrees=90.0,
        **_,
    ):
        self.rgb_key = str(rgb_key)
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = str(mask_key)
        self.folder_name = str(folder_name)
        self.file_path_key = str(file_path_key)
        self.max_degrees = float(max_degrees)

    def __call__(self, batch):
        rgb = get_nested_key(batch, self.rgb_key)
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        mask = get_nested_key(batch, self.mask_key)
        output_dir = _output_dir(batch, self.folder_name)

        for index in range(_batch_size(pred)):
            rgb_i = _rgb_image(_batch_item(rgb, index))
            gt_i = _batch_item(gt, index)
            pred_i = _batch_item(pred, index)
            mask_i = _batch_item(mask, index)
            panels = [
                _title_panel(rgb_i, "Input RGB"),
                _title_panel(_normal_rgb(gt_i, mask_i), "Normal GT"),
                _title_panel(_normal_rgb(pred_i, mask_i), "Normal prediction"),
                _title_panel(
                    _error_rgb(pred_i, gt_i, mask_i, self.max_degrees),
                    f"Angular error (0-{self.max_degrees:g} deg)",
                ),
            ]
            comparison = np.concatenate(panels, axis=1)
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), comparison)
            _record(batch, self.folder_name, path, index)
        return batch
