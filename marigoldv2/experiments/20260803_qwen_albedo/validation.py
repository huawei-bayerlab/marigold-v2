from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from marigoldv2.core.registry import REGISTRY, register
from marigoldv2.validation.util import get_nested_key

ALBEDO_METRIC_NAMES = (
    "albedo_mae",
    "albedo_median_ae",
    "albedo_rmse",
    "albedo_psnr",
)


def _channels(value):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] not in {1, 3}:
        raise ValueError(f"Expected [B,1,H,W] or [B,3,H,W], got {tuple(value.shape)}")
    return value


def albedo_errors(pred, gt, mask=None):
    pred_channels = _channels(pred).float()
    gt_channels = _channels(gt).to(pred_channels.device).float()
    if pred_channels.shape[1] == 1 and gt_channels.shape[1] == 3:
        pred_channels = pred_channels.expand(-1, 3, -1, -1)
    elif pred_channels.shape[1] == 3 and gt_channels.shape[1] == 1:
        gt_channels = gt_channels.expand(-1, 3, -1, -1)
    if pred_channels.shape[1] != gt_channels.shape[1]:
        raise ValueError(
            f"Prediction and ground truth channel counts differ: "
            f"{pred_channels.shape[1]} vs {gt_channels.shape[1]}"
        )
    if pred_channels.shape[-2:] != gt_channels.shape[-2:]:
        pred_channels = F.interpolate(
            pred_channels,
            size=gt_channels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    valid = torch.isfinite(pred_channels).all(dim=1) & torch.isfinite(gt_channels).all(
        dim=1
    )
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        valid &= mask.to(device=pred_channels.device, dtype=torch.bool)
    return (pred_channels - gt_channels).abs(), valid


def albedo_metric_values(pred, gt, mask=None):
    errors, valid = albedo_errors(pred, gt, mask)
    selected = errors.permute(0, 2, 3, 1)[valid]
    if selected.numel() == 0:
        return None
    mse = selected.square().mean()
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    values = {
        "albedo_mae": selected.mean(),
        "albedo_median_ae": selected.quantile(0.5),
        "albedo_rmse": mse.sqrt(),
        "albedo_psnr": psnr,
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
    value = annotation.get(file_path_key) if isinstance(annotation, dict) else None
    if isinstance(value, (list, tuple)):
        value = value[index]
    if value is None:
        return f"sample_{index:04d}"
    scene = annotation.get("scene") if isinstance(annotation, dict) else None
    if isinstance(scene, (list, tuple)):
        scene = scene[index]
    stem = Path(str(value)).stem
    return f"{scene}_{stem}" if scene else stem


def _output_dir(batch, folder_name):
    cfg = REGISTRY["cfg"]
    dataset_name = batch["dataset_disp_name"]
    root = (
        Path(cfg.paths.override_vis_dir) / dataset_name
        if "override_vis_dir" in cfg.get("paths", {})
        else Path(cfg["vis_out_dir"])
    )
    output = root / folder_name
    output.mkdir(parents=True, exist_ok=True)
    return output


def _record(batch, folder_name, path, index):
    saved = batch.setdefault("_saved_output_paths", {})
    saved.setdefault(folder_name, []).append(
        {"batch_index": int(index), "path": str(path)}
    )


def _gray_rgb(value, mask=None):
    scalar = value.detach().cpu().float()
    if scalar.ndim == 3:
        scalar = scalar[0]
    image = (scalar.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    rgb = np.repeat(image[..., None], 3, axis=2)
    if mask is not None:
        mask_np = mask.detach().cpu().bool()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        rgb[~mask_np.numpy()] = 0
    return rgb


def _rgb_image(value, mask=None):
    value = value.detach().cpu().float()
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3 or value.shape[0] not in {1, 3}:
        raise ValueError(f"Expected [1, H, W] or [3, H, W], got {tuple(value.shape)}")
    if value.min() < 0:
        value = (value + 1.0) * 0.5
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    image = (value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    if mask is not None:
        mask_np = mask.detach().cpu().bool()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        image[~mask_np.numpy()] = 0
    return image


def _error_rgb(pred, gt, mask, max_error):
    error, valid = albedo_errors(pred.unsqueeze(0), gt.unsqueeze(0), mask.unsqueeze(0))
    scaled = error.mean(dim=1)[0].clamp(0, max_error) / max_error * 255.0
    heat_bgr = cv2.applyColorMap(
        scaled.detach().cpu().round().to(torch.uint8).numpy(), cv2.COLORMAP_TURBO
    )
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb[~valid[0].detach().cpu().numpy()] = 0
    return heat_rgb


def _title_panel(image_rgb, title):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    canvas = np.full((image_bgr.shape[0] + 28, image_bgr.shape[1], 3), 255, np.uint8)
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
class ComputeAlbedoMetrics:
    def __init__(
        self, pred_key="out/albedo_pred", gt_key="albedo_gt", mask_key="valid_mask", **_
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
                "ComputeAlbedoMetrics could not resolve prediction or target"
            )
        tracker = batch["metric_tracker"]
        for index in range(_batch_size(pred)):
            values = albedo_metric_values(
                _batch_item(pred, index),
                _batch_item(gt, index),
                None if mask is None else _batch_item(mask, index),
            )
            if values is not None:
                for name, value in values.items():
                    tracker.update(name, value)
        return batch


@register("validation_steps")
class VisualizeRGBAlbedo:
    def __init__(
        self,
        batch_key="out/albedo_pred",
        mask_key="valid_mask",
        folder_name="albedo_prediction",
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
            image = _rgb_image(
                _batch_item(value, index),
                None if mask is None else _batch_item(mask, index),
            )
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            _record(batch, self.folder_name, path, index)
        return batch


@register("validation_steps")
class VisualizeGrayscaleAlbedo:
    def __init__(
        self,
        batch_key="out/albedo_pred",
        mask_key="valid_mask",
        folder_name="albedo_prediction",
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
            image = _gray_rgb(
                _batch_item(value, index),
                None if mask is None else _batch_item(mask, index),
            )
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            _record(batch, self.folder_name, path, index)
        return batch


@register("validation_steps")
class VisualizeAlbedoAbsoluteError:
    def __init__(
        self,
        pred_key="out/albedo_pred",
        gt_key="albedo_gt",
        mask_key="valid_mask",
        folder_name="albedo_absolute_error",
        file_path_key="rgb_path",
        max_error=1.0,
        **_,
    ):
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = str(mask_key)
        self.folder_name = str(folder_name)
        self.file_path_key = str(file_path_key)
        self.max_error = float(max_error)

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
                self.max_error,
            )
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            _record(batch, self.folder_name, path, index)
        return batch


@register("validation_steps")
class VisualizeAlbedoComparison:
    def __init__(
        self,
        rgb_key="rgb_norm",
        pred_key="out/albedo_pred",
        gt_key="albedo_gt",
        mask_key="valid_mask",
        folder_name="albedo_comparison",
        file_path_key="rgb_path",
        max_error=1.0,
        **_,
    ):
        self.rgb_key = str(rgb_key)
        self.pred_key = str(pred_key)
        self.gt_key = str(gt_key)
        self.mask_key = str(mask_key)
        self.folder_name = str(folder_name)
        self.file_path_key = str(file_path_key)
        self.max_error = float(max_error)

    def __call__(self, batch):
        rgb = get_nested_key(batch, self.rgb_key)
        pred = get_nested_key(batch, self.pred_key)
        gt = get_nested_key(batch, self.gt_key)
        mask = get_nested_key(batch, self.mask_key)
        output_dir = _output_dir(batch, self.folder_name)
        for index in range(_batch_size(pred)):
            rgb_i = _batch_item(rgb, index)
            pred_i = _batch_item(pred, index)
            gt_i = _batch_item(gt, index)
            mask_i = _batch_item(mask, index)
            panels = [
                _title_panel(_rgb_image(rgb_i), "Input RGB"),
                _title_panel(_rgb_image(gt_i, mask_i), "Albedo GT"),
                _title_panel(_rgb_image(pred_i, mask_i), "Albedo prediction"),
                _title_panel(
                    _error_rgb(pred_i, gt_i, mask_i, self.max_error),
                    f"Absolute error (0-{self.max_error:g})",
                ),
            ]
            image = np.concatenate(panels, axis=1)
            path = output_dir / f"{_base_name(batch, index, self.file_path_key)}.png"
            cv2.imwrite(str(path), image)
            _record(batch, self.folder_name, path, index)
        return batch
