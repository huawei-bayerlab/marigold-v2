"""Output adapters and writers used by ``scripts/infer.py``.

``Folder*Prediction`` graph nodes turn the decoded VAE image into the modality
output; ``SaveFolderPredictionNpy`` and ``VisualizeFolder*`` write one file per
input image, mirroring the input folder structure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import cm

from marigoldv2.core.registry import REGISTRY, register
from marigoldv2.validation.util import get_nested_key


def _get_orig_hw(batch: dict, index: int) -> tuple[int, int] | None:
    meta = batch.get("meta", {})
    orig_res = meta.get("orig_res") if isinstance(meta, dict) else None
    if torch.is_tensor(orig_res):
        if orig_res.ndim == 2 and index < orig_res.shape[0]:
            values = orig_res[index]
        elif orig_res.ndim == 1 and index == 0:
            values = orig_res
        else:
            return None
        if values.numel() < 2:
            return None
        height, width = int(values[0].item()), int(values[1].item())
    elif isinstance(orig_res, (list, tuple)):
        if index < len(orig_res) and isinstance(orig_res[index], (list, tuple)):
            height, width = map(int, orig_res[index][:2])
        elif index == 0 and len(orig_res) >= 2:
            height, width = map(int, orig_res[:2])
        else:
            return None
    else:
        return None
    return (height, width) if height > 0 and width > 0 else None


def _batch_item(value, index: int):
    if torch.is_tensor(value):
        if value.ndim >= 4:
            return value[index]
        return value if index == 0 else None
    if isinstance(value, (list, tuple)):
        return value[index]
    return value if index == 0 else None


def _batch_size(value) -> int:
    return int(value.shape[0]) if torch.is_tensor(value) and value.ndim >= 4 else 1


def _set_output(batch: dict, key: str, value) -> None:
    keys = key.split("/")
    target = batch
    for part in keys[:-1]:
        target = target.setdefault(part, {})
    target[keys[-1]] = value


def _resize_chw(value: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    height, width = orig_hw
    if value.shape[-2:] == orig_hw:
        return value
    if value.ndim == 2:
        return cv2.resize(value.astype(np.float32), (width, height), cv2.INTER_LINEAR)
    channels = [
        cv2.resize(channel.astype(np.float32), (width, height), cv2.INTER_LINEAR)
        for channel in value
    ]
    return np.stack(channels, axis=0)


def _prediction_path(
    batch: dict, folder_name: str, base_dir: Path, index: int, suffix: str
):
    annotation = batch.get("annotation", {})
    source_value = annotation["rgb_path"]
    if isinstance(source_value, (list, tuple)):
        source_value = source_value[index]
    source = Path(str(source_value))
    try:
        relative = source.relative_to(base_dir)
    except ValueError:
        relative = Path(source.name)
    output_root = Path(REGISTRY["cfg"].paths.override_vis_dir)
    output_dir = (
        output_root / batch["dataset_disp_name"] / folder_name / relative.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{relative.stem}{suffix}"


def _prediction_numpy(
    batch: dict,
    batch_key: str,
    index: int,
    resize: bool,
):
    value = _batch_item(get_nested_key(batch, batch_key), index)
    if value is None:
        return None
    array = (
        value.detach().float().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or CHW prediction, got shape {array.shape}")
    if resize:
        orig_hw = _get_orig_hw(batch, index)
        if orig_hw is not None:
            array = _resize_chw(array, orig_hw)
    return array.astype(np.float32, copy=False)


def _depth_rgb(depth: np.ndarray, inverse_spectral: bool = False) -> np.ndarray:
    finite = np.isfinite(depth)
    if np.any(finite):
        low, high = float(np.nanmin(depth[finite])), float(np.nanmax(depth[finite]))
    else:
        low, high = 0.0, 1.0
    if high > low:
        normalized = (depth - low) / (high - low)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0).clip(0.0, 1.0)
    cmap_name = "Spectral_r" if inverse_spectral else "Spectral"
    return (
        (cm.get_cmap(cmap_name)(normalized)[..., :3] * 255.0).round().astype(np.uint8)
    )


def _normal_rgb(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    magnitude = np.linalg.norm(normals, axis=0, keepdims=True)
    valid = magnitude > 1.0e-6
    normalized = np.divide(normals, np.maximum(magnitude, 1.0e-6), where=True)
    normalized = np.where(valid, normalized, 0.0)
    return (
        ((normalized.transpose(1, 2, 0) + 1.0) * 127.5)
        .clip(0, 255)
        .round()
        .astype(np.uint8)
    )


def _albedo_rgb(albedo: np.ndarray) -> np.ndarray:
    return (albedo.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Apply the Marigold V1 gamma-2.2 linear-RGB to sRGB conversion."""
    return value.clamp_min(0.0).pow(1.0 / 2.2)


@register("network_graph")
class FolderDepthPrediction:
    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "out/depth_pred"))

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"FolderDepthPrediction expects [B,3,H,W], got {tuple(decoded.shape)}"
            )
        _set_output(batch, self.output_key, decoded.mean(dim=1, keepdim=True))
        return batch


@register("network_graph")
class FolderNormalizeSurfaceNormals:
    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "out/normal_pred"))
        self.eps = float(kwargs.get("eps", 1.0e-6))

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"FolderNormalizeSurfaceNormals expects [B,3,H,W], got {tuple(decoded.shape)}"
            )
        decoded_float = decoded.float()
        magnitude = torch.linalg.vector_norm(decoded_float, dim=1, keepdim=True)
        normals = decoded_float / magnitude.clamp_min(self.eps)
        normals = torch.where(magnitude > self.eps, normals, torch.zeros_like(normals))
        _set_output(batch, self.output_key, normals.to(dtype=decoded.dtype))
        return batch


@register("network_graph")
class FolderRGBAlbedoPrediction:
    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "out/albedo_pred"))
        self.output_color_space = str(kwargs.get("output_color_space", "linear"))
        if self.output_color_space not in {"linear", "srgb"}:
            raise ValueError(
                "FolderRGBAlbedoPrediction output_color_space must be 'linear' or 'srgb'"
            )

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"FolderRGBAlbedoPrediction expects [B,3,H,W], got {tuple(decoded.shape)}"
            )
        albedo = (decoded.float() + 1.0) * 0.5
        if self.output_color_space == "srgb":
            albedo = _linear_to_srgb(albedo)
        _set_output(batch, self.output_key, albedo.to(decoded.dtype))
        return batch


@register("validation_steps")
class SaveFolderPredictionNpy:
    def __init__(self, **kwargs):
        self.batch_key = str(kwargs.get("batch_key"))
        self.base_dir = Path(str(kwargs["base_dir"])).resolve()
        self.resize_to_orig_res = bool(kwargs.get("resize_to_orig_res", False))
        self.folder_name = str(kwargs.get("folder_name", "predictions_npy"))

    def __call__(self, batch):
        for index in range(_batch_size(get_nested_key(batch, self.batch_key))):
            array = _prediction_numpy(
                batch,
                self.batch_key,
                index,
                self.resize_to_orig_res,
            )
            if array is not None:
                np.save(
                    _prediction_path(
                        batch, self.folder_name, self.base_dir, index, ".npy"
                    ),
                    array,
                )
        return batch


class _FolderVisualization:
    def __init__(self, **kwargs):
        self.batch_key = str(kwargs["batch_key"])
        self.base_dir = Path(str(kwargs["base_dir"])).resolve()
        self.resize_to_orig_res = bool(kwargs.get("resize_to_orig_res", False))
        self.inverse_spectral = bool(kwargs.get("inverse_spectral", False))
        self.folder_name = str(kwargs["folder_name"])

    def _arrays(self, batch):
        value = get_nested_key(batch, self.batch_key)
        for index in range(_batch_size(value)):
            yield (
                index,
                _prediction_numpy(
                    batch,
                    self.batch_key,
                    index,
                    self.resize_to_orig_res,
                ),
            )


@register("validation_steps")
class VisualizeFolderDepth(_FolderVisualization):
    def __call__(self, batch):
        for index, depth in self._arrays(batch):
            if depth is None:
                continue
            image = _depth_rgb(
                depth,
                inverse_spectral=self.inverse_spectral,
            )
            path = _prediction_path(
                batch, self.folder_name, self.base_dir, index, ".png"
            )
            cv2.imwrite(
                str(path),
                image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        return batch


@register("validation_steps")
class VisualizeFolderNormals(_FolderVisualization):
    def __call__(self, batch):
        for index, normals in self._arrays(batch):
            if normals is None:
                continue
            path = _prediction_path(
                batch, self.folder_name, self.base_dir, index, ".png"
            )
            cv2.imwrite(
                str(path), cv2.cvtColor(_normal_rgb(normals), cv2.COLOR_RGB2BGR)
            )
        return batch


@register("validation_steps")
class VisualizeFolderAlbedo(_FolderVisualization):
    def __call__(self, batch):
        for index, albedo in self._arrays(batch):
            if albedo is None:
                continue
            path = _prediction_path(
                batch, self.folder_name, self.base_dir, index, ".png"
            )
            cv2.imwrite(str(path), cv2.cvtColor(_albedo_rgb(albedo), cv2.COLOR_RGB2BGR))
        return batch
