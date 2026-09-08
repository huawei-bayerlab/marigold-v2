import os
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .util import load_rgb_data, _lanczos_resize_chw
from marigoldv2.core.registry import register


def _is_spatial(key, value, exclude_keys):
    """True for tensors with trailing H, W dims that transforms should touch."""
    if not (torch.is_tensor(value) and value.ndim >= 2):
        return False
    return "orig" not in key and key not in exclude_keys


def _resize_like(key, x, size, use_lanczos):
    """Nearest for depth/masks, Lanczos or bilinear for everything else."""
    if "depth" in key or "mask" in key:
        return F.interpolate(x.unsqueeze(0), size, mode="nearest").squeeze(0)
    if use_lanczos:
        return _lanczos_resize_chw(x, size)
    return F.interpolate(
        x.unsqueeze(0), size, mode="bilinear", align_corners=False
    ).squeeze(0)


def _restore_dtype(x, like):
    if like.dtype == torch.bool:
        return x > 0.5
    if not torch.is_floating_point(like):
        return x.round().to(like.dtype)
    return x


@register("dataset_transform")
class ReadRGBImage:
    def __init__(
        self,
        key,
        name,
        use_rel_dir=True,
        set_as_orig_res=False,
        read_icc_profile=False,
        save_as_orig_gt=False,
        save_orig=False,
        jpeg_aug_percent: float = 0.0,
        jpeg_quality_min: int = 30,
        jpeg_quality_max: int = 95,
    ):
        self.key = key
        self.read_icc_profile = read_icc_profile
        self.name = name
        self.use_rel_dir = use_rel_dir
        self.set_as_orig_res = set_as_orig_res
        self.save_as_orig_gt = save_as_orig_gt
        self.save_orig = save_orig
        self.jpeg_aug_percent = jpeg_aug_percent
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max

    def _maybe_jpeg_augment_path(self, img_path: str) -> str:
        if self.jpeg_aug_percent <= 0.0:
            return img_path
        if random.random() >= (self.jpeg_aug_percent / 100.0):
            return img_path
        quality = random.randint(self.jpeg_quality_min, self.jpeg_quality_max)
        with Image.open(img_path).convert("RGB") as im:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            im.save(tmp_path, format="JPEG", quality=quality)
        return tmp_path

    def __call__(self, sample):
        img_path = sample["annotation"][self.key]
        if self.use_rel_dir:
            img_path = os.path.join(sample["rel_dir"], img_path)

        aug_path = self._maybe_jpeg_augment_path(img_path)
        img = load_rgb_data(aug_path, self.name)

        if aug_path != img_path and os.path.exists(aug_path):
            try:
                os.remove(aug_path)
            except OSError:
                pass

        sample.update(img)

        if self.set_as_orig_res:
            sample.setdefault("meta", {})
            sample["meta"]["orig_res"] = [
                sample[self.name + "_norm"].shape[-2],
                sample[self.name + "_norm"].shape[-1],
            ]

        if self.read_icc_profile:
            im_raw = Image.open(img_path)
            icc_profile = im_raw.info.get("icc_profile")
            if icc_profile:
                sample.setdefault("meta", {})
                sample["meta"]["icc_profile"] = icc_profile

        if self.save_orig:
            sample["orig_" + self.name] = img[self.name + "_int"]

        if self.save_as_orig_gt:
            sample["gt_orig"] = img[self.name + "_int"]

        return sample


@register("dataset_transform")
class Reshape:
    """Resize all spatial tensors of a sample to a fixed height and width."""

    def __init__(self, height=768, width=1280, exclude_keys=(), use_lanczos=False):
        self.height = int(height)
        self.width = int(width)
        self.exclude_keys = set(exclude_keys)
        self.use_lanczos = use_lanczos

    def __call__(self, sample):
        size = (self.height, self.width)
        for k, v in list(sample.items()):
            if not _is_spatial(k, v, self.exclude_keys):
                continue
            x = _resize_like(k, v.to(torch.float32), size, self.use_lanczos)
            sample[k] = _restore_dtype(x, v)
        return sample


@register("dataset_transform")
class ReshapeToMultiple:
    """Resize spatial tensors up to the next size divisible by ``multiple``."""

    def __init__(self, multiple=16, exclude_keys=(), use_lanczos=False):
        self.multiple = int(multiple)
        self.exclude_keys = set(exclude_keys)
        self.use_lanczos = use_lanczos

    def __call__(self, sample):
        if self.multiple <= 1:
            return sample
        size = None
        for k, v in sample.items():
            if _is_spatial(k, v, self.exclude_keys):
                h, w = int(v.shape[-2]), int(v.shape[-1])
                m = self.multiple
                size = ((h + m - 1) // m * m, (w + m - 1) // m * m)
                break
        if size is None:
            return sample
        for k, v in list(sample.items()):
            if not _is_spatial(k, v, self.exclude_keys):
                continue
            if tuple(v.shape[-2:]) == size:
                continue
            x = _resize_like(k, v.to(torch.float32), size, self.use_lanczos)
            sample[k] = _restore_dtype(x, v)
        return sample


@register("dataset_transform")
class KittiBMCrop:
    """Apply KITTI benchmark-style crop to tensor fields.

    Default crop keeps the bottom-centered 352x1216 region.
    """

    def __init__(
        self,
        height=352,
        width=1216,
        exclude_keys=None,
        image_key=None,
        depth_key=None,
        mask_key=None,
        **_,
    ):
        self.height = int(height)
        self.width = int(width)
        self.exclude_keys = set(exclude_keys or [])
        self.include_keys = {
            k for k in [image_key, depth_key, mask_key] if k is not None
        }

    def __call__(self, sample):
        Ht, Wt = self.height, self.width
        for k, v in list(sample.items()):
            if not _is_spatial(k, v, self.exclude_keys):
                continue

            H, W = int(v.shape[-2]), int(v.shape[-1])
            if H < Ht or W < Wt:
                continue

            top = H - Ht
            left = max(0, (W - Wt) // 2)
            sample[k] = v[..., top : top + Ht, left : left + Wt]

        return sample


@register("dataset_transform")
class ReadDepthNpyWithNpyMask:
    """Read metric depth and validity mask from .npy files."""

    def __init__(
        self,
        depth_key="depth_path",
        mask_key="mask_path",
        output_key="depth_m",
        inverse_output_key="inv_depth_m",
        min_depth=0.1,
        max_depth=10.0,
        create_inverse=True,
    ):
        self.depth_key = depth_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.inverse_output_key = inverse_output_key
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.create_inverse = bool(create_inverse)

    def __call__(self, sample):
        annotation = sample["annotation"]
        depth_path = annotation[self.depth_key]
        try:
            depth_np = np.load(depth_path).astype(np.float32)
        except (OSError, ValueError, EOFError) as exc:
            raise OSError(
                f"ReadDepthNpyWithNpyMask: failed to load depth file {depth_path}"
            ) from exc
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            raise OSError(
                f"ReadDepthNpyWithNpyMask: invalid depth shape {depth_np.shape} in {depth_path}"
            )

        depth = torch.from_numpy(depth_np).unsqueeze(0).float()
        valid = torch.isfinite(depth)
        valid = valid & (depth >= self.min_depth) & (depth <= self.max_depth)

        mask_path = annotation.get(self.mask_key)
        if mask_path:
            try:
                mask_np = np.load(mask_path)
            except (OSError, ValueError, EOFError) as exc:
                raise OSError(
                    f"ReadDepthNpyWithNpyMask: failed to load mask file {mask_path}"
                ) from exc
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            if mask_np.ndim != 2:
                raise OSError(
                    f"ReadDepthNpyWithNpyMask: invalid mask shape {mask_np.shape} in {mask_path}"
                )
            mask_valid = torch.from_numpy(mask_np > 0).unsqueeze(0)
            valid = valid & mask_valid

        depth = torch.where(valid, depth, torch.zeros_like(depth))
        sample[self.output_key] = depth
        sample["valid_mask"] = valid

        if self.create_inverse:
            inv = torch.zeros_like(depth)
            inv[valid] = 1.0 / depth[valid].clamp(min=self.min_depth)
            sample[self.inverse_output_key] = inv

        return sample


@register("dataset_transform")
class ReadDepthPng16:
    """Read 16-bit depth PNG into metric depth and validity mask.

    NYUv2 official depth maps are typically stored in millimeters.
    """

    def __init__(
        self,
        depth_key="depth_path",
        output_key="depth_m",
        inverse_output_key="inv_depth_m",
        depth_scale=1000.0,
        min_depth=0.1,
        max_depth=10.0,
        create_inverse=True,
    ):
        self.depth_key = depth_key
        self.output_key = output_key
        self.inverse_output_key = inverse_output_key
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.create_inverse = bool(create_inverse)

    def __call__(self, sample):
        annotation = sample["annotation"]
        depth_path = annotation[self.depth_key]
        depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
        if depth_raw.ndim != 2:
            raise ValueError(
                f"ReadDepthPng16 expects 2D depth map, got {depth_raw.shape}"
            )

        depth_np = depth_raw / self.depth_scale
        depth = torch.from_numpy(depth_np).unsqueeze(0).float()

        valid = torch.isfinite(depth) & (depth > 0.0)
        valid = valid & (depth >= self.min_depth) & (depth <= self.max_depth)

        depth = torch.where(valid, depth, torch.zeros_like(depth))
        sample[self.output_key] = depth
        sample["valid_mask"] = valid

        if self.create_inverse:
            inv = torch.zeros_like(depth)
            inv[valid] = 1.0 / depth[valid].clamp(min=self.min_depth)
            sample[self.inverse_output_key] = inv

        return sample


@register("dataset_transform")
class ReadDepthFileAuto:
    """Read depth from .npy/.png/.pfm/.exr or raw float32 maps with image-like suffixes."""

    def __init__(
        self,
        depth_key="depth_path",
        mask_key="mask_path",
        output_key="depth_m",
        inverse_output_key="inv_depth_m",
        png_depth_scale=1000.0,
        min_depth=0.1,
        max_depth=100.0,
        invalid_depth_max=None,
        zero_inverse_above=None,
        create_inverse=True,
    ):
        self.depth_key = depth_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.inverse_output_key = inverse_output_key
        self.png_depth_scale = float(png_depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.invalid_depth_max = (
            None if invalid_depth_max is None else float(invalid_depth_max)
        )
        self.zero_inverse_above = (
            None if zero_inverse_above is None else float(zero_inverse_above)
        )
        self.create_inverse = bool(create_inverse)

    @staticmethod
    def _read_pfm(path):
        with open(path, "rb") as f:
            header = f.readline().decode("ascii", errors="ignore").strip()
            if header not in {"PF", "Pf"}:
                raise ValueError(
                    f"ReadDepthFileAuto: invalid PFM header in {path}: {header}"
                )

            dim_line = f.readline().decode("ascii", errors="ignore").strip()
            while dim_line.startswith("#"):
                dim_line = f.readline().decode("ascii", errors="ignore").strip()
            width_str, height_str = dim_line.split()
            width, height = int(width_str), int(height_str)

            scale = float(f.readline().decode("ascii", errors="ignore").strip())
            endian = "<" if scale < 0 else ">"

            n_channels = 3 if header == "PF" else 1
            data = np.fromfile(f, endian + "f")
            expected = width * height * n_channels
            if data.size != expected:
                raise ValueError(
                    f"ReadDepthFileAuto: malformed PFM in {path}; expected {expected} floats, got {data.size}"
                )

            if n_channels == 1:
                data = data.reshape((height, width))
            else:
                data = data.reshape((height, width, n_channels))[..., 0]
            data = np.flipud(data)
            return data.astype(np.float32)

    @staticmethod
    def _infer_hw_from_sample(sample):
        meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
        orig_res = meta.get("orig_res", None)
        if isinstance(orig_res, (list, tuple)) and len(orig_res) == 2:
            return int(orig_res[0]), int(orig_res[1])

        rgb = sample.get("rgb_norm", None) if isinstance(sample, dict) else None
        if torch.is_tensor(rgb) and rgb.ndim >= 3:
            return int(rgb.shape[-2]), int(rgb.shape[-1])

        return None

    @staticmethod
    def _read_raw_float32_depth(path, hw):
        if hw is None:
            raise OSError(
                "ReadDepthFileAuto: cannot infer depth map resolution for raw float32 depth; "
                "expected sample meta.orig_res or rgb_norm"
            )

        h, w = hw
        depth = np.fromfile(path, dtype=np.float32)
        expected = h * w
        if depth.size != expected:
            raise OSError(
                f"ReadDepthFileAuto: raw depth size mismatch in {path}; expected {expected} floats, got {depth.size}"
            )
        return depth.reshape((h, w)).astype(np.float32)

    @staticmethod
    def _read_exr_depth(path):
        def _pick_depth_channel(arr):
            # Multi-channel EXR: take the channel with the largest finite range.
            if arr.ndim != 3:
                return arr
            best_idx = 0
            best_range = -1.0
            for i in range(arr.shape[2]):
                ch = arr[..., i]
                finite = np.isfinite(ch)
                if not finite.any():
                    continue
                ch_f = ch[finite]
                rng = float(ch_f.max() - ch_f.min())
                if rng > best_range:
                    best_range = rng
                    best_idx = i
            return arr[..., best_idx]

        cv2_exc = None
        try:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2

            exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if exr is None:
                raise OSError(f"cv2.imread returned None for {path}")
            exr = _pick_depth_channel(exr)
            if exr.ndim != 2:
                raise OSError(f"invalid EXR shape {exr.shape} in {path}")
            return exr.astype(np.float32)
        except Exception as exc:
            cv2_exc = exc

        imageio_exc = None
        try:
            import imageio.v3 as iio

            exr = iio.imread(path)
            exr = _pick_depth_channel(np.asarray(exr))
            if exr.ndim != 2:
                raise OSError(f"invalid EXR shape {exr.shape} in {path}")
            return np.asarray(exr, dtype=np.float32)
        except Exception as exc:
            imageio_exc = exc

        raise OSError(
            f"ReadDepthFileAuto: failed to decode EXR depth file {path}; "
            f"cv2 error={cv2_exc}; imageio error={imageio_exc}"
        )

    def _read_depth(self, depth_path, sample):
        suffix = os.path.splitext(depth_path)[1].lower()
        try:
            if suffix == ".npy":
                depth = np.load(depth_path).astype(np.float32)
            elif suffix == ".png":
                depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
                depth = depth_raw / self.png_depth_scale
            elif suffix == ".pfm":
                depth = self._read_pfm(depth_path)
            elif suffix == ".exr":
                depth = self._read_exr_depth(depth_path)
            elif suffix in {".jpg", ".jpeg"}:  # ETH3D raw float32 with .JPG suffix
                depth = self._read_raw_float32_depth(
                    depth_path,
                    hw=self._infer_hw_from_sample(sample),
                )
            else:
                raise OSError(
                    f"ReadDepthFileAuto: unsupported depth file extension '{suffix}' for {depth_path}"
                )
        except (OSError, ValueError, EOFError) as exc:
            raise OSError(
                f"ReadDepthFileAuto: failed to load depth file {depth_path}"
            ) from exc

        if depth.ndim == 3:  # accept HxWx1 and 1xHxW
            if depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif depth.shape[0] == 1:
                depth = depth[0]

        if depth.ndim != 2:
            raise OSError(
                f"ReadDepthFileAuto: invalid depth shape {depth.shape} in {depth_path}"
            )
        return depth

    def _read_mask(self, mask_path):
        suffix = os.path.splitext(mask_path)[1].lower()
        if suffix == ".npy":
            try:
                mask_np = np.load(mask_path)
            except (OSError, ValueError, EOFError) as exc:
                raise OSError(
                    f"ReadDepthFileAuto: failed to load mask file {mask_path}"
                ) from exc
            if mask_np.ndim != 2:
                raise OSError(
                    f"ReadDepthFileAuto: invalid mask shape {mask_np.shape} in {mask_path}"
                )
            return mask_np > 0

        try:
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise OSError(
                f"ReadDepthFileAuto: failed to load mask file {mask_path}"
            ) from exc
        return mask_np > 0

    def __call__(self, sample):
        annotation = sample["annotation"]
        depth_path = annotation.get(self.depth_key)
        if not depth_path:
            raise ValueError(
                "ReadDepthFileAuto: depth_path is missing in sample annotation"
            )

        depth_np = self._read_depth(depth_path, sample)
        depth = torch.from_numpy(depth_np).unsqueeze(0).float()
        depth_unclamped = depth.clone()

        valid = torch.isfinite(depth) & (depth > 0.0)

        if self.invalid_depth_max is not None:
            valid = valid & (depth <= float(self.invalid_depth_max))

        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)

        mask_path = annotation.get(self.mask_key)
        if mask_path:
            mask_valid = torch.from_numpy(self._read_mask(mask_path)).unsqueeze(0)
            valid = valid & mask_valid

        depth = torch.where(valid, depth, torch.zeros_like(depth))
        sample[self.output_key] = depth
        sample["valid_mask"] = valid

        if self.create_inverse:
            inv = torch.zeros_like(depth)
            inverse_valid = valid
            if self.zero_inverse_above is not None:
                inverse_valid = inverse_valid & (
                    depth_unclamped <= self.zero_inverse_above
                )
            inv[inverse_valid] = 1.0 / depth[inverse_valid].clamp(min=self.min_depth)
            sample[self.inverse_output_key] = inv

        return sample


@register("dataset_transform")
class MetricDepthToLogDepth:
    """Convert metric depth to log depth via log(depth + eps)."""

    def __init__(self, key="depth_m", output_key="log_depth", eps=1e-6):
        self.key = key
        self.output_key = output_key
        self.eps = float(eps)

    def __call__(self, sample):
        depth = sample[self.key].float()
        sample[self.output_key] = torch.log(depth + self.eps)
        return sample


@register("dataset_transform")
class NormalizeDepthPercentileAffine:
    """Marigold V1 affine normalization with per-image robust percentiles.

    Maps ``x`` to ``((x - p_low) / (p_high - p_low) - 0.5) * 2`` and clips.
    With ``scale_output_key``/``shift_output_key`` the least-squares inverse
    affine map (normalized -> original) is stored as well. Invalid pixels are
    excluded from the percentiles and set to ``invalid_fill_value``.
    """

    def __init__(
        self,
        key="depth_m",
        output_key=None,
        mask_key="valid_mask",
        low_percentile=2.0,
        high_percentile=98.0,
        dst_min=-1.0,
        dst_max=1.0,
        eps=1e-6,
        invalid_fill_value=0.0,
        scale_output_key=None,
        shift_output_key=None,
    ):
        self.key = key
        self.output_key = output_key if output_key is not None else key
        self.mask_key = mask_key
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self.dst_min = float(dst_min)
        self.dst_max = float(dst_max)
        self.eps = float(eps)
        self.invalid_fill_value = float(invalid_fill_value)
        self.scale_output_key = scale_output_key
        self.shift_output_key = shift_output_key

        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError(
                "Percentiles must satisfy 0 <= low_percentile < high_percentile <= 100."
            )
        if self.dst_max <= self.dst_min:
            raise ValueError(
                "NormalizeDepthPercentileAffine requires dst_max > dst_min."
            )

    def __call__(self, sample):
        x = sample[self.key].float()
        mask = sample.get(self.mask_key)

        if mask is None:
            valid = torch.ones_like(x, dtype=torch.bool)
        else:
            valid = mask.bool()

        out = torch.full_like(x, self.invalid_fill_value)
        valid_values = x[valid]

        if valid_values.numel() == 0:
            sample[self.output_key] = out
            zero_value = torch.zeros((), dtype=x.dtype, device=x.device)
            if self.scale_output_key is not None:
                sample[self.scale_output_key] = zero_value
            if self.shift_output_key is not None:
                sample[self.shift_output_key] = zero_value
            return sample

        q_low = self.low_percentile / 100.0
        q_high = self.high_percentile / 100.0
        p_low, p_high = torch.quantile(
            valid_values,
            torch.tensor(
                [q_low, q_high], dtype=valid_values.dtype, device=valid_values.device
            ),
        )

        denom = (p_high - p_low).clamp_min(self.eps)
        normalized_01 = ((x - p_low) / denom).clamp(0.0, 1.0)

        normalized = normalized_01 * (self.dst_max - self.dst_min) + self.dst_min
        out[valid] = normalized[valid]

        sample[self.output_key] = out
        finite_valid = valid & torch.isfinite(x) & torch.isfinite(normalized)
        fit_values = normalized[finite_valid]
        valid_values = x[finite_valid]
        if valid_values.numel() == 0:
            inverse_scale = torch.zeros((), dtype=x.dtype, device=x.device)
            inverse_shift = torch.zeros((), dtype=x.dtype, device=x.device)
            if self.scale_output_key is not None:
                sample[self.scale_output_key] = inverse_scale
            if self.shift_output_key is not None:
                sample[self.shift_output_key] = inverse_shift
            return sample

        fit_values_mean = fit_values.mean()
        valid_values_mean = valid_values.mean()
        fit_values_centered = fit_values - fit_values_mean
        valid_values_centered = valid_values - valid_values_mean
        fit_var = (fit_values_centered * fit_values_centered).mean()
        fit_cov = (fit_values_centered * valid_values_centered).mean()

        if fit_var <= self.eps:
            inverse_scale = torch.zeros((), dtype=x.dtype, device=x.device)
        else:
            inverse_scale = fit_cov / fit_var
        inverse_shift = valid_values_mean - inverse_scale * fit_values_mean
        if self.scale_output_key is not None:
            sample[self.scale_output_key] = inverse_scale
        if self.shift_output_key is not None:
            sample[self.shift_output_key] = inverse_shift
        return sample


@register("dataset_transform")
class RandomHorizontalFlip:
    """Flip all spatial tensors horizontally with probability ``p``."""

    def __init__(self, p=0.5, exclude_keys=None):
        self.p = float(p)
        self.exclude_keys = set(exclude_keys or [])

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        for k, v in list(sample.items()):
            if _is_spatial(k, v, self.exclude_keys):
                sample[k] = torch.flip(v, dims=[-1])
        return sample
