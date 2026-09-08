import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from marigoldv2.core.registry import register

_RGB_NAME_RE = re.compile(r"^rgb_(?P<camera>.+)_fr(?P<frame>\d+)$")


@register("manifest_transform")
class LoadHypersimNormalsManifest:
    """Load two-column RGB/normal manifests produced by the Marigold V1.1 preprocessor."""

    def __init__(
        self,
        base_dir,
        split,
        filelist_path=None,
        max_samples=None,
        seed=42,
        verify_files=True,
    ):
        self.base_dir = Path(base_dir)
        self.split = str(split)
        self.filelist_path = (
            Path(filelist_path)
            if filelist_path is not None
            else self.base_dir / f"hypersim_filtered_{self.split}.txt"
        )
        self.max_samples = None if max_samples is None else int(max_samples)
        self.seed = int(seed)
        self.verify_files = bool(verify_files)

    def _resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.base_dir / path

    def _metadata(self, rgb_path):
        try:
            rel = rgb_path.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError(
                f"RGB path is outside the Hypersim normals root: {rgb_path}"
            ) from exc

        if len(rel.parts) != 3:
            raise ValueError(
                "Expected RGB path '<split>/<scene>/<filename>', "
                f"got '{rel.as_posix()}'"
            )
        split, scene, filename = rel.parts
        if split != self.split:
            raise ValueError(
                f"Manifest split mismatch: requested '{self.split}', found '{split}' "
                f"in '{rel.as_posix()}'"
            )

        match = _RGB_NAME_RE.match(Path(filename).stem)
        if match is None:
            raise ValueError(f"Unexpected Hypersim RGB filename: {filename}")
        return scene, match.group("camera"), match.group("frame")

    def __call__(self, manifest):
        if not self.filelist_path.is_file():
            raise FileNotFoundError(
                f"Hypersim normals manifest does not exist: {self.filelist_path}"
            )

        entries = []
        with self.filelist_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                columns = line.split()
                if len(columns) != 2:
                    raise ValueError(
                        f"{self.filelist_path}:{line_number}: expected exactly two "
                        f"columns (RGB normal), got {len(columns)}"
                    )

                rgb_path = self._resolve(columns[0])
                normal_path = self._resolve(columns[1])
                if self.verify_files:
                    missing = [
                        str(path)
                        for path in (rgb_path, normal_path)
                        if not path.is_file()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"{self.filelist_path}:{line_number}: missing files: {missing}"
                        )

                scene, camera, frame = self._metadata(rgb_path)
                entries.append(
                    {
                        "split": self.split,
                        "scene": scene,
                        "camera": camera,
                        "frame": frame,
                        "rgb_path": str(rgb_path),
                        "normal_path": str(normal_path),
                    }
                )

        if not entries:
            raise ValueError(f"No samples found in {self.filelist_path}")

        if self.max_samples is not None and self.max_samples < len(entries):
            indices = list(range(len(entries)))
            random.Random(self.seed).shuffle(indices)
            selected = set(indices[: self.max_samples])
            entries = [
                entry for index, entry in enumerate(entries) if index in selected
            ]

        for index, entry in enumerate(entries):
            key = (
                f"hypersim_normals_{entry['split']}_{entry['scene']}_"
                f"{entry['camera']}_{entry['frame']}_{index:06d}"
            )
            if key in manifest:
                raise ValueError(f"Duplicate manifest key: {key}")
            manifest[key] = entry
        return manifest


def _normalize_normals(normal, valid_mask, eps):
    finite = torch.isfinite(normal).all(dim=0, keepdim=True)
    safe = torch.where(torch.isfinite(normal), normal, torch.zeros_like(normal))
    norm = torch.linalg.vector_norm(safe, dim=0, keepdim=True)
    valid = valid_mask.bool() & finite & (norm > eps)
    normalized = safe / norm.clamp_min(eps)
    normalized = torch.where(
        valid.expand_as(normalized), normalized, torch.zeros_like(normalized)
    )
    return normalized, valid


@register("dataset_transform")
class ReadSurfaceNormalsNpy:
    """Read HWC camera-space normals (Marigold V1.1 format) and build a valid mask."""

    def __init__(
        self,
        key="normal_path",
        output_key="normal_gt",
        mask_key="valid_mask",
        eps=1e-6,
    ):
        self.key = str(key)
        self.output_key = str(output_key)
        self.mask_key = str(mask_key)
        self.eps = float(eps)

    def __call__(self, sample):
        normal_path = sample["annotation"][self.key]
        array = np.load(normal_path, allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected HWC normals with three channels in {normal_path}, "
                f"got shape {array.shape}"
            )

        normal = torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(2, 0, 1)
        initial_mask = torch.ones(
            (1, normal.shape[-2], normal.shape[-1]), dtype=torch.bool
        )
        normal, valid = _normalize_normals(normal, initial_mask, self.eps)
        sample[self.output_key] = normal
        sample[self.mask_key] = valid
        return sample


@register("dataset_transform")
class RandomHorizontalFlipSurfaceNormals:
    """Flip aligned tensors and negate the camera-space normal x component."""

    def __init__(
        self,
        p=0.5,
        normal_key="normal_gt",
        include_keys=None,
        exclude_keys=None,
    ):
        self.p = float(p)
        self.normal_key = str(normal_key)
        self.include_keys = None if include_keys is None else set(include_keys)
        self.exclude_keys = set(exclude_keys or [])

    def __call__(self, sample):
        do_flip = random.random() < self.p
        sample["hflip_applied"] = do_flip
        if not do_flip:
            return sample

        for key, value in list(sample.items()):
            if not torch.is_tensor(value) or value.ndim < 2:
                continue
            if "orig" in key or key in self.exclude_keys:
                continue
            if self.include_keys is not None and key not in self.include_keys:
                continue
            sample[key] = torch.flip(value, dims=[-1])

        normal = sample.get(self.normal_key)
        if normal is None or not torch.is_tensor(normal) or normal.ndim != 3:
            raise ValueError(
                f"Expected '{self.normal_key}' as [3,H,W] after horizontal flip"
            )
        if normal.shape[0] != 3:
            raise ValueError(
                f"Expected '{self.normal_key}' to have three channels, got {normal.shape}"
            )
        normal = normal.clone()
        normal[0].neg_()
        sample[self.normal_key] = normal
        return sample


@register("dataset_transform")
class ResizeRGBSurfaceNormals:
    """Resize aligned RGB, normals, and masks with task-appropriate interpolation."""

    def __init__(
        self,
        height=576,
        width=768,
        rgb_keys=("rgb_norm", "rgb_int"),
        normal_key="normal_gt",
        mask_key="valid_mask",
        eps=1e-6,
    ):
        self.height = int(height)
        self.width = int(width)
        self.rgb_keys = list(rgb_keys)
        self.normal_key = str(normal_key)
        self.mask_key = str(mask_key)
        self.eps = float(eps)

    @staticmethod
    def _resize_float(value, size, mode):
        kwargs = {"mode": mode, "size": size}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(value.unsqueeze(0).float(), **kwargs).squeeze(0)

    def __call__(self, sample):
        size = (self.height, self.width)
        for key in self.rgb_keys:
            value = sample.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim != 3:
                raise ValueError(f"Expected '{key}' as [C,H,W], got {type(value)}")
            resized = self._resize_float(value, size, "bilinear")
            if not torch.is_floating_point(value):
                resized = resized.round().clamp(0, 255).to(value.dtype)
            else:
                resized = resized.to(value.dtype)
            sample[key] = resized

        normal = sample.get(self.normal_key)
        mask = sample.get(self.mask_key)
        if normal is None or mask is None:
            raise KeyError(
                f"ResizeRGBSurfaceNormals requires '{self.normal_key}' and '{self.mask_key}'"
            )
        if normal.ndim != 3 or normal.shape[0] != 3:
            raise ValueError(f"Expected normals [3,H,W], got {tuple(normal.shape)}")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Expected mask [1,H,W], got {tuple(mask.shape)}")

        resized_normal = self._resize_float(normal, size, "bilinear")
        resized_mask = self._resize_float(mask.float(), size, "nearest") > 0.5
        resized_normal, resized_mask = _normalize_normals(
            resized_normal, resized_mask, self.eps
        )
        sample[self.normal_key] = resized_normal.to(normal.dtype)
        sample[self.mask_key] = resized_mask
        return sample


@register("dataset_transform")
class ResizeSurfaceNormalsToMultiple:
    """Resize aligned normals inputs using Lotus-2's multiple-of-16 rule.

    Lotus-2 scales the shorter side down to its largest multiple of ``multiple``;
    it then rounds both independently scaled dimensions down to that multiple.
    The original resolution remains in ``meta.orig_res`` so the output writer
    can restore predictions to the evaluation image size.
    """

    def __init__(
        self,
        multiple=16,
        rgb_norm_key="rgb_norm",
        rgb_int_key="rgb_int",
        normal_key="normal_gt",
        mask_key="valid_mask",
        eps=1e-6,
    ):
        self.multiple = int(multiple)
        self.rgb_norm_key = str(rgb_norm_key)
        self.rgb_int_key = str(rgb_int_key)
        self.normal_key = str(normal_key)
        self.mask_key = str(mask_key)
        self.eps = float(eps)
        if self.multiple <= 0:
            raise ValueError("multiple must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def _target_size(self, height, width):
        min_side = min(int(height), int(width))
        if min_side < self.multiple:
            raise ValueError(
                f"Input dimensions {(height, width)} are too small for multiple "
                f"{self.multiple}"
            )

        scale = (min_side // self.multiple) * self.multiple / min_side
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        resized_height = max(
            self.multiple, (resized_height // self.multiple) * self.multiple
        )
        resized_width = max(
            self.multiple, (resized_width // self.multiple) * self.multiple
        )
        return resized_height, resized_width

    @staticmethod
    def _resize(value, size, mode):
        kwargs = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(value.unsqueeze(0).float(), **kwargs).squeeze(0)

    def __call__(self, sample):
        reference = sample.get(self.rgb_norm_key)
        if reference is None or not torch.is_tensor(reference) or reference.ndim != 3:
            raise ValueError(f"Expected '{self.rgb_norm_key}' as [C,H,W]")

        height, width = map(int, reference.shape[-2:])
        target_size = self._target_size(height, width)
        if target_size == (height, width):
            return sample

        for key in (self.rgb_norm_key, self.rgb_int_key):
            value = sample.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim != 3:
                raise ValueError(f"Expected '{key}' as [C,H,W]")
            if tuple(value.shape[-2:]) != (height, width):
                raise ValueError(
                    f"'{key}' has spatial shape {tuple(value.shape[-2:])}; "
                    f"expected {(height, width)}"
                )
            resized = self._resize(value, target_size, "bilinear")
            if not torch.is_floating_point(value):
                resized = resized.round().clamp(0, 255).to(value.dtype)
            else:
                resized = resized.to(value.dtype)
            sample[key] = resized

        normal = sample.get(self.normal_key)
        mask = sample.get(self.mask_key)
        if normal is None or mask is None:
            raise KeyError(
                f"ResizeSurfaceNormalsToMultiple requires '{self.normal_key}' "
                f"and '{self.mask_key}'"
            )
        if not torch.is_tensor(normal) or normal.ndim != 3 or normal.shape[0] != 3:
            raise ValueError(f"Expected '{self.normal_key}' as [3,H,W]")
        if tuple(normal.shape[-2:]) != (height, width):
            raise ValueError(
                f"'{self.normal_key}' has spatial shape {tuple(normal.shape[-2:])}; "
                f"expected {(height, width)}"
            )
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if not torch.is_tensor(mask) or mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Expected '{self.mask_key}' as [1,H,W]")
        if tuple(mask.shape[-2:]) != (height, width):
            raise ValueError(
                f"'{self.mask_key}' has spatial shape {tuple(mask.shape[-2:])}; "
                f"expected {(height, width)}"
            )

        resized_normal = self._resize(normal, target_size, "bilinear")
        resized_mask = self._resize(mask.float(), target_size, "nearest") > 0.5
        resized_normal, resized_mask = _normalize_normals(
            resized_normal, resized_mask, self.eps
        )
        sample[self.normal_key] = resized_normal.to(normal.dtype)
        sample[self.mask_key] = resized_mask
        return sample
