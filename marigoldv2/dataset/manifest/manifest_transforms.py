import csv
import random
from pathlib import Path

from marigoldv2.core.registry import register


def _subsample(entries, max_samples, seed):
    if max_samples is None or int(max_samples) >= len(entries):
        return entries
    keys = list(entries.keys())
    random.Random(seed).shuffle(keys)
    keep = set(keys[: int(max_samples)])
    return {k: v for k, v in entries.items() if k in keep}


@register("manifest_transform")
class LoadRGBDepthPairsFromFileList:
    """Build a manifest from text files with one RGB/depth pair per line.

    Expected line format:
      <relative_or_absolute_rgb_path> <relative_or_absolute_depth_path>

    Paths that are not absolute are resolved against ``base_dir``.
    """

    def __init__(
        self,
        base_dir,
        filelist_path,
        max_samples=None,
        seed=42,
        skip_missing_files=True,
        key_prefix="pair",
    ):
        self.base_dir = Path(base_dir)
        self.filelist_path = Path(filelist_path)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.skip_missing_files = bool(skip_missing_files)
        self.key_prefix = str(key_prefix)

    def _resolve_path(self, raw_path):
        candidate = Path(str(raw_path).strip())
        if candidate.is_absolute():
            return candidate
        return self.base_dir / candidate

    def _infer_scene_camera_frame(self, rgb_path):
        try:
            rel = rgb_path.relative_to(self.base_dir)
            parts = rel.parts
        except Exception:
            parts = rgb_path.parts

        scene = parts[0] if len(parts) > 0 else "unknown_scene"
        camera = parts[-2] if len(parts) > 1 else "camera"
        frame = rgb_path.stem
        return scene, camera, frame

    def __call__(self, manifest):
        if not self.filelist_path.exists():
            raise ValueError(
                "LoadRGBDepthPairsFromFileList: file list does not exist: "
                f"{self.filelist_path}"
            )

        entries = {}
        with self.filelist_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                cols = stripped.split()
                if len(cols) < 2:
                    raise ValueError(
                        "LoadRGBDepthPairsFromFileList: expected at least 2 columns "
                        f"(rgb depth) at line {line_idx} in {self.filelist_path}"
                    )

                rgb_path = self._resolve_path(cols[0])
                depth_path = self._resolve_path(cols[1])

                if self.skip_missing_files and not (
                    rgb_path.exists() and depth_path.exists()
                ):
                    continue

                scene, camera, frame = self._infer_scene_camera_frame(rgb_path)
                key = f"{self.key_prefix}_{scene}_{camera}_{frame}_{len(entries):06d}"
                entries[key] = {
                    "scene": scene,
                    "camera": camera,
                    "frame": frame,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                }

        if not entries:
            raise ValueError(
                "LoadRGBDepthPairsFromFileList: no valid samples found in "
                f"{self.filelist_path}"
            )

        manifest.update(_subsample(entries, self.max_samples, self.seed))
        return manifest


@register("manifest_transform")
class LoadHypersimManifest:
    """Load a CSV manifest with columns scene,camera,frame,rgb_path,depth_path,mask_path.

    Paths may be absolute or relative to ``base_dir``. Used for the zero-shot
    depth benchmarks, whose manifests are generated from the V1 split files.
    """

    def __init__(
        self,
        manifest_csv,
        base_dir=None,
        max_samples=None,
        seed=42,
        skip_missing_files=True,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.base_dir = (
            Path(base_dir) if base_dir is not None else self.manifest_csv.parent
        )
        self.max_samples = max_samples
        self.seed = int(seed)
        self.skip_missing_files = bool(skip_missing_files)

    def _to_abs_path(self, path_str):
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def __call__(self, manifest):
        if not self.manifest_csv.exists():
            raise ValueError(
                f"LoadHypersimManifest: manifest CSV does not exist: {self.manifest_csv}"
            )

        entries = {}
        with self.manifest_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "scene",
                "camera",
                "frame",
                "rgb_path",
                "depth_path",
                "mask_path",
            }
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    "LoadHypersimManifest: CSV is missing required columns. "
                    f"Found={reader.fieldnames}, required={sorted(required)}"
                )

            for idx, row in enumerate(reader):
                rgb_path = self._to_abs_path(row["rgb_path"])
                depth_path = self._to_abs_path(row["depth_path"])
                mask_path = self._to_abs_path(row["mask_path"])

                if self.skip_missing_files:
                    if not (
                        rgb_path.exists() and depth_path.exists() and mask_path.exists()
                    ):
                        continue

                scene = row["scene"]
                camera = row["camera"]
                frame = row["frame"]
                key = f"hypersim_{scene}_{camera}_{frame}_{idx:06d}"
                entries[key] = {
                    "scene": scene,
                    "camera": camera,
                    "frame": frame,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "mask_path": str(mask_path),
                }

        if not entries:
            raise ValueError(
                f"LoadHypersimManifest: no samples found in {self.manifest_csv}"
            )

        manifest.update(_subsample(entries, self.max_samples, self.seed))
        return manifest


@register("manifest_transform")
class LoadMarigoldHypersimManifest:
    """Load the Marigold V1 Hypersim split metadata CSV.

    Reads ``<base_dir>/<split>/filename_meta_<split>.csv`` (or ``metadata_csv``)
    with columns scene_name,camera_name,frame_id,rgb_path,depth_path plus the
    depth/RGB range and invalid-ratio statistics used for filtering.
    """

    def __init__(
        self,
        base_dir,
        split="train",
        metadata_csv=None,
        max_samples=None,
        seed=42,
        skip_missing_files=True,
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        if metadata_csv is None:
            metadata_csv = (
                self.base_dir / self.split / f"filename_meta_{self.split}.csv"
            )
        self.metadata_csv = Path(metadata_csv)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.skip_missing_files = bool(skip_missing_files)

    @staticmethod
    def _is_true(value):
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def __call__(self, manifest):
        if not self.metadata_csv.exists():
            raise ValueError(
                "LoadMarigoldHypersimManifest: metadata CSV does not exist: "
                f"{self.metadata_csv}"
            )

        split_root = self.base_dir / self.split
        entries = {}

        # Filtering thresholds from Marigold V1 issue #158.
        min_depth_range = 0.5  # meters
        min_rgb_range = 10  # 0-255
        max_invalid_ratio = 1e-3

        with self.metadata_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "scene_name",
                "camera_name",
                "frame_id",
                "rgb_path",
                "depth_path",
            }
            # These columns are needed for filtering:
            filter_cols = {
                "depth_min",
                "depth_max",
                "rgb_min",
                "rgb_max",
                "invalid_ratio",
            }
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    "LoadMarigoldHypersimManifest: CSV is missing required columns. "
                    f"Found={reader.fieldnames}, required={sorted(required)}"
                )
            if not filter_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    "LoadMarigoldHypersimManifest: CSV is missing required filter columns. "
                    f"Found={reader.fieldnames}, required filter columns={sorted(filter_cols)}"
                )

            for idx, row in enumerate(reader):
                rgb_rel = row["rgb_path"]
                depth_rel = row["depth_path"]

                rgb_path = Path(rgb_rel)
                if not rgb_path.is_absolute():
                    rgb_path = split_root / rgb_path

                depth_path = Path(depth_rel)
                if not depth_path.is_absolute():
                    depth_path = split_root / depth_path

                if self.skip_missing_files and not (
                    rgb_path.exists() and depth_path.exists()
                ):
                    continue

                # Filtering logic
                try:
                    depth_min = float(row["depth_min"])
                    depth_max = float(row["depth_max"])
                    rgb_min = float(row["rgb_min"])
                    rgb_max = float(row["rgb_max"])
                    invalid_ratio = float(row["invalid_ratio"])
                except Exception:
                    continue  # skip rows with missing or malformed values

                if (depth_max - depth_min) < min_depth_range:
                    continue
                if (rgb_max - rgb_min) < min_rgb_range:
                    continue
                if invalid_ratio > max_invalid_ratio:
                    continue

                scene = row["scene_name"]
                camera = row["camera_name"]
                frame = row["frame_id"]
                public_ok = self._is_true(row.get("included_in_public_release", "True"))
                key = (
                    f"marigold_hypersim_{self.split}_{scene}_{camera}_{frame}_{idx:06d}"
                )
                entries[key] = {
                    "scene": scene,
                    "camera": camera,
                    "frame": frame,
                    "split": self.split,
                    "included_in_public_release": public_ok,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                }

        if not entries:
            raise ValueError(
                "LoadMarigoldHypersimManifest: no valid samples found in "
                f"{self.metadata_csv}"
            )

        manifest.update(_subsample(entries, self.max_samples, self.seed))
        return manifest


@register("manifest_transform")
class LoadVKittiManifest:
    """Build a manifest from an image folder, optionally paired with depth.

    Scans ``<base_dir>/<split>`` (or ``<base_dir>``) for RGB files and looks
    for a depth file per image using the Virtual KITTI 2 layout conventions
    (``rgb/`` -> ``depth/``, ``rgb_`` -> ``depth_``, ``<stem>_depth.*``). With
    ``skip_missing_files=False`` images without depth are kept, which is how
    ``scripts/infer.py`` uses it for plain image folders.
    """

    def __init__(
        self,
        base_dir,
        split="train",
        scenes=None,
        max_samples=None,
        seed=42,
        skip_missing_files=True,
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        self.scenes = set(scenes) if scenes else None
        self.max_samples = max_samples
        self.seed = int(seed)
        self.skip_missing_files = bool(skip_missing_files)

    def _is_rgb(self, path: Path):
        s = path.suffix.lower()
        return (
            s in {".png", ".jpg", ".jpeg", ".webp"}
            and "depth" not in path.stem
            and "disp" not in path.stem
        )

    def _find_depth_candidate(self, rgb_path: Path):
        s = str(rgb_path)
        candidates = []

        try:
            rel = str(rgb_path.relative_to(self.base_dir))
        except Exception:
            rel = s

        # When rel starts with a top-level 'rgb/' component, drop it and
        # prefix with 'depth/' so candidates point under <base_dir>/depth/...
        if rel.startswith("rgb/"):
            rel_no_top = rel[len("rgb/") :]
            c1_rel = (
                str(Path("depth") / rel_no_top)
                .replace("/frames/rgb/", "/frames/depth/")
                .replace("rgb_", "depth_")
            )
            candidates.append(str(self.base_dir / c1_rel).rsplit(".", 1)[0] + ".png")
            c2_rel = (
                str(Path("depth") / rel_no_top)
                .replace("/rgb/", "/depth/")
                .replace("rgb_", "depth_")
            )
            candidates.append(str(self.base_dir / c2_rel).rsplit(".", 1)[0] + ".png")
        else:
            c2 = rel.replace("/rgb/", "/depth/").replace("rgb_", "depth_")
            candidates.append(str(self.base_dir / c2).rsplit(".", 1)[0] + ".png")

        # candidate 3: same directory, depth_ prefix
        stem = rgb_path.stem
        if stem.startswith("rgb_"):
            candidates.append(
                str(rgb_path.with_name(stem.replace("rgb_", "depth_") + ".png"))
            )
            candidates.append(
                str(rgb_path.with_name(stem.replace("rgb_", "depth_") + ".npy"))
            )

        # candidate 4: same directory, _depth suffix
        for ext in [".png", ".npy", ".pfm"]:
            candidates.append(str(rgb_path.with_name(f"{stem}_depth{ext}")))

        # check candidates
        for cstr in candidates:
            try:
                c = Path(cstr)
                if not c.exists():
                    continue
                if c.resolve() == rgb_path.resolve():
                    continue
                return c
            except Exception:
                continue

        return None

    def _candidate_strings(self, rgb_path: Path):
        """Candidate depth paths for an RGB file, for error messages."""
        s = str(rgb_path)
        try:
            rel = str(rgb_path.relative_to(self.base_dir))
        except Exception:
            rel = s

        cands = []
        c1 = rel.replace("/frames/rgb/", "/frames/depth/").replace("rgb_", "depth_")
        cands.append(str(self.base_dir / c1).rsplit(".", 1)[0] + ".png")
        c2 = rel.replace("/rgb/", "/depth/").replace("rgb_", "depth_")
        cands.append(str(self.base_dir / c2).rsplit(".", 1)[0] + ".png")

        stem = rgb_path.stem
        if stem.startswith("rgb_"):
            cands.append(
                str(rgb_path.with_name(stem.replace("rgb_", "depth_") + ".png"))
            )
            cands.append(
                str(rgb_path.with_name(stem.replace("rgb_", "depth_") + ".npy"))
            )

        for ext in [".png", ".npy", ".pfm"]:
            cands.append(str(rgb_path.with_name(f"{stem}_depth{ext}")))

        return cands

    def __call__(self, manifest):
        split_root = self.base_dir / self.split
        if not split_root.exists():
            split_root = self.base_dir  # rgb/ and depth/ directly under base_dir

        entries = {}
        rgb_paths = sorted(p for p in split_root.rglob("*") if p.is_file())
        idx = 0
        for rgb_path in rgb_paths:
            rgb_path = Path(rgb_path)
            if not self._is_rgb(rgb_path):
                continue

            rel_parts = rgb_path.relative_to(split_root).parts
            scene = rel_parts[0] if len(rel_parts) > 0 else "unknown_scene"
            camera = rel_parts[1] if len(rel_parts) > 1 else "camera"
            frame = rgb_path.stem

            if self.scenes is not None and scene not in self.scenes:
                continue

            depth_candidate = self._find_depth_candidate(rgb_path)
            if depth_candidate is None:
                if self.skip_missing_files:
                    continue
                depth_path = None
            else:
                depth_path = depth_candidate

            key = f"vkitti_{self.split}_{scene}_{camera}_{frame}_{idx:06d}"
            entries[key] = {
                "scene": scene,
                "camera": camera,
                "frame": frame,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path) if depth_path is not None else None,
            }
            idx += 1

        if not entries:
            sample_rgb = []
            for i, p in enumerate(sorted(split_root.rglob("*.jpg"))[:5]):
                ps = Path(p)
                sample_rgb.append((str(ps), self._candidate_strings(ps)))

            report_lines = [f"LoadVKittiManifest: no samples found in {split_root}"]
            for rgb, cands in sample_rgb:
                report_lines.append(f"  example rgb: {rgb}")
                for cand in cands:
                    report_lines.append(
                        f"    tried: {cand} (exists={Path(cand).exists()})"
                    )

            raise ValueError("\n".join(report_lines))

        manifest.update(_subsample(entries, self.max_samples, self.seed))
        return manifest
