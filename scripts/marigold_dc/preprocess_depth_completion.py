#!/usr/bin/env python3
"""Prepare Marigold-DC depth-completion evaluation datasets.

The transformations follow the upstream Marigold-DC dataset preparation
script: https://github.com/prs-eth/Marigold-DC.

The VPP4DC filtering helper is implemented with SciPy array operations here
instead of the upstream Numba helper, because Numba is not a project
dependency.  It is mathematically equivalent for the fixed 7x7 window used
by KITTI DC and DDAD.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import io
from scipy.ndimage import minimum_filter
from tqdm import tqdm

DATASETS = ("ibims1", "nyudepthv2", "kittidc", "ddad")
SEED = 2024
EXPECTED_COUNTS = {
    "ibims1": 100,
    "nyudepthv2": 654,
    "kittidc": 1000,
    "ddad": 3950,
}
NYU_INTRINSICS = np.array(
    [
        [5.1885790117450188e02 / 2.0, 0, 3.2558244941119034e02 / 2.0 - 8.0],
        [0, 5.1946961112127485e02 / 2.0, 2.5373616633400465e02 / 2.0 - 6.0],
        [0, 0, 1.0],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get(
                "OUTPUT_DIR",
                str(
                    Path(
                        os.environ.get(
                            "DEPTH_ASSETS_DIR",
                            Path(__file__).resolve().parents[2] / "assets",
                        )
                    )
                    / "datasets"
                    / "marigold_depth_completion"
                ),
            )
        ),
        help="Dataset root containing raw downloads and processed outputs.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated subset of ibims1,nyudepthv2,kittidc,ddad.",
    )
    parser.add_argument(
        "--clean-raw",
        action="store_true",
        help="Remove raw archives and extracted source data after validation.",
    )
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(DATASETS))
    if invalid:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(DATASETS)}"
        )
    if not selected:
        raise ValueError("At least one dataset must be selected")
    return list(dict.fromkeys(selected))


def output_dirs(root: Path, dataset: str) -> dict[str, Path]:
    output = root / dataset
    return {name: output / name for name in ("rgb", "sparse", "gt", "intrinsics")}


def ensure_output_dirs(root: Path, dataset: str) -> dict[str, Path]:
    directories = output_dirs(root, dataset)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def deterministic_select_k_true(
    valid_mask: np.ndarray, count: int, seed: int = SEED
) -> np.ndarray:
    """Match Marigold-DC's seeded ``np.random.shuffle`` selection exactly."""
    true_indices = np.argwhere(valid_mask)
    if count <= 0 or true_indices.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    count = min(count, len(true_indices))
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        np.random.shuffle(true_indices)
    finally:
        np.random.set_state(state)
    selected = np.zeros_like(valid_mask, dtype=bool)
    selected[tuple(zip(*true_indices[:count]))] = True
    return selected


def filter_heuristic_depth(
    depth: np.ndarray, *, window: int = 7, threshold: float = 1.5
) -> np.ndarray:
    """Apply the fixed VPP4DC local-window sparse-depth filter."""
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}")
    if window % 2 != 1:
        raise ValueError(f"Window must be odd, got {window}")

    positive = depth > 0
    positive_depth = np.where(positive, depth, np.inf)
    local_min = minimum_filter(
        positive_depth,
        size=(window, window),
        mode="constant",
        cval=np.inf,
    )
    local_min = np.where(positive, local_min, np.inf)
    neighborhood_min = minimum_filter(
        local_min,
        size=(window, window),
        mode="constant",
        cval=np.inf,
    )
    delta = depth - neighborhood_min
    return np.where(positive & (delta <= threshold), depth, 0).astype(
        depth.dtype, copy=False
    )


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def atomic_save_png(path: Path, image: Image.Image) -> None:
    temporary = path.with_name(f".{path.name}.part")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def atomic_save_intrinsics(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.part")
    np.savetxt(temporary, matrix)
    temporary.replace(path)


def load_depth_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) / 256


def load_intrinsics(path: Path) -> np.ndarray:
    matrix = np.asarray(np.loadtxt(path))
    if matrix.size == 9:
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {path}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite intrinsics in {path}")
    return matrix


def validate_output(root: Path, dataset: str) -> int:
    directories = output_dirs(root, dataset)
    missing = [str(path) for path in directories.values() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"Missing processed directories for {dataset}: {missing}")

    stems = {
        name: {path.stem for path in directory.iterdir() if path.is_file()}
        for name, directory in directories.items()
    }
    if not stems["rgb"]:
        raise RuntimeError(f"No processed samples found for {dataset}")
    expected_stems = stems["rgb"]
    for name in ("sparse", "gt", "intrinsics"):
        if stems[name] != expected_stems:
            raise RuntimeError(f"Processed file mismatch for {dataset}: rgb vs {name}")

    for stem in sorted(expected_stems):
        rgb_path = directories["rgb"] / f"{stem}.png"
        sparse = np.load(directories["sparse"] / f"{stem}.npy")
        gt = np.load(directories["gt"] / f"{stem}.npy")
        intrinsics = np.loadtxt(directories["intrinsics"] / f"{stem}.txt")
        if sparse.ndim != 2 or gt.ndim != 2:
            raise RuntimeError(f"Expected 2-D depth maps for {dataset}/{stem}")
        if sparse.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch for {dataset}/{stem}")
        if not np.isfinite(sparse).all() or not np.isfinite(gt).all():
            raise RuntimeError(f"Non-finite depth values for {dataset}/{stem}")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise RuntimeError(f"Invalid intrinsics for {dataset}/{stem}")
        with Image.open(rgb_path) as rgb:
            if rgb.size != (sparse.shape[1], sparse.shape[0]):
                raise RuntimeError(f"RGB/depth size mismatch for {dataset}/{stem}")
    return len(expected_stems)


def is_complete(root: Path, dataset: str) -> bool:
    try:
        count = validate_output(root, dataset)
    except (OSError, RuntimeError, ValueError):
        return False
    return count == EXPECTED_COUNTS[dataset]


def require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {description}: {path}")


def process_ibims1(root: Path) -> None:
    source = root / "ibims1_core_mat" / "ibims1_core_mat"
    mat_files = sorted(source.glob("*.mat"))
    if len(mat_files) != EXPECTED_COUNTS["ibims1"]:
        raise RuntimeError(
            f"Expected 100 iBims-1 MAT files in {source}, found {len(mat_files)}"
        )
    directories = ensure_output_dirs(root, "ibims1")
    for mat_path in tqdm(mat_files, desc="iBims-1"):
        basename = mat_path.stem
        data = io.loadmat(mat_path)["data"]
        rgb = np.asarray(data["rgb"][0][0])
        depth = np.asarray(data["depth"][0][0])
        calibration = np.asarray(data["calib"][0][0])
        mask_invalid = np.asarray(data["mask_invalid"][0][0])
        mask_transparent = np.asarray(data["mask_transp"][0][0])
        mask_missing = (depth != 0).astype(depth.dtype)
        valid = (mask_transparent * mask_invalid * mask_missing).astype(bool)
        gt = depth * valid
        sparse_mask = deterministic_select_k_true(valid, 1000)
        sparse = np.where(sparse_mask, gt, 0)
        atomic_save_npy(directories["gt"] / f"{basename}.npy", gt)
        atomic_save_npy(directories["sparse"] / f"{basename}.npy", sparse)
        atomic_save_png(directories["rgb"] / f"{basename}.png", Image.fromarray(rgb))
        atomic_save_intrinsics(
            directories["intrinsics"] / f"{basename}.txt", calibration.T
        )


def process_nyudepthv2(root: Path) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "NYUv2 preprocessing requires h5py; install the project dependencies"
        ) from exc

    image_file = root / "nyu_img_gt.h5"
    sparse_file = root / "nyu_pred_with_500.h5"
    require_file(image_file, "NYUv2 ground-truth HDF5")
    require_file(sparse_file, "NYUv2 sparse-depth HDF5")
    directories = ensure_output_dirs(root, "nyudepthv2")
    with (
        h5py.File(image_file, "r") as images,
        h5py.File(sparse_file, "r") as sparse_data,
    ):
        count = len(images["gt"])
        if count != EXPECTED_COUNTS["nyudepthv2"]:
            raise RuntimeError(f"Expected 654 NYUv2 samples, found {count}")
        if len(sparse_data["hints"]) != count:
            raise RuntimeError("NYUv2 RGB/GT and sparse HDF5 lengths differ")
        for index in tqdm(range(count), desc="NYUv2"):
            stem = f"{index:04d}"
            gt = np.asarray(images["gt"][index].squeeze())
            rgb = np.asarray(images["img"][index])
            sparse = np.asarray(sparse_data["hints"][index].squeeze())
            atomic_save_npy(directories["gt"] / f"{stem}.npy", gt)
            atomic_save_npy(directories["sparse"] / f"{stem}.npy", sparse)
            atomic_save_png(directories["rgb"] / f"{stem}.png", Image.fromarray(rgb))
            atomic_save_intrinsics(
                directories["intrinsics"] / f"{stem}.txt", NYU_INTRINSICS
            )


def _kitti_samples(root: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    source = root / "depth_selection" / "val_selection_cropped"
    groups = tuple(
        sorted((source / name).glob("*.png" if name != "intrinsics" else "*.txt"))
        for name in ("image", "groundtruth_depth", "velodyne_raw", "intrinsics")
    )
    if len({len(group) for group in groups}) != 1:
        raise RuntimeError(
            "KITTI DC source directories do not contain matching sample counts"
        )
    if len(groups[0]) != EXPECTED_COUNTS["kittidc"]:
        raise RuntimeError(f"Expected 1000 KITTI DC samples, found {len(groups[0])}")
    return groups


def process_kittidc(root: Path) -> None:
    image_files, gt_files, sparse_files, intrinsics_files = _kitti_samples(root)
    directories = ensure_output_dirs(root, "kittidc")
    for index, (image_path, gt_path, sparse_path, intrinsics_path) in enumerate(
        tqdm(
            zip(image_files, gt_files, sparse_files, intrinsics_files),
            total=len(image_files),
            desc="KITTI DC",
        )
    ):
        stem = f"{index:04d}"
        gt = load_depth_png(gt_path)
        sparse = filter_heuristic_depth(load_depth_png(sparse_path))
        atomic_save_npy(directories["gt"] / f"{stem}.npy", gt)
        atomic_save_npy(directories["sparse"] / f"{stem}.npy", sparse)
        with Image.open(image_path) as image:
            atomic_save_png(directories["rgb"] / f"{stem}.png", image.copy())
        atomic_save_intrinsics(
            directories["intrinsics"] / f"{stem}.txt",
            load_intrinsics(intrinsics_path),
        )


def process_ddad(root: Path) -> None:
    source = root / "pregenerated" / "val"
    groups = tuple(
        sorted((source / name).glob("*.png" if name != "intrinsics" else "*.txt"))
        for name in ("rgb", "gt", "hints", "intrinsics")
    )
    if len({len(group) for group in groups}) != 1:
        raise RuntimeError("DDAD source directories do not contain matching counts")
    stems = [{path.stem for path in group} for group in groups]
    if any(current != stems[0] for current in stems[1:]):
        raise RuntimeError("DDAD source directories do not contain matching names")
    if len(groups[0]) != EXPECTED_COUNTS["ddad"]:
        raise RuntimeError(f"Expected 3950 DDAD samples, found {len(groups[0])}")
    directories = ensure_output_dirs(root, "ddad")
    for index, (rgb_path, gt_path, hints_path, intrinsics_path) in enumerate(
        tqdm(zip(*groups), total=len(groups[0]), desc="DDAD")
    ):
        stem = f"{index:04d}"
        gt = load_depth_png(gt_path)
        sparse = filter_heuristic_depth(load_depth_png(hints_path))
        atomic_save_npy(directories["gt"] / f"{stem}.npy", gt)
        atomic_save_npy(directories["sparse"] / f"{stem}.npy", sparse)
        with Image.open(rgb_path) as image:
            atomic_save_png(directories["rgb"] / f"{stem}.png", image.copy())
        atomic_save_intrinsics(
            directories["intrinsics"] / f"{stem}.txt",
            load_intrinsics(intrinsics_path),
        )


def clean_raw(root: Path, datasets: Iterable[str]) -> None:
    raw_paths = {
        "ibims1": (root / "ibims1_core_mat.zip", root / "ibims1_core_mat"),
        "nyudepthv2": (root / "nyu_img_gt.h5", root / "nyu_pred_with_500.h5"),
        "kittidc": (root / "data_depth_selection.zip", root / "depth_selection"),
        "ddad": (root / "ddad_pregenerated.zip", root / "pregenerated"),
    }
    for dataset in datasets:
        for path in raw_paths[dataset]:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"[cleanup] {path}")


def process(root: Path, datasets: list[str], cleanup: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    processors = {
        "ibims1": process_ibims1,
        "nyudepthv2": process_nyudepthv2,
        "kittidc": process_kittidc,
        "ddad": process_ddad,
    }
    for dataset in datasets:
        if is_complete(root, dataset):
            print(f"[skip] {dataset}: validated {EXPECTED_COUNTS[dataset]} samples")
            continue
        print(f"[process] {dataset}")
        processors[dataset](root)
        count = validate_output(root, dataset)
        if count != EXPECTED_COUNTS[dataset]:
            raise RuntimeError(
                f"{dataset} produced {count} samples; expected "
                f"{EXPECTED_COUNTS[dataset]}"
            )
        print(f"[ready] {dataset}: validated {count} samples")
    if cleanup:
        clean_raw(root, datasets)


def main() -> None:
    args = parse_args()
    try:
        datasets = selected_datasets(args.datasets)
        process(args.root.expanduser().resolve(), datasets, args.clean_raw)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
