#!/usr/bin/env python3
"""Write per-dataset split files and configs for ``eval.py``.

Keeps only the rows of the V1 split files that have a prediction in the
prediction directory, so partial runs (``--max_samples``) can be scored.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import textwrap

import cv2
import numpy as np

SUPPORTED_DATASETS = ("diode", "eth3d", "kitti", "nyuv2", "scannet")


def _pred_name(rgb_rel: str, mode: str) -> str:
    p = Path(rgb_rel)
    stem = p.stem
    if mode == "id":
        out_name = f"pred_{stem}.npy"
    elif mode == "rgb_id":
        out_name = f"pred_{stem.replace('rgb_', '')}.npy"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return str(p.parent / out_name)


def _kitti_benchmark_crop(arr: np.ndarray) -> np.ndarray:
    kb_h, kb_w = 352, 1216
    h, w = arr.shape[-2:]
    if h < kb_h or w < kb_w:
        raise RuntimeError(
            f"Cannot KITTI-crop array of shape {(h, w)} to {(kb_h, kb_w)}"
        )
    top = int(h - kb_h)
    left = int((w - kb_w) / 2)
    return arr[top : top + kb_h, left : left + kb_w]


def _resize_pred_to_rgb_if_needed(
    pred_path: Path, rgb_path: Path, apply_kitti_bm_crop: bool = False
) -> bool:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image for shape check: {rgb_path}")
    h, w = rgb.shape[:2]

    pred = np.load(pred_path).astype(np.float32)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError(f"Expected 2D prediction at {pred_path}, got {pred.shape}")

    changed = False

    if apply_kitti_bm_crop:
        if pred.shape != (352, 1216):
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                changed = True
            if pred.shape != (352, 1216):
                pred = _kitti_benchmark_crop(pred)
                changed = True
    else:
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            changed = True

    if changed:
        np.save(pred_path, pred.astype(np.float32, copy=False))
    return changed


def build_subsets(
    repo_root: Path,
    pred_root: Path,
    out_dir: Path,
    base_data_dir: Path,
    datasets: list[str] | None = None,
    resize_predictions: bool = True,
) -> None:
    split_root = repo_root / "evaluation" / "data_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = {
        "diode": {
            "data_root": base_data_dir / "diode",
            "split": split_root / "diode_depth" / "diode_val_all_filename_list.txt",
            "pred_dir": pred_root / "diode_rel_depth_qwen_768",
            "name_mode": "id",
            "cfg": """name: diode_depth
disp_name: diode_depth_subset
dir: diode
filenames: {subset_file}
""",
        },
        "eth3d": {
            "data_root": base_data_dir / "eth3d",
            "split": split_root / "eth3d_depth" / "eth3d_filename_list.txt",
            "pred_dir": pred_root / "eth3d_rel_depth_qwen_768",
            "name_mode": "id",
            "cfg": """name: eth3d_depth
disp_name: eth3d_depth_subset
dir: eth3d
filenames: {subset_file}
""",
        },
        "kitti": {
            "data_root": base_data_dir / "kitti",
            "split": split_root / "kitti_depth" / "eigen_test_files_with_gt.txt",
            "pred_dir": pred_root / "kitti_rel_depth_qwen_768",
            "name_mode": "id",
            "cfg": """name: kitti_depth
disp_name: kitti_depth_eigen_test_subset
dir: kitti
filenames: {subset_file}
kitti_bm_crop: true
valid_mask_crop: eigen
""",
        },
        "nyuv2": {
            "data_root": base_data_dir / "nyuv2",
            "split": split_root / "nyu_depth" / "labeled" / "filename_list_test.txt",
            "pred_dir": pred_root / "nyuv2_rel_depth_qwen_768",
            "name_mode": "rgb_id",
            "cfg": """name: nyu_depth
disp_name: nyu_depth_test_subset
dir: nyuv2
filenames: {subset_file}
eigen_valid_mask: true
""",
        },
        "scannet": {
            "data_root": base_data_dir / "scannet",
            "split": split_root
            / "scannet_depth"
            / "scannet_val_sampled_list_800_1.txt",
            "pred_dir": pred_root / "scannet_rel_depth_qwen_768",
            "name_mode": "id",
            "cfg": """name: scannet_depth
disp_name: scannet_depth_val_subset
dir: scannet
filenames: {subset_file}
""",
        },
    }

    selected = set(datasets) if datasets is not None else set(SUPPORTED_DATASETS)

    for ds, spec in dataset_specs.items():
        if ds not in selected:
            continue
        split_file = spec["split"]
        pred_dir = spec["pred_dir"]
        subset_file = out_dir / f"{ds}_subset.txt"
        cfg_file = out_dir / f"data_{ds}_subset.yaml"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file missing: {split_file}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory missing: {pred_dir}")

        kept = []
        resized = 0
        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                rgb_rel = cols[0]
                pred_rel = _pred_name(rgb_rel, spec["name_mode"])
                pred_path = pred_dir / pred_rel
                if pred_path.exists():
                    rgb_path = spec["data_root"] / rgb_rel
                    if resize_predictions:
                        if _resize_pred_to_rgb_if_needed(
                            pred_path=pred_path,
                            rgb_path=rgb_path,
                            apply_kitti_bm_crop=(ds == "kitti"),
                        ):
                            resized += 1
                    kept.append(line)

        if not kept:
            raise RuntimeError(f"No matching predictions found for dataset: {ds}")

        subset_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        cfg_text = textwrap.dedent(spec["cfg"]).format(subset_file=str(subset_file))
        cfg_file.write_text(cfg_text, encoding="utf-8")
        print(f"{ds}: kept {len(kept)} samples, resized {resized}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--pred_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--base_data_dir",
        type=Path,
        default=Path(
            os.environ.get(
                "MARIGOLD_DEPTH_EVAL_DATASET_ROOT",
                str(
                    Path(
                        os.environ.get(
                            "DEPTH_ASSETS_DIR",
                            str(Path(__file__).resolve().parents[2] / "assets"),
                        )
                    )
                    / "datasets"
                    / "marigold_depth_eval"
                ),
            )
        ),
        help="Depth benchmark root (or set MARIGOLD_DEPTH_EVAL_DATASET_ROOT).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset list (choices: diode,eth3d,kitti,nyuv2,scannet) or 'all'.",
    )
    parser.add_argument(
        "--resize_predictions",
        type=str,
        default="true",
        help="Whether to resize predictions to RGB/crop shape before eval subset generation (true/false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resize_predictions = args.resize_predictions.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if args.datasets.strip().lower() == "all":
        selected_datasets = list(SUPPORTED_DATASETS)
    else:
        selected_datasets = [
            d.strip().lower() for d in args.datasets.split(",") if d.strip()
        ]
        invalid = [d for d in selected_datasets if d not in SUPPORTED_DATASETS]
        if invalid:
            raise ValueError(
                f"Unsupported dataset(s): {', '.join(invalid)}. Supported: {', '.join(SUPPORTED_DATASETS)}"
            )
        if not selected_datasets:
            raise ValueError("No datasets provided via --datasets")

    build_subsets(
        repo_root=args.repo_root.resolve(),
        pred_root=args.pred_root.resolve(),
        out_dir=args.out_dir.resolve(),
        base_data_dir=args.base_data_dir.resolve(),
        datasets=selected_datasets,
        resize_predictions=resize_predictions,
    )


if __name__ == "__main__":
    main()
