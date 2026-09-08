#!/usr/bin/env python3
"""Soft Edge Error (SEE) on Hypersim depth predictions after affine alignment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from PIL import Image
except Exception:
    Image = None

from marigoldv2.core.builder import build_dataset, register_experiment_modules
from marigoldv2.core.registry import REGISTRY
from marigoldv2.dataset.dataloading.custom_collate import custom_collate
from marigoldv2.validation import metric


def _parse_kernel_sizes_csv(csv_text: str) -> list[int]:
    vals = []
    for tok in str(csv_text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except ValueError as e:
            raise ValueError(f"Invalid kernel size '{tok}' in see_kernel_sizes.") from e
        if k < 1:
            raise ValueError(f"Kernel sizes must be >= 1, got {k}")
        if k % 2 == 0:
            raise ValueError(f"Kernel sizes must be odd, got {k}")
        vals.append(k)

    if not vals:
        raise ValueError("see_kernel_sizes must contain at least one odd integer >= 1.")

    dedup = []
    seen = set()
    for k in vals:
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def _get_pred_name(
    rgb_basename: str, name_mode: str = "id", suffix: str = ".npy"
) -> str:
    if name_mode == "rgb_id":
        parts = rgb_basename.split("_", 1)
        pred_basename = "pred_" + (parts[1] if len(parts) > 1 else rgb_basename)
    elif name_mode == "i_d_rgb":
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif name_mode == "id":
        pred_basename = "pred_" + rgb_basename
    elif name_mode == "rgb_i_d":
        parts = rgb_basename.split("_")
        pred_basename = "pred_" + (
            "_".join(parts[1:]) if len(parts) > 1 else rgb_basename
        )
    elif name_mode in {"basename", "none"}:
        pred_basename = rgb_basename
    else:
        raise ValueError(f"Unknown name_mode: {name_mode}")
    return os.path.splitext(pred_basename)[0] + suffix


def _to_hwc_path_from_batch(batch: dict, file_path_key: str) -> Path:
    if "annotation" in batch and file_path_key in batch["annotation"]:
        return Path(str(batch["annotation"][file_path_key][0]))
    if file_path_key in batch:
        val = batch[file_path_key]
        if isinstance(val, (list, tuple)):
            return Path(str(val[0]))
        return Path(str(val))
    raise KeyError(f"Cannot find file path key '{file_path_key}' in batch")


def _prepare_runtime_cfg(output_dir: str, device: torch.device) -> None:
    runtime_cfg = OmegaConf.create(
        {
            "register_modules": [
                "marigoldv2.dataset.generic_dataset",
                "marigoldv2.dataset.dataloading.transform",
                "marigoldv2.dataset.manifest.manifest",
                "marigoldv2.dataset.manifest.manifest_transforms",
                "marigoldv2.dataset.util",
                "marigoldv2.validation.validate_steps",
            ],
            "paths": {"override_vis_dir": output_dir},
            "validation": {"file_path_key": "rgb_path"},
            "device": str(device),
        }
    )
    REGISTRY["cfg"] = runtime_cfg
    register_experiment_modules(runtime_cfg)


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _first_existing_file(candidates: Iterable[Path]) -> Path | None:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _resolve_prediction_dir(prediction_dir_arg: str, dataset_disp_name: str) -> Path:
    raw_dir = Path(prediction_dir_arg).resolve()
    nested_dataset_dir = raw_dir / dataset_disp_name
    if nested_dataset_dir.is_dir():
        return nested_dataset_dir
    return raw_dir


def _find_prediction_path(
    pred_dir: Path, rel_parent: Path, pred_stem: str
) -> Path | None:
    exts = (".npy", ".npz", ".png", ".tif", ".tiff")

    search_dirs: list[Path] = [pred_dir / rel_parent]
    if rel_parent != Path(""):
        search_dirs.append(pred_dir / rel_parent.name)
    search_dirs.append(pred_dir)

    dedup_dirs: list[Path] = []
    seen = set()
    for d in search_dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        dedup_dirs.append(d)

    candidates = []
    for d in dedup_dirs:
        for ext in exts:
            candidates.append(d / f"{pred_stem}{ext}")

    return _first_existing_file(candidates)


def _load_prediction_array(pred_path: Path) -> np.ndarray | None:
    suffix = pred_path.suffix.lower()
    if suffix == ".npy":
        return np.load(str(pred_path)).astype(np.float32)
    if suffix == ".npz":
        npz = np.load(str(pred_path))
        if len(npz.files) == 0:
            return None
        return np.asarray(npz[npz.files[0]], dtype=np.float32)

    if suffix in {".png", ".tif", ".tiff"}:
        if Image is None:
            raise ImportError("Pillow is required to read non-npy prediction files.")
        with Image.open(str(pred_path)) as img:
            return np.asarray(img, dtype=np.float32)

    return None


def _prediction_to_2d_tensor(
    pred_array: np.ndarray, device: torch.device
) -> torch.Tensor | None:
    pred = torch.from_numpy(np.asarray(pred_array, dtype=np.float32)).to(device)
    pred = torch.squeeze(pred)
    if pred.ndim == 2:
        return pred
    if pred.ndim == 3:
        if pred.shape[0] == 1:
            return pred[0]
        if pred.shape[-1] == 1:
            return pred[..., 0]
    return None


def _metric_depth_to_log_tensor(
    pred_metric_2d: torch.Tensor,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    if pred_metric_2d.ndim != 2:
        raise ValueError(
            f"Expected 2D metric depth tensor, got shape {tuple(pred_metric_2d.shape)}"
        )
    return torch.log(torch.clamp(pred_metric_2d.float(), min=float(log_eps)))


def _mask_outside_depth_quantiles(
    gt_depth_metric: torch.Tensor,
    valid_mask: torch.Tensor,
    low_q: float,
    high_q: float,
) -> torch.Tensor:
    """Invalidate pixels outside [low_q, high_q] GT depth percentiles."""
    finite_mask = torch.isfinite(gt_depth_metric)
    base_mask = valid_mask & finite_mask

    valid_depth = gt_depth_metric[base_mask]
    if valid_depth.numel() == 0:
        return base_mask

    low = float(torch.quantile(valid_depth, low_q / 100.0).item())
    high = float(torch.quantile(valid_depth, high_q / 100.0).item())

    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        return base_mask

    quantile_mask = (gt_depth_metric >= low) & (gt_depth_metric <= high)
    return base_mask & quantile_mask


def _fit_affine_least_squares(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    if x.numel() < 2:
        return 1.0, 0.0
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    try:
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    except RuntimeError:
        sol = torch.linalg.lstsq(A.cpu(), y.unsqueeze(1).cpu()).solution.to(A.device)
    a = float(sol[0, 0].item())
    b = float(sol[1, 0].item())
    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    return a, b


def _align_log_ransac(
    pred_log: torch.Tensor,
    gt_log: torch.Tensor,
    valid_mask: torch.Tensor,
    max_trials: int,
    residual_threshold: float,
) -> torch.Tensor:
    """Robustly align pred_log to gt_log using per-sample affine RANSAC fits."""
    out = pred_log.float()
    tgt = gt_log.float()
    vm = valid_mask.bool()

    if out.shape != tgt.shape:
        out = out.expand_as(tgt)
    if vm.shape != tgt.shape:
        if vm.ndim == tgt.ndim - 1:
            vm = vm.unsqueeze(0)
        vm = vm.expand_as(tgt)

    squeeze_back = False
    if out.ndim == 3:
        out = out.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        vm = vm.unsqueeze(0)
        squeeze_back = True
    elif out.ndim != 4:
        raise ValueError(f"Unsupported tensor ndim for RANSAC alignment: {out.ndim}")

    aligned = out.clone()
    eps_denom = 1e-12
    trials = max(1, int(max_trials))
    thresh = float(residual_threshold)

    for i in range(out.shape[0]):
        x = out[i][vm[i]].reshape(-1)
        y = tgt[i][vm[i]].reshape(-1)
        n = int(x.numel())
        if n < 2:
            continue

        best_inliers = None
        best_count = -1

        for _ in range(trials):
            idx = torch.randperm(n, device=x.device)[:2]
            x1, x2 = x[idx[0]], x[idx[1]]
            y1, y2 = y[idx[0]], y[idx[1]]
            denom = x2 - x1
            if torch.abs(denom) <= eps_denom:
                continue
            a = (y2 - y1) / denom
            b = y1 - a * x1
            residuals = torch.abs(a * x + b - y)
            inliers = residuals <= thresh
            count = int(inliers.sum().item())
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or int(best_inliers.sum().item()) < 2:
            a, b = _fit_affine_least_squares(x, y)
        else:
            a, b = _fit_affine_least_squares(x[best_inliers], y[best_inliers])

        aligned[i] = a * out[i] + b

    if squeeze_back:
        return aligned.squeeze(0)
    return aligned


def _align_log_to_depth(
    pred_log: torch.Tensor,
    gt_log: torch.Tensor,
    valid_mask: torch.Tensor,
    metric_depth_gt: torch.Tensor,
    depth_min: float,
    depth_max: float,
    method: str,
    ransac_max_trials: int,
    ransac_residual_threshold: float,
    log_eps: float = 1.0,
    depth_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "least_squares":
        return metric.align_scale_shift_log_to_depth(
            pred_log,
            gt_log,
            valid_mask=valid_mask,
            metric_depth_gt=metric_depth_gt,
            depth_min=depth_min,
            depth_max=depth_max,
            log_eps=log_eps,
            depth_eps=depth_eps,
        )

    if method != "ransac":
        raise ValueError(f"Unknown alignment method: {method}")

    aligned_pred_log = _align_log_ransac(
        pred_log=pred_log,
        gt_log=gt_log,
        valid_mask=valid_mask,
        max_trials=ransac_max_trials,
        residual_threshold=ransac_residual_threshold,
    )

    min_val = max(float(depth_eps), float(depth_min))
    max_val = float(depth_max)
    if max_val < min_val:
        max_val = min_val

    aligned_pred_depth = torch.exp(aligned_pred_log) - float(log_eps)
    aligned_pred_depth = torch.clamp(aligned_pred_depth, min=min_val, max=max_val)
    gt_depth = torch.clamp(metric_depth_gt.float(), min=float(depth_eps))
    return aligned_pred_depth, gt_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_config", required=True, help="Dataset config used for inference."
    )
    parser.add_argument(
        "--dataset_key", default="hypersim_test_origres_rel_log_depth_qwen"
    )
    parser.add_argument(
        "--prediction_dir",
        "--prediction_root",
        dest="prediction_dir",
        required=True,
        help=(
            "Folder containing prediction files. Supports either a direct prediction directory "
            "or a parent folder that contains a <dataset_disp_name>/ subfolder."
        ),
    )
    parser.add_argument(
        "--dataset_base_dir",
        default=os.environ.get(
            "MARIGOLD_DATASET_ROOT",
            str(
                Path(
                    os.environ.get(
                        "DEPTH_ASSETS_DIR",
                        str(Path(__file__).resolve().parent.parent.parent / "assets"),
                    )
                )
                / "datasets"
                / "marigold_train"
            ),
        ),
        help="Dataset root (or set MARIGOLD_DATASET_ROOT).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_path_key", default="rgb_path")
    parser.add_argument("--name_mode", default="id")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--depth_min", type=float, default=1e-5)
    parser.add_argument("--depth_max", type=float, default=65.0)
    parser.add_argument(
        "--alignment_method",
        type=str,
        default="least_squares",
        choices=["least_squares", "ransac"],
        help="Affine log-space alignment method used before metric computation.",
    )
    parser.add_argument(
        "--ransac_max_trials",
        type=int,
        default=1000,
        help="Maximum RANSAC trials when --alignment_method=ransac.",
    )
    parser.add_argument(
        "--ransac_residual_threshold",
        type=float,
        default=0.05,
        help="Inlier residual threshold in log-space when --alignment_method=ransac.",
    )
    parser.add_argument(
        "--valid_mask_clip_low",
        type=float,
        default=2.0,
        help="Lower percentile (GT depth) used to invalidate outlier pixels before alignment/metrics.",
    )
    parser.add_argument(
        "--valid_mask_clip_high",
        type=float,
        default=98.0,
        help="Upper percentile (GT depth) used to invalidate outlier pixels before alignment/metrics.",
    )
    parser.add_argument(
        "--prediction_space",
        type=str,
        default="auto",
        choices=["auto", "log", "ppd", "metric"],
        help=(
            "log: log-depth predictions (Marigold V2); ppd: log(1 + depth) as saved by "
            "Pixel-Perfect Depth; metric: metric depth; auto: 'ppd' if the name contains it."
        ),
    )
    parser.add_argument(
        "--see_kernel_sizes",
        type=str,
        default="1,3,5,7",
        help=(
            "Comma-separated SEE patch sizes used to compute/report soft_edge_error separately "
            "(e.g. '1,3,5,7')."
        ),
    )
    parser.add_argument(
        "--see_disparity_jump_threshold",
        type=float,
        default=0.05,
        help=(
            "Disparity-jump threshold used to extract SEE boundary pixels "
            "(default: 0.05)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _prepare_runtime_cfg(output_dir=str(out_dir), device=device)

    cfg = OmegaConf.load(str(Path(args.dataset_config).resolve()))
    if args.dataset_key in cfg:
        ds_cfg = cfg[args.dataset_key]
    elif (
        "dataset" in cfg
        and "vis" in cfg["dataset"]
        and args.dataset_key in cfg["dataset"]["vis"]
    ):
        ds_cfg = cfg["dataset"]["vis"][args.dataset_key]
    else:
        raise KeyError(
            f"Dataset key '{args.dataset_key}' not found in {args.dataset_config} "
            "(checked top-level and dataset.vis)."
        )

    dataset = build_dataset(ds_cfg)
    if (
        args.max_samples is not None
        and args.max_samples > 0
        and len(dataset) > args.max_samples
    ):
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(dataset), size=int(args.max_samples), replace=False
        ).tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
    )

    dataset_disp_name = str(ds_cfg.get("disp_name", args.dataset_key))
    pred_root = _resolve_prediction_dir(args.prediction_dir, dataset_disp_name)
    base_dir = Path(args.dataset_base_dir).resolve()

    prediction_space = str(args.prediction_space).lower()
    if prediction_space == "auto":
        detect_text = " ".join(
            [
                str(args.dataset_key).lower(),
                str(dataset_disp_name).lower(),
                str(args.prediction_dir).lower(),
            ]
        )
        use_ppd_alignment = "ppd" in detect_text
    else:
        use_ppd_alignment = prediction_space == "ppd"
    use_metric_predictions = prediction_space == "metric"

    see_kernel_sizes = _parse_kernel_sizes_csv(args.see_kernel_sizes)

    rows = []
    soft_edge_vals_by_kernel: dict[int, list[float]] = {k: [] for k in see_kernel_sizes}
    missing_predictions = 0
    invalid_predictions = 0

    for batch in tqdm(loader, desc="edge-metrics"):
        rgb_path = _to_hwc_path_from_batch(batch, args.file_path_key)
        rgb_basename = rgb_path.name
        pred_name = _get_pred_name(
            rgb_basename, name_mode=args.name_mode, suffix=".npy"
        )
        pred_stem = Path(pred_name).stem

        try:
            rel_parent = rgb_path.resolve().parent.relative_to(base_dir)
        except ValueError:
            rel_parent = Path("")
        pred_path = _find_prediction_path(pred_root, rel_parent, pred_stem)

        if pred_path is None:
            missing_predictions += 1
            continue

        pred_array = _load_prediction_array(pred_path)
        if pred_array is None:
            invalid_predictions += 1
            continue

        pred_like_2d = _prediction_to_2d_tensor(pred_array, device)
        if pred_like_2d is None:
            invalid_predictions += 1
            continue

        gt_log = batch["log_depth_m"].float().to(device)
        gt_depth_metric = batch["depth_m"].float().to(device)
        valid_mask = (
            batch.get("valid_mask", torch.isfinite(gt_depth_metric)).bool().to(device)
        )
        if use_ppd_alignment:
            gt_log_target = torch.log(gt_depth_metric + 1.0)
            align_log_eps = 1.0
        else:
            gt_log_target = gt_log
            align_log_eps = 1e-6

        if gt_log.ndim == 3:
            gt_log = gt_log.unsqueeze(0)
        if gt_depth_metric.ndim == 3:
            gt_depth_metric = gt_depth_metric.unsqueeze(0)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)

        valid_mask = _mask_outside_depth_quantiles(
            gt_depth_metric=gt_depth_metric,
            valid_mask=valid_mask,
            low_q=float(args.valid_mask_clip_low),
            high_q=float(args.valid_mask_clip_high),
        )

        h, w = gt_log.shape[-2], gt_log.shape[-1]
        if pred_like_2d.shape[-2:] != (h, w):
            pred_like_2d = torch.nn.functional.interpolate(
                pred_like_2d.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if use_metric_predictions:
            pred_log_like_2d = _metric_depth_to_log_tensor(pred_like_2d, log_eps=1e-6)
        else:
            pred_log_like_2d = pred_like_2d

        pred_log_like = pred_log_like_2d.unsqueeze(0).unsqueeze(0)

        aligned_pred_depth, gt_depth = _align_log_to_depth(
            pred_log_like,
            gt_log_target,
            valid_mask,
            metric_depth_gt=gt_depth_metric,
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            method=str(args.alignment_method),
            ransac_max_trials=int(args.ransac_max_trials),
            ransac_residual_threshold=float(args.ransac_residual_threshold),
            log_eps=float(align_log_eps),
        )

        soft_edge_by_kernel_v: dict[int, float] = {}
        for see_k in see_kernel_sizes:
            soft_edge_k = metric.soft_edge_error(
                aligned_pred_depth,
                gt_depth,
                valid_mask,
                patch_size=int(see_k),
                disparity_jump_threshold=float(args.see_disparity_jump_threshold),
            )
            se_k_v = float(
                soft_edge_k.item() if torch.is_tensor(soft_edge_k) else soft_edge_k
            )
            soft_edge_by_kernel_v[see_k] = se_k_v
            soft_edge_vals_by_kernel[see_k].append(se_k_v)

        rows.append(
            {
                "rgb_path": str(rgb_path),
                "prediction_path": str(pred_path),
                **{
                    f"soft_edge_error_k{k}": soft_edge_by_kernel_v[k]
                    for k in see_kernel_sizes
                },
            }
        )

    soft_edge_by_kernel_summary = {
        f"k{k}": _safe_mean(soft_edge_vals_by_kernel[k]) for k in see_kernel_sizes
    }

    summary = {
        "num_samples": int(len(rows)),
        "prediction_dir": str(pred_root),
        "prediction_space": (
            "ppd"
            if use_ppd_alignment
            else ("metric" if use_metric_predictions else "log")
        ),
        "alignment_method": str(args.alignment_method),
        "missing_predictions": int(missing_predictions),
        "invalid_predictions": int(invalid_predictions),
        "soft_edge_error_by_see_kernel_size": soft_edge_by_kernel_summary,
        "see_disparity_jump_threshold": float(args.see_disparity_jump_threshold),
    }

    json_path = out_dir / "hypersim_edge_metrics_summary.json"
    csv_path = out_dir / "hypersim_edge_metrics_per_sample.csv"
    txt_path = out_dir / "hypersim_edge_metrics_summary.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rgb_path",
                "prediction_path",
                *[f"soft_edge_error_k{k}" for k in see_kernel_sizes],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with txt_path.open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Summary JSON: {json_path}")
    print(f"Per-sample CSV: {csv_path}")
    print(f"Summary TXT: {txt_path}")


if __name__ == "__main__":
    main()
