#!/usr/bin/env python3
"""Angular metrics and Soft Angular Edge Error (SAEE) for Hypersim normals.

SAEE: GT normal edges are pixels with a 4-neighbour angular jump above a
threshold; each edge pixel scores the minimum angular error between the
prediction and any GT normal in a local k x k patch.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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

try:
    _NORMAL_VALIDATION = importlib.import_module(
        "marigoldv2.experiments.20260728_qwen_normals.validation"
    )
    normal_angular_errors_degrees = _NORMAL_VALIDATION.normal_angular_errors_degrees
    normal_metric_values = _NORMAL_VALIDATION.normal_metric_values
except (ImportError, OSError) as exc:
    # The validation module imports OpenCV; keep a copy of the formulas so the
    # metrics also run where libGL is missing.
    logging.warning(
        "Could not import validation normal metrics (%s); using equivalent fallback",
        exc,
    )

    def normal_angular_errors_degrees(pred, gt, mask=None, eps=1e-6):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if gt.ndim == 3:
            gt = gt.unsqueeze(0)
        pred = F.normalize(pred.float(), p=2, dim=1, eps=eps)
        gt_normalized = F.normalize(gt.float(), p=2, dim=1, eps=eps)
        valid = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
        valid &= torch.linalg.vector_norm(gt.float(), dim=1) > eps
        if mask is not None:
            if mask.ndim == 4:
                mask = mask[:, 0]
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0)
            valid &= mask.to(device=pred.device, dtype=torch.bool)
        dot = (pred * gt_normalized).sum(dim=1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(dot)), valid

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


def _read_entries(
    base_dir: Path, split: str, manifest: Path | None
) -> list[tuple[str, Path]]:
    filelist = manifest or (base_dir / f"hypersim_filtered_{split}.txt")
    if not filelist.is_file():
        raise FileNotFoundError(f"Hypersim normals manifest does not exist: {filelist}")
    entries: list[tuple[str, Path]] = []
    for line_no, raw in enumerate(filelist.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(f"{filelist}:{line_no}: expected RGB and normal paths")
        rgb_name, normal_name = columns
        normal_path = Path(normal_name)
        if not normal_path.is_absolute():
            normal_path = base_dir / normal_path
        if not normal_path.is_file():
            raise FileNotFoundError(
                f"{filelist}:{line_no}: missing normal file {normal_path}"
            )
        entries.append((rgb_name, normal_path))
    if not entries:
        raise ValueError(f"No entries found in {filelist}")
    return entries


def _select_entries(
    entries: list[tuple[str, Path]], max_samples: int | None, seed: int
) -> list[tuple[str, Path]]:
    if max_samples is None or max_samples >= len(entries):
        return entries
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    indices = list(range(len(entries)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:max_samples])
    return [entry for index, entry in enumerate(entries) if index in selected]


def _load_normals(path: Path) -> torch.Tensor:
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D normal array in {path}, got {array.shape}")
    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Expected three normal channels in {path}, got {array.shape}")
    return torch.from_numpy(chw)


def _prediction_path(prediction_root: Path, base_dir: Path, rgb_name: str) -> Path:
    rgb_path = Path(rgb_name)
    try:
        relative_parent = (base_dir / rgb_path).parent.relative_to(base_dir)
    except ValueError:
        relative_parent = rgb_path.parent
    return prediction_root / relative_parent / f"{rgb_path.stem}.npy"


def _normalize_pair(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}"
        )
    pred = pred.float()
    gt = gt.float()
    pred_finite = torch.isfinite(pred).all(dim=1)
    gt_finite = torch.isfinite(gt).all(dim=1)
    pred_norm = F.normalize(
        torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred)), dim=1, eps=eps
    )
    gt_norm = F.normalize(
        torch.where(torch.isfinite(gt), gt, torch.zeros_like(gt)), dim=1, eps=eps
    )
    valid = pred_finite & gt_finite & (torch.linalg.vector_norm(gt, dim=1) > eps)
    return pred_norm[0], gt_norm[0], valid[0]


def _angular_map(pred_norm: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
    dot = (pred_norm * gt_norm).sum(dim=0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def _normal_edge_mask(
    gt_norm: torch.Tensor,
    valid: torch.Tensor,
    threshold_deg: float,
    dilation_radius: int,
) -> torch.Tensor:
    """Detect GT normal discontinuities using 4-neighbour angular jumps."""
    edge = torch.zeros_like(valid, dtype=torch.bool)
    if gt_norm.shape[-2] > 1:
        pair_valid = valid[:-1, :] & valid[1:, :]
        jump = _angular_map(gt_norm[:, :-1, :], gt_norm[:, 1:, :])
        pair = pair_valid & (jump >= float(threshold_deg))
        edge[:-1, :] |= pair
        edge[1:, :] |= pair
    if gt_norm.shape[-1] > 1:
        pair_valid = valid[:, :-1] & valid[:, 1:]
        jump = _angular_map(gt_norm[:, :, :-1], gt_norm[:, :, 1:])
        pair = pair_valid & (jump >= float(threshold_deg))
        edge[:, :-1] |= pair
        edge[:, 1:] |= pair
    edge &= valid
    if dilation_radius > 0:
        size = 2 * int(dilation_radius) + 1
        edge = (
            F.max_pool2d(
                edge.float()[None, None],
                kernel_size=size,
                stride=1,
                padding=dilation_radius,
            )[0, 0]
            > 0
        )
        edge &= valid
    return edge


def soft_edge_angular_error(
    pred_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    valid: torch.Tensor,
    edge_mask: torch.Tensor,
    patch_size: int,
) -> float:
    """SEE-style minimum angular error around GT normal edges."""
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd and >= 1, got {patch_size}")
    if not torch.any(edge_mask):
        return float("nan")
    radius = patch_size // 2
    gt_patches = F.unfold(gt_norm.unsqueeze(0), kernel_size=patch_size, padding=radius)[
        0
    ].reshape(3, patch_size * patch_size, -1)
    valid_patches = (
        F.unfold(
            valid.float().unsqueeze(0).unsqueeze(0),
            kernel_size=patch_size,
            padding=radius,
        )[0]
        > 0.5
    )
    pred_center = pred_norm.reshape(3, 1, -1)
    dots = (pred_center * gt_patches).sum(dim=0).clamp(-1.0, 1.0)
    patch_angles = torch.rad2deg(torch.acos(dots))
    patch_angles = torch.where(
        valid_patches, patch_angles, torch.full_like(patch_angles, float("inf"))
    )
    min_angles = patch_angles.min(dim=0).values.reshape_as(valid)
    selected = min_angles[edge_mask & valid]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean().item())


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return round(float(np.mean(finite)), 4) if finite else float("nan")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_data_dir).resolve()
    raw_prediction_root = Path(args.prediction_dir).resolve()
    nested = raw_prediction_root / args.dataset_name
    prediction_root = nested if nested.is_dir() else raw_prediction_root
    entries = _select_entries(
        _read_entries(
            base_dir,
            args.split,
            Path(args.manifest).resolve() if args.manifest else None,
        ),
        args.max_samples,
        args.seed,
    )
    patch_sizes = [
        int(value.strip())
        for value in args.soft_edge_kernel_sizes.split(",")
        if value.strip()
    ]
    if not patch_sizes or any(value < 1 or value % 2 == 0 for value in patch_sizes):
        raise ValueError(
            "soft_edge_kernel_sizes must be a comma-separated list of odd positive integers"
        )
    patch_sizes = list(dict.fromkeys(patch_sizes))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_values: dict[str, list[float]] = {name: [] for name in NORMAL_METRIC_NAMES}
    edge_mean_values: list[float] = []
    soft_values: dict[int, list[float]] = {size: [] for size in patch_sizes}
    rows: list[dict[str, Any]] = []
    missing = 0
    invalid = 0

    for rgb_name, gt_path in tqdm(entries, desc=f"Evaluating {args.dataset_name}"):
        pred_path = _prediction_path(prediction_root, base_dir, rgb_name)
        if not pred_path.is_file():
            logging.warning("Missing prediction: %s", pred_path)
            missing += 1
            continue
        try:
            pred = _load_normals(pred_path)
            gt = _load_normals(gt_path)
            pred_norm, gt_norm, valid = _normalize_pair(pred, gt)
            standard = normal_metric_values(pred_norm, gt_norm, valid)
            if standard is None:
                invalid += 1
                continue
            angles = _angular_map(pred_norm, gt_norm)
            edge_mask = _normal_edge_mask(
                gt_norm,
                valid,
                threshold_deg=args.edge_angle_threshold_deg,
                dilation_radius=args.edge_dilate_radius,
            )
            edge_angles = angles[edge_mask & valid]
            edge_mean = (
                float(edge_angles.mean().item())
                if edge_angles.numel()
                else float("nan")
            )
            soft_by_size = {
                size: soft_edge_angular_error(
                    pred_norm, gt_norm, valid, edge_mask, size
                )
                for size in patch_sizes
            }
        except (OSError, ValueError, RuntimeError) as exc:
            logging.warning("Invalid prediction/GT pair for %s: %s", rgb_name, exc)
            invalid += 1
            continue

        for name, value in standard.items():
            metric_values[name].append(float(value))
        edge_mean_values.append(edge_mean)
        for size, value in soft_by_size.items():
            soft_values[size].append(value)
        rows.append(
            {
                "rgb_path": rgb_name,
                "prediction_path": str(pred_path),
                "gt_normal_path": str(gt_path),
                **standard,
                "gt_edge_pixels": int(edge_mask.sum().item()),
                "edge_mean_angular_error_deg": edge_mean,
                **{
                    f"soft_edge_angular_error_k{size}": value
                    for size, value in soft_by_size.items()
                },
            }
        )

    if not rows:
        raise RuntimeError("No valid Hypersim normals predictions were evaluated")

    summary: dict[str, Any] = {
        "dataset": args.dataset_name,
        "split": args.split,
        "prediction_dir": str(prediction_root),
        "evaluated_samples": len(rows),
        "missing_predictions": missing,
        "invalid_predictions": invalid,
        "aggregation": "mean of per-image metrics (validation-compatible)",
        "gt_normal_edge_angle_threshold_deg": float(args.edge_angle_threshold_deg),
        "gt_normal_edge_dilate_radius": int(args.edge_dilate_radius),
        **{name: _safe_mean(values) for name, values in metric_values.items()},
        "edge_mean_angular_error_deg": _safe_mean(edge_mean_values),
        "soft_edge_angular_error_by_kernel_size": {
            f"k{size}": _safe_mean(values) for size, values in soft_values.items()
        },
    }
    for size, values in soft_values.items():
        summary[f"soft_edge_angular_error_k{size}"] = _safe_mean(values)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    fieldnames = list(rows[0].keys())
    with (out_dir / "per_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Hypersim normals evaluation metrics:"]
    lines.extend(f"  {key}: {value}" for key, value in summary.items())
    (out_dir / "eval_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    print(f"Metrics saved to: {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--dataset_name", default="hypersim_test_normals_origres_qwen")
    repo_root = Path(__file__).resolve().parents[2]
    assets_dir = Path(
        os.environ.get("DEPTH_ASSETS_DIR", repo_root / "assets")
    ).expanduser()
    parser.add_argument(
        "--base_data_dir",
        default=str(assets_dir / "datasets" / "marigold_train_normals"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_angle_threshold_deg", type=float, default=5.0)
    parser.add_argument("--edge_dilate_radius", type=int, default=0)
    parser.add_argument("--soft_edge_kernel_sizes", default="1,3,5,7")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate(parse_args())
