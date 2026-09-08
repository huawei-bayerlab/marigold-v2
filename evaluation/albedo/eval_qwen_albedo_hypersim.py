#!/usr/bin/env python3
"""PSNR/SSIM/LPIPS for Hypersim albedo predictions (Marigold V1.1 IID protocol).

Prediction and ground truth are linear RGB, converted to sRGB with gamma 2.2
and compared at native resolution without alignment. PSNR and SSIM use
scikit-image; LPIPS uses the pinned ``lpips`` package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

from evaluation.src.util.metric import compute_iid_metric

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_NAMES = ("psnr", "ssim", "lpips")


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Marigold V1 ``img_linear2srgb`` (gamma 2.2)."""
    return value ** (1.0 / 2.2)


class Metric(ABC):
    """Abstract base class for image metrics."""

    def __init__(self, device):
        self.device = device

    def to(self, device):
        self.device = device
        return self

    @abstractmethod
    def __call__(self, pred, gt):
        pass


class _PSNR(Metric):
    def __call__(self, pred, gt):
        return torch.tensor(
            peak_signal_noise_ratio(
                pred.detach().cpu().numpy(),
                gt.detach().cpu().numpy(),
                data_range=1.0,
            ),
            device=self.device,
        )


class _SSIM(Metric):
    def __call__(self, pred, gt):
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        if pred_np.ndim == 4:
            pred_np = pred_np[0].transpose(1, 2, 0)
            gt_np = gt_np[0].transpose(1, 2, 0)
        else:
            pred_np = pred_np.transpose(1, 2, 0)
            gt_np = gt_np.transpose(1, 2, 0)
        return torch.tensor(
            structural_similarity(
                gt_np,
                pred_np,
                data_range=1.0,
                channel_axis=-1,
            ),
            device=self.device,
        )


class _LPIPS(Metric):
    def __init__(self, device, net_type):
        super().__init__(device)
        self.model = lpips.LPIPS(net=net_type).to(device).eval()

    def __call__(self, pred, gt):
        pred = pred.float().clamp(0, 1) * 2 - 1
        gt = gt.float().clamp(0, 1) * 2 - 1
        return self.model(pred, gt, normalize=False).mean()


def _read_entries(filelist: Path, base_dir: Path) -> list[tuple[str, Path]]:
    entries = []
    with filelist.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            if len(columns) < 2:
                raise ValueError(
                    f"{filelist}:{line_number}: expected RGB and albedo paths"
                )
            rgb_name, albedo_name = columns[:2]
            rgb_path = Path(rgb_name)
            albedo_path = Path(albedo_name)
            if not rgb_path.is_absolute():
                rgb_path = base_dir / rgb_path
            if not albedo_path.is_absolute():
                albedo_path = base_dir / albedo_path
            if not rgb_path.is_file() or not albedo_path.is_file():
                missing = [
                    str(path) for path in (rgb_path, albedo_path) if not path.is_file()
                ]
                raise FileNotFoundError(f"{filelist}:{line_number}: missing {missing}")
            entries.append((rgb_name, albedo_path))
    if not entries:
        raise ValueError(f"No samples found in {filelist}")
    return entries


def _prediction_path(prediction_dir: Path, rgb_name: str) -> Path:
    rgb_path = Path(rgb_name)
    return prediction_dir / rgb_path.parent / f"{rgb_path.stem}.npy"


def _load_chw(path: Path) -> torch.Tensor:
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB array in {path}, got {array.shape}")
    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Expected three RGB channels in {path}, got {array.shape}")
    return torch.from_numpy(chw.copy())


def _prepare_pair(
    pred_path: Path, gt_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_linear = _load_chw(pred_path)
    gt_linear = _load_chw(gt_path)
    if pred_linear.shape[-2:] != gt_linear.shape[-2:]:
        raise ValueError(
            "Native-resolution evaluation requires prediction and ground truth "
            f"to have identical spatial shapes, got prediction {tuple(pred_linear.shape)} "
            f"and ground truth {tuple(gt_linear.shape)} for {pred_path}"
        )

    valid = torch.isfinite(gt_linear)
    zero_albedo = (torch.nan_to_num(gt_linear, nan=0.0) == 0).all(dim=0, keepdim=True)
    valid &= ~zero_albedo.expand_as(valid)
    gt_linear = torch.nan_to_num(gt_linear, nan=0.0, posinf=0.0, neginf=0.0)

    pred_srgb = _linear_to_srgb(pred_linear.clamp_min(0.0))
    gt_srgb = _linear_to_srgb(gt_linear.clamp_min(0.0))
    return pred_srgb, gt_srgb, valid


def _score(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    metrics: dict[str, torch.nn.Module],
) -> dict[str, float]:
    values = {}
    for name in METRIC_NAMES:
        metric = metrics[name]
        value = compute_iid_metric(
            pred.clone(),
            gt.clone(),
            "albedo",
            name,
            metric,
            valid_mask=valid.clone(),
        )
        values[name] = float(value)
    return values


def evaluate(
    *,
    prediction_dir: Path,
    base_data_dir: Path,
    filelist: Path,
    output_dir: Path,
    max_samples: int | None,
    seed: int,
    device_name: str,
    lpips_net: str,
) -> dict[str, float | int | str]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )
    entries = _read_entries(filelist, base_data_dir)
    if max_samples is not None and max_samples < len(entries):
        indices = list(range(len(entries)))
        random.Random(seed).shuffle(indices)
        selected = set(indices[:max_samples])
        entries = [entry for index, entry in enumerate(entries) if index in selected]

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    metrics = {
        "psnr": _PSNR(device),
        "ssim": _SSIM(device),
        "lpips": _LPIPS(device, lpips_net),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metric_names = METRIC_NAMES
    per_sample_path = output_dir / "per_sample_metrics.csv"
    totals = {name: 0.0 for name in all_metric_names}
    finite_counts = {name: 0 for name in all_metric_names}
    evaluated = 0
    missing = 0
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", *all_metric_names])
        for rgb_name, gt_path in tqdm(entries, desc="Evaluating Hypersim albedo"):
            pred_path = _prediction_path(prediction_dir, rgb_name)
            if not pred_path.is_file():
                missing += 1
                continue
            pred, gt, valid = _prepare_pair(pred_path, gt_path)
            values = _score(
                pred.to(device),
                gt.to(device),
                valid.to(device),
                metrics,
            )
            writer.writerow([rgb_name, *(values[name] for name in all_metric_names)])
            for name, value in values.items():
                if math.isfinite(value):
                    totals[name] += value
                    finite_counts[name] += 1
            evaluated += 1

    if evaluated == 0:
        raise RuntimeError(f"No predictions found below {prediction_dir}")
    averages = {
        name: (
            totals[name] / finite_counts[name]
            if finite_counts[name] > 0
            else float("nan")
        )
        for name in all_metric_names
    }
    result = {
        **averages,
        "dataset": "hypersim_iid_lighting_test",
        "prediction_dir": str(prediction_dir),
        "filelist": str(filelist),
        "evaluated_samples": evaluated,
        "missing_predictions": missing,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = "\n".join(
        [
            "Hypersim albedo metrics",
            f"evaluated samples: {evaluated}",
            f"missing predictions: {missing}",
            *(f"{name}: {averages[name]:.6f}" for name in all_metric_names),
        ]
    )
    (output_dir / "eval_metrics.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument(
        "--base_data_dir",
        default=str(
            Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
            / "datasets/marigold_train_albedo"
        ),
    )
    parser.add_argument(
        "--filelist",
        default=str(REPO_ROOT / "evaluation/data_split/hypersim_iid/hypersim_test.txt"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="Torch device, or auto (default)."
    )
    parser.add_argument(
        "--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        prediction_dir=Path(args.prediction_dir).resolve(),
        base_data_dir=Path(args.base_data_dir).resolve(),
        filelist=Path(args.filelist).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        max_samples=args.max_samples,
        seed=args.seed,
        device_name=args.device,
        lpips_net=args.lpips_net,
    )


if __name__ == "__main__":
    main()
