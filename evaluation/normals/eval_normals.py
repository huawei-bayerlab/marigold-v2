#!/usr/bin/env python3
"""Angular-error metrics for surface-normal predictions (Marigold V1.1 protocol)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tabulate import tabulate
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_NAMES = (
    "mean_angular_error",
    "median_angular_error",
    "sub5_error",
    "sub7_5_error",
    "sub11_25_error",
    "sub22_5_error",
    "sub30_error",
)


def _resolve_dataset_config(config_path: Path):
    cfg = OmegaConf.load(str(config_path))
    filenames = cfg.get("filenames")
    if filenames is not None and not Path(str(filenames)).is_absolute():
        filename_path = Path(str(filenames))
        candidates = (REPO_ROOT / filename_path, config_path.parent / filename_path)
        cfg.filenames = str(
            next(
                (path for path in candidates if path.is_file()), candidates[0]
            ).resolve()
        )
    return cfg


def _read_entries(dataset_config: Path, base_data_dir: Path):
    cfg = _resolve_dataset_config(dataset_config)
    dataset_root = base_data_dir / str(cfg.dir)
    filelist = Path(str(cfg.filenames))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not filelist.is_file():
        raise FileNotFoundError(f"Dataset file list does not exist: {filelist}")

    entries = []
    with filelist.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            if len(columns) < 2:
                raise ValueError(
                    f"{filelist}:{line_number}: expected RGB and normal paths"
                )
            rgb_name, normal_name = columns[0], columns[1]
            normal_path = Path(normal_name)
            if not normal_path.is_absolute():
                normal_path = dataset_root / normal_path
            if not normal_path.is_file():
                raise FileNotFoundError(
                    f"{filelist}:{line_number}: missing normal file {normal_path}"
                )
            entries.append((rgb_name, normal_path))
    if not entries:
        raise ValueError(f"No samples found in {filelist}")
    return str(cfg.get("disp_name", dataset_config.stem)), filelist, entries


def _prediction_path(prediction_dir: Path, rgb_name: str) -> Path:
    rgb_path = Path(rgb_name)
    return prediction_dir / rgb_path.parent / f"{rgb_path.stem}.npy"


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
    return torch.from_numpy(chw).unsqueeze(0)


def _compute_cosine_error(normals_pred: torch.Tensor, normals_gt: torch.Tensor):
    if normals_pred.shape != normals_gt.shape:
        raise ValueError(
            f"Prediction/ground-truth shape mismatch: {normals_pred.shape} vs "
            f"{normals_gt.shape}"
        )
    gt = normals_gt.squeeze(0)
    pred = normals_pred.squeeze(0)
    valid = torch.linalg.vector_norm(gt, dim=0) > 0
    if not torch.any(valid):
        return torch.empty(0, dtype=torch.float32)
    cosine = F.cosine_similarity(pred[:, valid], gt[:, valid], dim=0)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).cpu()


def _metric_values(cosine_error: torch.Tensor) -> dict[str, float]:
    if cosine_error.numel() == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    values = {
        "mean_angular_error": torch.mean(cosine_error),
        # np.median as in Marigold V1.1 (averages the two middle values).
        "median_angular_error": np.median(cosine_error.numpy()),
        "sub5_error": (cosine_error < 5).float().mean() * 100,
        "sub7_5_error": (cosine_error < 7.5).float().mean() * 100,
        "sub11_25_error": (cosine_error < 11.25).float().mean() * 100,
        "sub22_5_error": (cosine_error < 22.5).float().mean() * 100,
        "sub30_error": (cosine_error < 30).float().mean() * 100,
    }
    return {name: round(float(values[name]), 4) for name in METRIC_NAMES}


class _MetricTracker:
    def __init__(self):
        self.total = {name: 0.0 for name in METRIC_NAMES}
        self.count = {name: 0 for name in METRIC_NAMES}

    def update(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            if np.isfinite(value):
                self.total[name] += value
                self.count[name] += 1

    def result(self) -> dict[str, float]:
        return {
            name: (
                round(self.total[name] / self.count[name], 4)
                if self.count[name]
                else float("nan")
            )
            for name in METRIC_NAMES
        }


def evaluate(
    *,
    prediction_dir: Path,
    dataset_config: Path,
    base_data_dir: Path,
    output_dir: Path,
    no_cuda: bool,
    max_samples: int | None,
    seed: int,
) -> dict[str, float | int | str]:
    del no_cuda  # metrics run on CPU
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    dataset_name, filename_ls_path, entries = _read_entries(
        dataset_config, base_data_dir
    )
    if max_samples is not None and max_samples < len(entries):
        indices = list(range(len(entries)))
        random.Random(seed).shuffle(indices)
        selected = set(indices[:max_samples])
        entries = [entry for index, entry in enumerate(entries) if index in selected]

    tracker = _MetricTracker()
    per_sample_path = output_dir / "per_sample_metrics.csv"
    evaluated = 0
    missing = 0
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", *METRIC_NAMES])
        for rgb_name, normal_path in tqdm(entries, desc=f"Evaluating {dataset_name}"):
            pred_path = _prediction_path(prediction_dir, rgb_name)
            if not pred_path.is_file():
                logging.warning("Can't find prediction: %s", pred_path)
                missing += 1
                continue
            normals_gt = _load_normals(normal_path)
            normals_pred = _load_normals(pred_path)
            values = _metric_values(_compute_cosine_error(normals_pred, normals_gt))
            tracker.update(values)
            writer.writerow([rgb_name, *(values[name] for name in METRIC_NAMES)])
            evaluated += 1

    if evaluated == 0:
        raise RuntimeError(
            f"No predictions were evaluated for {dataset_name}; looked under "
            f"{prediction_dir}"
        )

    metrics = tracker.result()
    result = {name: float(value) for name, value in metrics.items()}
    result.update(
        {
            "dataset": dataset_name,
            "prediction_dir": str(prediction_dir),
            "dataset_config": str(dataset_config),
            "evaluated_samples": evaluated,
            "missing_predictions": missing,
        }
    )
    text = (
        "Evaluation metrics:\n"
        f"  of predictions: {prediction_dir}\n"
        f"  on dataset: {dataset_name}\n"
        f"  with samples in: {filename_ls_path}\n"
        f"  evaluated samples: {evaluated}\n"
        f"  missing predictions: {missing}\n\n"
    )
    text += tabulate([list(metrics.keys()), list(metrics.values())])
    (output_dir / "eval_metrics.txt").write_text(text + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    logging.info("Evaluation metrics saved to %s", output_dir / "eval_metrics.txt")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--base_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    evaluate(
        prediction_dir=Path(args.prediction_dir).resolve(),
        dataset_config=Path(args.dataset_config).resolve(),
        base_data_dir=Path(args.base_data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        no_cuda=args.no_cuda,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
