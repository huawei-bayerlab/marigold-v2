#!/usr/bin/env python3
"""Average per-sample metrics per dataset into one summary JSON.

DIODE and ETH3D additionally get indoor/outdoor averages inferred from the
sample paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def find_metrics_files(metrics_dir: Path):
    """Return mapping dataset -> path to per-sample csv if available, else per-dataset csv."""
    mapping = {}
    if not metrics_dir.exists():
        raise FileNotFoundError(metrics_dir)

    for ds_dir in sorted(metrics_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        candidates = [
            ds_dir / "per_sample_metrics.csv",
            ds_dir / "per_sample.csv",
            ds_dir / "per_dataset_metrics.csv",
            ds_dir / "metrics.csv",
        ]
        for c in candidates:
            if c.exists():
                mapping[ds_dir.name] = c
                break
    return mapping


def infer_indoor_outdoor(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Indoor/outdoor boolean masks from a scene column or path substrings."""
    indoor_mask = pd.Series(False, index=df.index)
    outdoor_mask = pd.Series(False, index=df.index)

    col_candidates = [
        c for c in df.columns if c.lower() in {"scene", "location", "env", "split"}
    ]
    if col_candidates:
        col = col_candidates[0]
        vals = df[col].astype(str).str.lower()
        indoor_mask = vals.str.contains("indoor") | vals.str.contains("indoors")
        outdoor_mask = vals.str.contains("outdoor") | vals.str.contains("outdoors")
        return {"indoor": indoor_mask, "outdoor": outdoor_mask}

    path_cols = [
        c for c in df.columns if any(k in c.lower() for k in ("path", "file", "rgb"))
    ]
    for pc in path_cols:
        vals = df[pc].astype(str).str.lower()
        indoor_mask = (
            indoor_mask
            | vals.str.contains("/indoor")
            | vals.str.contains("indoor")
            | vals.str.contains("/indoors")
        )
        outdoor_mask = (
            outdoor_mask
            | vals.str.contains("/outdoor")
            | vals.str.contains("outdoor")
            | vals.str.contains("/outdoors")
        )
    return {"indoor": indoor_mask, "outdoor": outdoor_mask}


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean of numeric columns and return as dict."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return {}
    means = num.mean(axis=0, skipna=True).to_dict()
    return {k: float(v) for k, v in means.items()}


def summarize_metrics(metrics_dir: Path) -> Dict[str, Any]:
    files = find_metrics_files(metrics_dir)
    summary: Dict[str, Any] = {}

    for ds, csv_path in files.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: failed to read {csv_path}: {e}")
            continue

        ds_summary: Dict[str, Any] = {}
        ds_summary["full"] = summarize_dataframe(df)

        if ds.lower().startswith("diode") or ds.lower().startswith("eth3d"):
            masks = infer_indoor_outdoor(df)
            for split_name, mask in masks.items():
                if mask.any():
                    df_split = df[mask]
                    ds_summary[split_name] = summarize_dataframe(df_split)
                else:
                    ds_summary[split_name] = {}

        summary[ds] = ds_summary

    per_sample_paths = [p for p in files.values() if p.name.startswith("per_sample")]
    if per_sample_paths:
        dfs = []
        for p in per_sample_paths:
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                continue
        if dfs:
            big = pd.concat(dfs, ignore_index=True, sort=False)
            summary["ALL"] = {"full": summarize_dataframe(big)}

    return summary


def print_summary(summary: Dict[str, Any]):
    print("Summary of evaluation metrics:")
    for ds, parts in summary.items():
        print(f"\nDataset: {ds}")
        for key, metrics in parts.items():
            if not metrics:
                print(f"  {key}: <no numeric metrics>")
                continue
            metrics_str = ", ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items()))
            print(f"  {key}: {metrics_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize evaluation metrics across datasets"
    )
    parser.add_argument(
        "metrics_dir",
        type=Path,
        help="Path to the metrics directory (per-run metrics folder)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output JSON file for the summary"
    )
    args = parser.parse_args()

    summary = summarize_metrics(args.metrics_dir)
    print_summary(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to: {args.out}")


if __name__ == "__main__":
    main()
