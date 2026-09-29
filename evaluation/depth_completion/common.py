"""Shared by the depth-completion scripts: split loading, sample loading, metrics, result files.

Every row of the table is scored through `compute_metrics`, so rows differ only in how the depth
was predicted. Kept to numpy and PIL, so it imports without the model stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO / "assets")).expanduser()
DATASETS = REPO / "evaluation" / "depth_completion" / "datasets.json"
RESULTS_DIR = REPO / "output" / "eval_runs" / "depth_completion"
MIN_VALID_DEPTH_M = 1e-3


def dataset_root(spec):
    """Where a dataset lives, resolved against $DEPTH_ASSETS_DIR like every other asset."""
    root = spec["dataset_root"]
    return (
        ASSETS / root[len("assets/") :] if root.startswith("assets/") else REPO / root
    )


def sample_lines(root):
    """Every sample in a dataset as `rgb gt sparse`, ordered by sample id."""
    if not (root / "rgb").is_dir():
        raise SystemExit(
            f"no rgb/ under {root} -- see 'Getting the data' in "
            "REPRODUCING_DEPTH_COMPLETION.md"
        )
    stems = sorted(p.stem for p in (root / "rgb").iterdir() if p.suffix == ".png")
    return [f"rgb/{s}.png gt/{s}.npy sparse/{s}.npy" for s in stems]


def load_split(dataset):
    """(dataset_root, sample lines, spec) for one benchmark key in datasets.json.

    A dataset with a `split_file` is evaluated on that fixed subset; one without is evaluated on
    its complete test set. Either way the expected count in the spec is enforced, so a truncated
    download fails here rather than silently producing a table row over fewer samples.
    """
    spec = json.loads(DATASETS.read_text())[dataset]
    root = dataset_root(spec)
    if "split_file" in spec:
        source = spec["split_file"]
        lines = [ln for ln in (REPO / source).read_text().splitlines() if ln.strip()]
    else:
        source = f"{root}/rgb"
        lines = sample_lines(root)
    if len(lines) != spec["n"]:
        raise SystemExit(
            f"{dataset}: {source} yields {len(lines)} samples, expected {spec['n']}"
        )
    return root, lines, spec


def sample_id(line):
    """The sample id of a split line, without reading the sample from disk."""
    return Path(line.split()[0]).stem


def load_dc_sample(dataset_root, line):
    """One split line -> (sample id, rgb uint8, dense GT metres, provided sparse metres).

    GT and sparse are float32 metre maps with 0 / non-finite marking "no measurement"; both are
    read verbatim, at the benchmark's native resolution.
    """
    rgb_rel, gt_rel, sparse_rel = line.split()
    rgb = np.asarray(Image.open(dataset_root / rgb_rel).convert("RGB"), np.uint8)
    gt = np.load(dataset_root / gt_rel).astype(np.float32)
    sparse = np.load(dataset_root / sparse_rel).astype(np.float32)
    return Path(rgb_rel).stem, rgb, gt, sparse


def splat_sparse(sparse, nh, nw):
    """Move sparse points onto an (nh, nw) grid without interpolating their metric values.

    Points that land on the same target pixel collide and one wins.
    """
    h, w = sparse.shape
    out = np.zeros((nh, nw), np.float32)
    ys, xs = np.nonzero(sparse)
    if ys.size == 0:
        return out
    ny = np.clip(np.round(ys * (nh / h)).astype(np.intp), 0, nh - 1)
    nx = np.clip(np.round(xs * (nw / w)).astype(np.intp), 0, nw - 1)
    out[ny, nx] = sparse[ys, xs]
    return out


def valid_mask(depth_m):
    return np.isfinite(depth_m) & (depth_m > MIN_VALID_DEPTH_M)


def compute_metrics(pred_m, gt_m, valid, sid):
    """MAE / RMSE / AbsRel / delta1 in metres over the valid GT pixels, at native resolution."""
    valid = np.asarray(valid, dtype=bool)
    p = np.asarray(pred_m, dtype=np.float64)[valid]
    g = np.asarray(gt_m, dtype=np.float64)[valid]
    if p.size == 0:
        nan = float("nan")
        return {
            "sample_id": sid,
            "mae": nan,
            "rmse": nan,
            "absrel": nan,
            "delta1": nan,
            "n_valid": 0,
        }
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "sample_id": sid,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "absrel": float(np.mean(np.abs(diff) / g)),
        "delta1": float(np.mean(ratio < 1.25)),
        "n_valid": int(p.size),
    }


def open_results(out_dir, tag, dataset, offset, resume):
    """Open the shard's jsonl and return (path, file handle, ids already scored).

    One line per sample, so a killed run continues with --resume and shards concatenate.
    """
    out = Path(out_dir) / f"{tag}_{dataset}_o{offset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and out.exists():
        done = {
            json.loads(ln)["sample_id"]
            for ln in out.read_text().splitlines()
            if ln.strip()
        }
        print(f"[resume] {len(done)} sample(s) already in {out.name}", flush=True)
    return out, open(out, "a" if resume else "w"), done
