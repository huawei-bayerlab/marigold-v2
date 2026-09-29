"""Print the depth-completion table from the per-sample results.

Reads <results-dir>/<tag>_<dataset>_o*.jsonl, by default under output/eval_runs/depth_completion/.
A cell without results prints "--" and a partial cell prints its sample count in brackets. Every
row records MAE, RMSE, AbsRel and delta1 per sample, so --metric selects the table.

Unless --no-bootstrap is given, the increments between rows are tested with a paired bootstrap
over the samples both rows share (see "Reproducibility" in REPRODUCING_DEPTH_COMPLETION.md).

  python -m evaluation.depth_completion.summarize
  python -m evaluation.depth_completion.summarize --metric rmse
  python -m evaluation.depth_completion.summarize --no-bootstrap
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from evaluation.depth_completion.common import DATASETS, RESULTS_DIR

# Our three rows. Tiling needs enough anchors per tile to be worth it, so NYUv2 (500 points per
# image) is left untiled and the tiled row reuses its high-res result.
ROWS = [
    (
        "Marigold-V2-DC (LoRA)",
        dict.fromkeys(("ibims1", "nyudepthv2", "kittidc", "ddad"), "dc_baseline"),
    ),
    (
        "+ high-res inference",
        dict.fromkeys(("ibims1", "nyudepthv2", "kittidc", "ddad"), "dc_hires"),
    ),
    (
        "+ tiled local adaptation",
        {
            "ibims1": "dc_tiled",
            "nyudepthv2": "dc_hires",
            "kittidc": "dc_tiled",
            "ddad": "dc_tiled",
        },
    ),
]
SETS = [
    ("iBims-1", "ibims1"),
    ("NYUv2", "nyudepthv2"),
    ("KITTI-DC", "kittidc"),
    ("DDAD", "ddad"),
]


def load(results_dir, tag, dataset, metric):
    """Per-sample metric, keyed by sample id so shards merge and a rerun replaces its own rows."""
    values = {}
    for path in sorted(glob.glob(f"{results_dir}/{tag}_{dataset}_o*.jsonl")):
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                values[row["sample_id"]] = row[metric]
    return values


def paired_bootstrap(a_vals, b_vals, resamples, seed):
    """Relative difference of the means of `a` against `b` over the samples both have.

    Returns (n, relative difference in %, its 95% interval, win rate in %), or None when fewer
    than two samples are shared. Each resample draws one set of sample ids and scores both rows
    on it, which keeps the pairing.
    """
    shared = sorted(set(a_vals) & set(b_vals))
    if len(shared) < 2:
        return None
    a = np.array([a_vals[s] for s in shared], float)
    b = np.array([b_vals[s] for s in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), (resamples, len(shared)))
    rel = a[idx].mean(1) / b[idx].mean(1) - 1.0
    return (
        len(shared),
        100.0 * (a.mean() / b.mean() - 1.0),
        tuple(100.0 * np.percentile(rel, [2.5, 97.5])),
        100.0 * float((a < b).mean()),
    )


def bootstrap_section(title, pairs, results_dir, metric, resamples, seed):
    """Print one block of paired comparisons, `pairs` being (label, a tags, b tags)."""
    lines = []
    for label, a_tags, b_tags in pairs:
        for name, dataset in SETS:
            # NYUv2's tiled cell reuses its high-res cell; nothing to compare there.
            if a_tags[dataset] == b_tags[dataset]:
                continue
            got = paired_bootstrap(
                load(results_dir, a_tags[dataset], dataset, metric),
                load(results_dir, b_tags[dataset], dataset, metric),
                resamples,
                seed,
            )
            if got is None:
                continue
            n, rel, (lo, hi), win = got
            # An interval straddling zero means the two rows are not separated by this data.
            verdict = "significant" if lo * hi > 0 else "not resolved"
            lines.append(
                f"  {label:34s} {name:11s} n={n:4d}  {rel:+6.1f}%  "
                f"[{lo:+5.1f}%, {hi:+5.1f}%]  win {win:3.0f}%  {verdict}"
            )
    if lines:
        print(f"\n{title}")
        print("\n".join(lines))


def cell(values, expected):
    if not values:
        return "--"
    suffix = "" if len(values) >= expected else f"({len(values)})"
    return f"{np.mean(list(values.values())):.4f}" + suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument(
        "--metric", default="mae", choices=["mae", "rmse", "absrel", "delta1"]
    )
    ap.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="print the table only, without the paired comparisons",
    )
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    counts = {k: v["n"] for k, v in json.loads(DATASETS.read_text()).items()}

    print(
        f"{a.metric.upper()} ({'higher' if a.metric == 'delta1' else 'lower'} is better)"
    )
    print(
        f"{'Method':30s}" + "".join(f"{f'{n} (n={counts[d]})':>18s}" for n, d in SETS)
    )
    for label, tags in ROWS:
        cells = [
            cell(load(a.results_dir, tags[d], d, a.metric), counts[d]) for _, d in SETS
        ]
        print(f"{label:30s}" + "".join(f"{c:>18s}" for c in cells))
    if a.no_bootstrap:
        return

    base, hires, tiled = (tags for _, tags in ROWS)
    bootstrap_section(
        f"Our increments, paired bootstrap ({a.resamples} resamples), negative favours the "
        f"later row:",
        [
            ("+ high-res over baseline", hires, base),
            ("+ tiled over high-res", tiled, hires),
        ],
        a.results_dir,
        a.metric,
        a.resamples,
        a.seed,
    )


if __name__ == "__main__":
    main()
