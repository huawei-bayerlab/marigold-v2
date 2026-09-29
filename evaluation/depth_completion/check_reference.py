"""Compare a run with the published per-sample results.

Only the samples both sides have are compared, so a few images per cell suffice to catch a wrong
checkpoint, missing data or changed preprocessing:

  bash evaluation/depth_completion/reproduce_table.sh subset      # runs a few images, then this
  python -m evaluation.depth_completion.check_reference --results-dir DIR

The statistic is the median per-sample ratio to the published value, bootstrapped; a row is
flagged when the interval lies entirely outside +/- --tolerance. Per-sample agreement is not
expected, see "Reproducibility" in REPRODUCING_DEPTH_COMPLETION.md. Exit code 0 when every
compared row is consistent, 1 when one differs, 2 when nothing is shared with the reference.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from evaluation.depth_completion.common import REPO, RESULTS_DIR
from evaluation.depth_completion.summarize import ROWS, SETS, load

REFERENCE_DIR = REPO / "evaluation" / "depth_completion" / "results_reference"


def compare(run, reference, seed=0):
    """(n, run mean, reference mean, median ratio, its 95% interval) over the shared samples."""
    shared = sorted(set(run) & set(reference))
    a = np.array([run[s] for s in shared])
    b = np.array([reference[s] for s in shared])
    ratio = a / np.where(b == 0, np.nan, b)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return len(shared), float("nan"), float("nan"), float("nan"), (0.0, np.inf)
    if ratio.size == 1:
        # One sample gives a ratio but no interval. The default subset runs exactly one tiled
        # sample, and its ratio still catches an order-of-magnitude error.
        return len(shared), a.mean(), b.mean(), float(ratio[0]), (0.0, np.inf)
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(ratio, ratio.size)) for _ in range(2000)]
    return (
        len(shared),
        a.mean(),
        b.mean(),
        float(np.median(ratio)),
        tuple(np.percentile(boot, [2.5, 97.5])),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    ap.add_argument(
        "--metric", default="mae", choices=["mae", "rmse", "absrel", "delta1"]
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="relative difference treated as equivalent (default 0.05)",
    )
    a = ap.parse_args()
    band = (1 - a.tolerance, 1 + a.tolerance)

    print(
        f"{a.metric.upper()} over the samples present in both, versus the published results\n"
    )
    print(
        f"  {'row':26s} {'dataset':11s} {'n':>5s} {'yours':>9s} {'published':>10s} "
        f"{'ratio':>7s} {'95% interval':>15s}  verdict"
    )
    flagged = compared = inconclusive = 0
    for label, tags in ROWS:
        for name, dataset in SETS:
            run = load(a.results_dir, tags[dataset], dataset, a.metric)
            ref = load(a.reference_dir, tags[dataset], dataset, a.metric)
            if not run or not ref:
                continue
            n, mine, published, ratio, (lo, hi) = compare(run, ref)
            # A single shared sample gives a ratio but no interval, so it can only catch a gross
            # error. Count it apart, so one such row never reads as a row that passed.
            if n < 2:
                verdict = "1 sample: gross errors only" if n == 1 else "nothing shared"
                inconclusive += 1
            elif lo > band[1] or hi < band[0]:
                verdict = "DIFFERS"
                flagged += 1
                compared += 1
            else:
                verdict = "consistent"
                compared += 1
            span = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(hi) else "--"
            ratio_s = f"{ratio:7.2f}" if np.isfinite(ratio) else f"{'--':>7s}"
            print(
                f"  {label:26s} {name:11s} {n:5d} {mine:9.4f} {published:10.4f} "
                f"{ratio_s} {span:>15s}  {verdict}"
            )
    # Nothing in common with the reference is not a pass: it usually means the run wrote
    # somewhere else, or under a different --tag, so say so instead of printing an empty table.
    tail = (
        f" {inconclusive} row(s) had too few shared samples to judge."
        if inconclusive
        else ""
    )
    if not compared:
        print(
            f"\n  (nothing to compare)\n\nNo results under {a.results_dir} share a tag and a "
            f"dataset with {a.reference_dir}.\nRun a few samples first, e.g. "
            f"`bash evaluation/depth_completion/reproduce_table.sh subset`, and check that --tag "
            f"matches\nthe reference file names.{tail}"
        )
        return 2
    if flagged:
        print(
            f"\n{flagged} of {compared} row(s) differ from the published results by more than "
            f"{100 * a.tolerance:.0f}%.{tail}"
        )
    else:
        print(
            f"\n{compared} row(s) checked, all consistent with the published results.{tail}"
        )
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
