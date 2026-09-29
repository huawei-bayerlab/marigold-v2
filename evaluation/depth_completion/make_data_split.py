"""Regenerate, or verify, the fixed KITTI-DC and DDAD evaluation subsets.

KITTI-DC (1000 images) and DDAD (3950) are too expensive to complete with every method, so each is
scored on 150 samples drawn evenly over the sorted sample ids:

    index = round(i * (N - 1) / (n - 1))   for i in 0 .. n-1

which is `np.round(np.linspace(0, N - 1, n))`. The draw has no random component, so the subset is a
property of the dataset rather than of a seed, and it always contains the first and last sample.
iBims-1 and NYUv2 are evaluated in full and need no file.

The committed lists under data_split/ are the authority; this script checks them against the data
and rebuilds them after a dataset re-export.

  python -m evaluation.depth_completion.make_data_split            # verify, exit 1 on a mismatch
  python -m evaluation.depth_completion.make_data_split --write    # rewrite the lists
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from evaluation.depth_completion.common import (
    DATASETS,
    REPO,
    dataset_root,
    sample_lines,
)

SUBSETS = {"kittidc": 150, "ddad": 150}


def even_subset(lines, n):
    """`n` entries spaced evenly over `lines`, endpoints included."""
    idx = np.round(np.linspace(0, len(lines) - 1, n)).astype(int)
    return [lines[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--write", action="store_true", help="rewrite the lists instead of checking"
    )
    a = ap.parse_args()

    spec = json.loads(DATASETS.read_text())
    failures = 0
    for dataset, n in SUBSETS.items():
        path = REPO / spec[dataset]["split_file"]
        pool = sample_lines(dataset_root(spec[dataset]))
        want = even_subset(pool, n)
        if a.write:
            path.write_text("\n".join(want) + "\n")
            print(
                f"{dataset}: wrote {len(want)} of {len(pool)} -> {path.relative_to(REPO)}"
            )
            continue
        have = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if have == want:
            print(f"{dataset}: {len(have)} of {len(pool)} samples, matches the rule")
        else:
            failures += 1
            extra = len(set(have) - set(want))
            print(
                f"{dataset}: MISMATCH -- file has {len(have)} lines, rule gives {len(want)}"
                f" ({extra} in the file are not in the rule's draw)"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
