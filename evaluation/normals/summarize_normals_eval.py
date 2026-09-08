#!/usr/bin/env python3
"""Combine per-dataset normals ``metrics.json`` files into one JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.metrics_root)
    summary = {}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        summary[metrics_path.parent.name] = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

    if not summary:
        raise RuntimeError(f"No per-dataset metrics.json files found under {root}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
