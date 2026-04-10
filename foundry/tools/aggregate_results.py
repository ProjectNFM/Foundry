#!/usr/bin/env python3
"""Aggregate per-job results.csv files from a Hydra sweep into a single CSV.

Usage:
    uv run python -m foundry.tools.aggregate_results \
        outputs/runs/EEGNET_NEUROSOFT_SWEEP/*/results.csv \
        -o eegnet_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def aggregate(csv_paths: list[Path], output: Path) -> int:
    fieldnames: list[str] | None = None
    rows: list[dict] = []

    for p in sorted(csv_paths):
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

    if not rows or fieldnames is None:
        print("No results found.", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Per-job results.csv files (supports shell glob).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("eegnet_results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    sys.exit(aggregate(args.csv_files, args.output))


if __name__ == "__main__":
    main()
