"""Remove exact completed cells from a compiled downstream JSONL list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def exclude_cells(rows: list[dict], completed_rows: list[dict]) -> list[dict]:
    """Return rows not already completed, preserving source order and bytes."""
    source_ids = [str(row["cell_id"]) for row in rows]
    completed_ids = [str(row["cell_id"]) for row in completed_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source list contains duplicate cell IDs")
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("completed list contains duplicate cell IDs")
    unknown = set(completed_ids).difference(source_ids)
    if unknown:
        raise ValueError(
            "completed list contains cell IDs absent from source: "
            + ", ".join(sorted(unknown))
        )
    completed = set(completed_ids)
    return [row for row in rows if str(row["cell_id"]) not in completed]


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--completed", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load(args.input)
    completed_rows = [row for path in args.completed for row in _load(path)]
    pending = exclude_cells(rows, completed_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            )
            + "\n"
            for row in pending
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(pending)} pending cells: {args.output}")


if __name__ == "__main__":
    main()
