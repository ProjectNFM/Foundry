"""Select an exact, deterministic subset from a compiled downstream cell list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_cells(
    rows: list[dict],
    *,
    fraction: float,
    recordings: set[str],
    offset: int,
    count: int,
) -> list[dict]:
    """Return unchanged rows matching the requested fraction and recordings."""
    if offset < 0 or count < 1:
        raise ValueError(
            "offset must be non-negative and count must be positive"
        )
    matches = [
        row
        for row in rows
        if float(row["target_fraction"]) == float(fraction)
        and str(row["target_recording"]) in recordings
    ]
    selected = matches[offset : offset + count]
    if len(selected) != count:
        raise ValueError(
            f"requested {count} rows at offset {offset}, but only "
            f"{len(matches)} rows matched"
        )
    if len({str(row["cell_id"]) for row in selected}) != len(selected):
        raise ValueError("selected rows contain duplicate cell IDs")
    return selected


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--recording", action="append", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    selected = select_cells(
        rows,
        fraction=args.fraction,
        recordings=set(args.recording),
        offset=args.offset,
        count=args.count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
        for row in selected
    )
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {len(selected)} unchanged cells: {args.output}")


if __name__ == "__main__":
    main()
