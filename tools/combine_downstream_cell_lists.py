"""Combine exact compiled cell subsets without changing cell identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def combine_cell_lists(groups: list[list[dict]]) -> list[dict]:
    rows = [row for group in groups for row in group]
    cell_ids = [str(row["cell_id"]) for row in rows]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("input lists contain duplicate cell IDs")
    override_keys = {
        tuple(sorted(item.split("=", 1)[0] for item in row["overrides"]))
        for row in rows
    }
    if len(override_keys) != 1:
        raise ValueError("input rows do not have matching Hydra override keys")
    return rows


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = combine_cell_lists([_load(path) for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} unchanged cells: {args.output}")


if __name__ == "__main__":
    main()
