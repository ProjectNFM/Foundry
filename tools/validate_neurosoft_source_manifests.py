#!/usr/bin/env python
"""Data-backed validation of NeuroSoft supervised source manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from foundry.data.source_manifest import VALID_FAMILIES
from generate_neurosoft_source_manifests import (
    DEFAULT_BATCH_SIZE,
    WINDOW_SECONDS,
    validate_manifests_data_backed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/neurosoft-phase0-audit.json"),
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("configs/tasks/neurosoft_acoustic_stim_8band.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(VALID_FAMILIES),
        help="Validate only this family; repeat to validate multiple families.",
    )
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    validate_manifests_data_backed(
        args.manifest_root.resolve(),
        data_root=args.data_root,
        audit_path=args.audit.resolve(),
        task_path=args.task.resolve(),
        batch_size=args.batch_size,
        window_seconds=args.window_seconds,
        families=None if args.family is None else tuple(args.family),
    )
    print(f"Data-backed validation passed: {args.manifest_root.resolve()}")


if __name__ == "__main__":
    main()
