#!/usr/bin/env python3
"""Run `eegnet_neurosoft_sweep` sequentially on a single GPU.

Example:
    uv run python -m foundry.tools.run_eegnet_neurosoft_sweep_sequential \
        --start 0 --end 38 --gpu-id 0
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="eegnet_neurosoft_sweep",
        help="Hydra experiment name.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First session index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=38,
        help="Last session index (inclusive).",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="Value for CUDA_VISIBLE_DEVICES (e.g., 0).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if a session run fails.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    return parser


def _run_one(
    repo_root: Path,
    experiment: str,
    session_index: int,
    env: dict[str, str],
    extra_overrides: list[str],
) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        f"experiment={experiment}",
        f"hyperparameters.session_index={session_index}",
        "hydra/launcher=basic",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "stage.skip=true",
        *extra_overrides,
    ]
    print(f"\n=== Session {session_index} ===")
    print(" ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return int(completed.returncode)


def main() -> int:
    args = _build_parser().parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start.")

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    failures: list[tuple[int, int]] = []
    for session_index in range(args.start, args.end + 1):
        exit_code = _run_one(
            repo_root=repo_root,
            experiment=args.experiment,
            session_index=session_index,
            env=env,
            extra_overrides=args.override,
        )
        if exit_code == 0:
            continue

        failures.append((session_index, exit_code))
        if not args.continue_on_error:
            print(
                f"Stopping at session {session_index} (exit code {exit_code}).",
                file=sys.stderr,
            )
            return exit_code

    if failures:
        print("\nFailed sessions:")
        for session_index, exit_code in failures:
            print(f"  - session {session_index}: exit code {exit_code}")
        return 1

    print("\nAll sessions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
