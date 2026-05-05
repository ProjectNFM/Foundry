"""Launch a W&B hyperparameter sweep with one agent per GPU.

Reads a sweep configuration YAML from ``configs/sweep/``, creates (or
resumes) a W&B sweep, and spawns one ``wandb agent`` process per GPU.  Each
agent process has ``CUDA_VISIBLE_DEVICES`` pinned to a single GPU so that
multiple trials run in parallel across the available hardware.

Usage::

    # Launch a new sweep (auto-detect all GPUs)
    uv run python scripts/wandb_sweep.py configs/sweep/per_channel_cwt_behavior.yaml

    # Specify GPUs and limit trials per agent
    uv run python scripts/wandb_sweep.py configs/sweep/per_channel_cwt_behavior.yaml --gpus 0,1,2,3 --count 10

    # Resume an existing sweep (add more agents / GPUs)
    uv run python scripts/wandb_sweep.py --sweep-id <entity/project/sweep_id> --gpus 4,5,6,7

    # Run all four tokenizer sweeps sequentially
    for cfg in configs/sweep/*_behavior.yaml; do
      uv run python scripts/wandb_sweep.py "$cfg" --gpus 0,1,2,3,4,5,6,7 --count 20
    done
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wandb_sweep")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = "scripts/wandb_sweep_train.py"


def detect_gpus() -> list[int]:
    """Return indices of all GPUs visible to nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [int(line.strip()) for line in out.strip().splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return [0]


def load_sweep_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_wandb_config(sweep_meta: dict) -> dict:
    """Pull the ``wandb`` section and inject program/command for the agent."""
    wandb_config = dict(sweep_meta["wandb"])
    wandb_config["program"] = TRAIN_SCRIPT
    wandb_config["command"] = [
        "uv",
        "run",
        "python",
        TRAIN_SCRIPT,
    ]
    return wandb_config


def create_or_resume_sweep(
    sweep_meta: dict | None,
    sweep_id: str | None,
) -> str:
    """Create a new sweep or validate a resumed one.  Returns the full sweep path."""
    if sweep_id:
        log.info("Resuming existing sweep: %s", sweep_id)
        return sweep_id

    if sweep_meta is None:
        log.error("Either a config file or --sweep-id is required.")
        sys.exit(1)

    wandb_config = extract_wandb_config(sweep_meta)
    project = wandb_config.pop("project", "foundry")

    new_id = wandb.sweep(wandb_config, project=project)
    entity = wandb.Api().default_entity
    full_path = f"{entity}/{project}/{new_id}"

    log.info("Created sweep: %s", full_path)
    log.info("  Dashboard: https://wandb.ai/%s", full_path)
    return full_path


def launch_agents(
    sweep_path: str,
    gpus: list[int],
    config_path: str | None,
    count: int | None,
) -> list[subprocess.Popen]:
    """Spawn one ``wandb agent`` process per GPU."""
    processes: list[subprocess.Popen] = []

    for gpu_id in gpus:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        if config_path:
            env["SWEEP_CONFIG"] = str(Path(config_path).resolve())

        cmd = ["wandb", "agent", sweep_path]
        if count is not None:
            cmd.extend(["--count", str(count)])

        log.info("Starting agent on GPU %d: %s", gpu_id, " ".join(cmd))
        p = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        processes.append(p)

    return processes


def wait_for_agents(processes: list[subprocess.Popen]) -> None:
    """Wait for all agent processes, forwarding SIGINT/SIGTERM gracefully."""

    def _signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        log.warning("Received %s — terminating all agents...", sig_name)
        for p in processes:
            p.terminate()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for p in processes:
        p.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a sweep config YAML (e.g. configs/sweep/per_channel_cwt_behavior.yaml)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing sweep by its full path (entity/project/sweep_id)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices (default: auto-detect all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max number of runs each agent will execute (default: unlimited)",
    )
    args = parser.parse_args()

    if not args.config and not args.sweep_id:
        parser.error("Provide either a sweep config YAML or --sweep-id.")

    gpus = (
        [int(g) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    )
    log.info("Using GPUs: %s", gpus)

    sweep_meta = load_sweep_config(args.config) if args.config else None
    sweep_path = create_or_resume_sweep(sweep_meta, args.sweep_id)

    processes = launch_agents(sweep_path, gpus, args.config, args.count)
    log.info("Launched %d agent(s). Waiting for completion...", len(processes))

    wait_for_agents(processes)
    log.info("All agents finished.")


if __name__ == "__main__":
    main()
