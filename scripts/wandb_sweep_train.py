"""W&B sweep training wrapper.

Called by ``wandb agent`` for each trial.  Receives sweep parameters as
``key=value`` CLI arguments (via ``${args_no_hyphens}`` in the sweep command),
maps them to Hydra CLI overrides using the ``parameter_map`` from the sweep
config, and launches ``main.py`` as a subprocess.

The wrapper passes ``logger.id=<WANDB_RUN_ID>`` so that ``main.py``'s
WandbLogger creates the W&B run with the ID that the sweep controller
allocated.  ``WANDB_SWEEP_ID`` stays in the environment so the run is
automatically associated with the sweep — no ``wandb.init()`` is called here,
which avoids creating extraneous short-lived runs.

This script is not meant to be invoked manually; it is set as the sweep
``program`` by :mod:`scripts.wandb_sweep`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wandb_sweep_train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_sweep_meta() -> dict:
    config_path = os.environ.get("SWEEP_CONFIG")
    if not config_path:
        log.error("SWEEP_CONFIG environment variable is not set.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_sweep_args(args: list[str]) -> dict[str, str]:
    """Parse ``key=value`` positional args passed by ``wandb agent``."""
    params: dict[str, str] = {}
    for arg in args:
        if "=" in arg:
            key, _, value = arg.partition("=")
            params[key] = value
    return params


def _build_hydra_overrides(
    sweep_meta: dict,
    wandb_params: dict[str, str],
    run_id: str,
    sweep_id: str,
) -> list[str]:
    experiment = sweep_meta["experiment"]
    tokenizer = sweep_meta.get("tokenizer")
    extra_overrides = sweep_meta.get("extra_overrides", [])
    parameter_map = sweep_meta.get("parameter_map", {})

    overrides = [
        f"experiment={experiment}",
        f"logger.id={run_id}",
        "logger.resume=allow",
        f"run.name=sweep_{run_id}",
        f"run.group=sweep_{sweep_id}",
    ]

    if tokenizer:
        overrides.append(f"model/tokenizer={tokenizer}")

    overrides.extend(extra_overrides)

    for param_name, value in wandb_params.items():
        hydra_path = parameter_map.get(param_name, param_name)
        overrides.append(f"{hydra_path}={value}")

    return overrides


def main() -> None:
    sweep_meta = _load_sweep_meta()

    run_id = os.environ.get("WANDB_RUN_ID", "")
    sweep_id = os.environ.get("WANDB_SWEEP_ID", "wandb_sweep")

    if not run_id:
        log.error("WANDB_RUN_ID not set — is this being called by wandb agent?")
        sys.exit(1)

    wandb_params = _parse_sweep_args(sys.argv[1:])
    overrides = _build_hydra_overrides(
        sweep_meta, wandb_params, run_id, sweep_id
    )

    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")] + overrides

    log.info("Trial %s — launching main.py", run_id)
    log.info("  overrides: %s", " ".join(overrides))

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        log.error("Trial %s FAILED (exit %d)", run_id, result.returncode)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
