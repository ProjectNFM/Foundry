"""W&B sweep training wrapper.

Called by ``wandb agent`` for each trial.  Reads the sweep configuration from
the ``SWEEP_CONFIG`` environment variable, initialises a W&B run to receive
the trial's hyperparameters, then launches ``main.py`` as a subprocess with
the corresponding Hydra CLI overrides.

The wrapper passes ``logger.id=<run_id>`` so that ``main.py``'s WandbLogger
resumes the *same* W&B run that the sweep controller allocated — all training
metrics flow directly into the sweep dashboard with no duplicate runs.

This script is not meant to be invoked manually; it is set as the sweep
``program`` by :mod:`scripts.wandb_sweep`.
"""

from __future__ import annotations

import logging
import os
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
log = logging.getLogger("wandb_sweep_train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_sweep_meta() -> dict:
    config_path = os.environ.get("SWEEP_CONFIG")
    if not config_path:
        log.error("SWEEP_CONFIG environment variable is not set.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_hydra_overrides(
    sweep_meta: dict,
    wandb_config: dict,
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

    for param_name, value in wandb_config.items():
        hydra_path = parameter_map.get(param_name, param_name)
        overrides.append(f"{hydra_path}={value}")

    return overrides


def main() -> None:
    sweep_meta = _load_sweep_meta()

    # wandb.init() picks up WANDB_SWEEP_ID / WANDB_RUN_ID set by the agent
    run = wandb.init()
    config = dict(run.config)
    run_id = run.id
    sweep_id = run.sweep_id or "wandb_sweep"
    wandb.finish(quiet=True)

    overrides = _build_hydra_overrides(sweep_meta, config, run_id, sweep_id)

    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")] + overrides

    # Remove WANDB_RUN_ID so main.py's WandbLogger doesn't conflict with the
    # explicit logger.id we pass.  Keep WANDB_SWEEP_ID so the resumed run
    # stays associated with the sweep.
    env = {**os.environ}
    env.pop("WANDB_RUN_ID", None)

    log.info("Trial %s — launching main.py", run_id)
    log.info("  overrides: %s", " ".join(overrides))

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        log.error("Trial %s FAILED (exit %d)", run_id, result.returncode)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
