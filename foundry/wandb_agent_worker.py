"""Run a WandB sweep agent that launches Foundry training via Hydra.

Modern wandb (>=0.24) no longer supports ``wandb agent <id> -- python train.py``.
Use this module instead::

    WANDB_SWEEP_EXPERIMENT=auditory_decoding/foo \\
        uv run python -m foundry.wandb_agent_worker suarezul/Foundry/abc123
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

logger = logging.getLogger(__name__)


def _format_override(key: str, value) -> str:
    if isinstance(value, str):
        return f"{key}='{value}'"
    return f"{key}={value}"


def _load_sweep_params() -> dict[str, object]:
    """Load sweep hyperparameters written by the wandb agent.

    The agent callback runs before ``wandb.init()``, so ``wandb.config`` is
    unavailable. Params are read from the YAML file at WANDB_SWEEP_PARAM_PATH.
    """
    param_path = os.environ.get(wandb.env.SWEEP_PARAM_PATH)
    if not param_path or not os.path.isfile(param_path):
        raise RuntimeError(
            f"Missing sweep param file (WANDB_SWEEP_PARAM_PATH={param_path!r})"
        )

    with open(param_path) as f:
        raw = yaml.safe_load(f) or {}

    params: dict[str, object] = {}
    for key, val in raw.items():
        if key == "wandb_version":
            continue
        if isinstance(val, dict) and "value" in val:
            params[key] = val["value"]
        else:
            params[key] = val
    return params


def run_trial() -> None:
    """Execute one sweep trial by calling main.py with Hydra overrides."""
    project_dir = Path(os.environ.get("FOUNDRY_ROOT", os.getcwd()))
    experiment = os.environ["WANDB_SWEEP_EXPERIMENT"]

    overrides = [
        f"experiment={experiment}",
        "cluster=cscs",
    ]
    for key, value in _load_sweep_params().items():
        overrides.append(_format_override(key, value))

    cmd = ["uv", "run", "python", "main.py", *overrides]
    logger.info("Running trial: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=project_dir, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m foundry.wandb_agent_worker <sweep_id>")
    if "WANDB_SWEEP_EXPERIMENT" not in os.environ:
        sys.exit("WANDB_SWEEP_EXPERIMENT must be set")

    sweep_id = sys.argv[1]
    count_raw = os.environ.get("WANDB_SWEEP_COUNT", "")
    count = int(count_raw) if count_raw else None

    wandb.agent(sweep_id, function=run_trial, count=count)


if __name__ == "__main__":
    main()
