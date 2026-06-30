"""WandB sweep utilities for managing and launching hyperparameter sweeps."""

import logging
import os
from typing import Any, Optional

import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_sweep_config(
    experiment_name: str,
    search_space: dict[str, Any],
    metric_name: str,
    metric_goal: str = "minimize",
    method: str = "bayes",
    early_stopping_patience: int = 10,
) -> dict[str, Any]:
    """Programmatically build a WandB sweep config.
    
    Args:
        experiment_name: Experiment identifier for the sweep
        search_space: Dict mapping hyperparameter names to their search ranges.
                     Each value should be either:
                     - {"values": [v1, v2, ...]} for discrete choices
                     - {"distribution": "uniform"/"log_uniform", "min": a, "max": b}
        metric_name: Name of the metric to optimize (e.g., "val/accuracy")
        metric_goal: "minimize" or "maximize"
        method: Search method: "bayes", "random", or "grid"
        early_stopping_patience: Stop if no improvement after this many trials
        
    Returns:
        A WandB sweep config dict ready to pass to wandb.sweep()
    """
    sweep_config = {
        "method": method,
        "name": f"{experiment_name}_sweep",
        "metric": {
            "name": metric_name,
            "goal": metric_goal,
        },
        "parameters": search_space,
    }
    
    if method == "bayes":
        sweep_config["early_stopping"] = {
            "type": "hyperband",
            "min_iter": 3,
            "s": 2,
            "eta": 3,
            "max_iter": 100,
        }
    
    return sweep_config


def launch_wandb_sweep(
    sweep_config: dict[str, Any],
    project: str,
    entity: Optional[str] = None,
) -> str:
    """Initialize a WandB sweep and return the sweep ID.
    
    Args:
        sweep_config: Sweep configuration dict from create_sweep_config()
        project: WandB project name
        entity: WandB entity (team/username), defaults to WANDB_ENTITY env var
        
    Returns:
        Sweep ID to use with wandb agent
    """
    if entity is None:
        entity = os.environ.get("WANDB_ENTITY")
    
    logger.info(
        "Initializing WandB sweep: project=%s entity=%s method=%s",
        project,
        entity,
        sweep_config.get("method"),
    )
    
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    logger.info("Created sweep: %s", sweep_id)
    return sweep_id


def run_sweep_worker(
    sweep_id: str,
    main_fn,
    experiment_config: DictConfig,
    count: int = 10,
) -> None:
    """Worker loop for a WandB sweep agent.
    
    Pulls trials from the sweep, runs training with swept hyperparameters,
    and logs results back to WandB.
    
    Args:
        sweep_id: Sweep ID from launch_wandb_sweep()
        main_fn: Training function to call (e.g., main from main.py)
                Should accept a DictConfig and run one trial
        experiment_config: Base Hydra config for the experiment
        count: Number of trials for this worker to complete (None = infinite)
    """
    
    def trial_wrapper():
        """Wrapper to inject swept hyperparams into config."""
        cfg = OmegaConf.create(experiment_config)
        
        # Pull hyperparameters from wandb.config (set by sweep)
        for key, value in dict(wandb.config).items():
            try:
                OmegaConf.update(cfg, key, value)
            except Exception as e:
                logger.warning(
                    "Failed to set config.%s = %s: %s",
                    key,
                    value,
                    e,
                )
        
        # Run training
        try:
            main_fn(cfg)
        except Exception as e:
            logger.error("Trial failed with error: %s", e)
            raise
    
    logger.info("Starting sweep worker: sweep_id=%s count=%s", sweep_id, count)
    wandb.agent(sweep_id, function=trial_wrapper, count=count)


def is_sweep_mode() -> bool:
    """Check if running under WandB sweep."""
    return "WANDB_SWEEP_ID" in os.environ


def get_sweep_id() -> Optional[str]:
    """Get the current sweep ID if running under sweep."""
    return os.environ.get("WANDB_SWEEP_ID")
