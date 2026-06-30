"""CLI for launching Optuna hyperparameter sweeps with Hydra integration."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from foundry.optuna_integration import HydraOptunaOptimizer
from foundry.config_resolvers import register_resolvers

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging with Rich handler."""
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(name)s - %(message)s",
    )


def extract_search_space_from_config(
    cfg: DictConfig, search_space_key: str = "sweep.search_space"
) -> dict:
    """Extract search space definition from Hydra config.
    
    Expected format in config:
        sweep:
          search_space:
            hyperparameters.learning_rate:
              type: float
              low: 1e-5
              high: 1e-2
              log: true
            hyperparameters.batch_size:
              type: categorical
              choices: [32, 64, 128]
    """
    search_space_cfg = OmegaConf.select(cfg, search_space_key)
    if search_space_cfg is None:
        raise ValueError(
            f"No search space found at {search_space_key} in config"
        )
    
    return OmegaConf.to_container(search_space_cfg, resolve=True)


def create_train_fn(base_cfg: DictConfig):
    """Create a training function that reports metrics back to Optuna.
    
    The training function should:
    1. Build model/datamodule
    2. Run training
    3. Extract validation metric
    4. Log it back via wandb.log or direct return
    """
    
    def train_fn(cfg: DictConfig) -> float:
        """Train one trial and return the final metric value."""
        import torch
        from lightning.pytorch import Trainer
        from foundry.seed import set_seed
        from foundry.data.datamodules.base import normalize_data_config
        from hydra.utils import instantiate, get_class
        from foundry.training.module import FoundryModule
        
        # Setup
        set_seed(cfg.run.seed)
        torch.set_float32_matmul_precision("high")
        
        logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))
        
        # Build components
        from main import (
            _populate_data_driven_hyperparams,
            _load_task_configs,
            _apply_auto_class_weights,
            _configure_output_paths,
            _configure_wandb,
            _build_trainer,
        )
        
        _populate_data_driven_hyperparams(cfg)
        task_configs = _load_task_configs(cfg)
        normalize_data_config(cfg.data)
        datamodule = instantiate(cfg.data, tokenizer=None)
        datamodule._task_configs = task_configs
        task_configs = _apply_auto_class_weights(cfg, datamodule, task_configs)
        
        ModelClass = get_class(cfg.model._target_)
        model_kwargs = {
            k: instantiate(v) if OmegaConf.is_config(v) else v
            for k, v in cfg.model.items()
            if k != "_target_"
        }
        model = ModelClass(task_configs=task_configs, **model_kwargs)
        tokenizer = model.tokenize if hasattr(model, "tokenize") else None
        normalize_data_config(cfg.data)
        datamodule = instantiate(cfg.data, tokenizer=tokenizer)
        datamodule._task_configs = task_configs
        
        lightning_module = instantiate(cfg.module, model=model)
        trainer = _build_trainer(cfg)
        
        # Train
        trainer.fit(lightning_module, datamodule)
        
        # Extract metric from trainer state
        # Look for the best validation metric
        if trainer.callback_metrics:
            metric_key = OmegaConf.select(
                cfg, "sweep.metric", default=None
            )
            if metric_key and metric_key in trainer.callback_metrics:
                metric_value = trainer.callback_metrics[metric_key]
                logger.info("Trial metric %s = %f", metric_key, metric_value)
                return float(metric_value)
        
        logger.warning("Could not extract metric from trainer; using loss")
        return trainer.state.best_score or 0.0
    
    return train_fn


def launch_optuna_sweep(
    experiment: str,
    search_space: Optional[dict] = None,
    n_trials: int = 10,
    sampler: str = "tpe",
    pruner: str = "successive_halving",
    n_jobs: int = 1,
    storage_url: Optional[str] = None,
):
    """Launch an Optuna hyperparameter sweep.
    
    Args:
        experiment: Hydra experiment path (e.g., "auditory_decoding/poyo_neurosoft_8band_hp_sweep")
        search_space: Search space dict (overrides config if provided)
        n_trials: Number of trials
        sampler: "tpe", "random", or "grid"
        pruner: "successive_halving" or "none"
        n_jobs: Number of parallel jobs
        storage_url: SQLite storage URL for resumable studies
    """
    setup_logging()
    register_resolvers()
    
    # Initialize Hydra and load config
    @hydra.main(
        version_base=None,
        config_path="configs",
        config_name="config",
    )
    def load_config(cfg: DictConfig):
        return cfg
    
    cfg = load_config(["experiment=" + experiment])
    
    # Extract search space from config or use provided
    if search_space is None:
        search_space = extract_search_space_from_config(cfg)
    
    # Get metric from config
    metric_name = OmegaConf.select(
        cfg, "sweep.metric_name", default="val_loss"
    )
    metric_direction = OmegaConf.select(
        cfg, "sweep.metric_direction", default="minimize"
    )
    
    logger.info(
        "Launching Optuna sweep: experiment=%s metric=%s direction=%s",
        experiment,
        metric_name,
        metric_direction,
    )
    
    # Create optimizer
    optimizer = HydraOptunaOptimizer(
        experiment_name=experiment.replace("/", "_"),
        search_space=search_space,
        metric_name=metric_name,
        metric_direction=metric_direction,
        sampler_type=sampler,
        pruner_type=pruner,
        storage_url=storage_url,
        n_trials=n_trials,
    )
    
    # Create training function
    train_fn = create_train_fn(cfg)
    
    # Run optimization
    best_params = optimizer.optimize(
        cfg,
        train_fn,
        n_jobs=n_jobs,
        show_progress=True,
    )
    
    logger.info(
        "Sweep complete. Best parameters:\n%s",
        json.dumps(best_params, indent=2),
    )
    
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch an Optuna hyperparameter sweep"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Hydra experiment path",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials (default: 10)",
    )
    parser.add_argument(
        "--sampler",
        choices=["tpe", "random", "grid"],
        default="tpe",
        help="Optuna sampler type (default: tpe)",
    )
    parser.add_argument(
        "--pruner",
        choices=["successive_halving", "none"],
        default="successive_halving",
        help="Optuna pruner type (default: successive_halving)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    parser.add_argument(
        "--storage-url",
        help="SQLite storage URL for resumable studies",
    )
    
    args = parser.parse_args()
    
    launch_optuna_sweep(
        experiment=args.experiment,
        n_trials=args.n_trials,
        sampler=args.sampler,
        pruner=args.pruner,
        n_jobs=args.n_jobs,
        storage_url=args.storage_url,
    )
