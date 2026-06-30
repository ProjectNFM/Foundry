"""Optuna integration for intelligent hyperparameter optimization with Hydra."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class HydraOptunaOptimizer:
    """Manages Optuna trials and Hydra config injection for hyperparameter optimization."""
    
    def __init__(
        self,
        experiment_name: str,
        search_space: dict[str, dict[str, Any]],
        metric_name: str,
        metric_direction: str = "minimize",
        sampler_type: str = "tpe",
        pruner_type: str = "successive_halving",
        storage_url: Optional[str] = None,
        n_trials: int = 10,
    ):
        """Initialize the Optuna optimizer.
        
        Args:
            experiment_name: Experiment identifier (used for study name)
            search_space: Dict mapping param names to bounds/choices:
                         {"param_name": {"type": "float", "low": 0.1, "high": 1.0}}
                         {"param_name": {"type": "int", "low": 1, "high": 100}}
                         {"param_name": {"type": "categorical", "choices": [a, b, c]}}
            metric_name: Name of the metric to optimize (e.g., "val_loss")
            metric_direction: "minimize" or "maximize"
            sampler_type: "tpe", "random", or "grid"
            pruner_type: "successive_halving" or "none"
            storage_url: SQLite URL for resumable studies (None = in-memory)
            n_trials: Total number of trials to run
        """
        self.experiment_name = experiment_name
        self.search_space = search_space
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.n_trials = n_trials
        
        # Create storage for resumable studies
        if storage_url is None:
            storage_url = f"sqlite:///./{experiment_name}_optuna_study.db"
        self.storage_url = storage_url
        
        # Select sampler
        if sampler_type == "tpe":
            sampler = TPESampler(seed=42)
        elif sampler_type == "random":
            sampler = RandomSampler(seed=42)
        elif sampler_type == "grid":
            # Grid sampler requires search space in specific format
            sampler = GridSampler()
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
        
        # Select pruner
        if pruner_type == "successive_halving":
            pruner = SuccessiveHalvingPruner()
        elif pruner_type == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner_type: {pruner_type}")
        
        # Create or load study
        storage = optuna.storages.RDBStorage(storage_url)
        self.study = optuna.create_study(
            study_name=experiment_name,
            direction=metric_direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        
        logger.info(
            "Initialized Optuna study: name=%s sampler=%s n_trials=%d",
            experiment_name,
            sampler_type,
            n_trials,
        )
    
    def _suggest_trial(self, trial: optuna.Trial) -> dict[str, Any]:
        """Generate suggestions for one trial based on search_space."""
        suggestions = {}
        
        for param_name, bounds in self.search_space.items():
            param_type = bounds.get("type", "float")
            
            if param_type == "float":
                suggestions[param_name] = trial.suggest_float(
                    param_name,
                    bounds["low"],
                    bounds["high"],
                    log=bounds.get("log", False),
                )
            elif param_type == "int":
                suggestions[param_name] = trial.suggest_int(
                    param_name,
                    bounds["low"],
                    bounds["high"],
                    log=bounds.get("log", False),
                )
            elif param_type == "categorical":
                suggestions[param_name] = trial.suggest_categorical(
                    param_name,
                    bounds["choices"],
                )
            else:
                raise ValueError(f"Unknown param_type: {param_type}")
        
        return suggestions
    
    def objective(
        self,
        trial: optuna.Trial,
        base_cfg: DictConfig,
        train_fn: Callable[[DictConfig], float],
    ) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            base_cfg: Base Hydra config to modify
            train_fn: Training function that takes config and returns metric value
                     
        Returns:
            Metric value to optimize
        """
        # Generate hyperparameter suggestions
        suggestions = self._suggest_trial(trial)
        cfg = OmegaConf.create(base_cfg)
        
        # Inject suggestions into config
        for param_name, value in suggestions.items():
            try:
                OmegaConf.update(cfg, param_name, value)
            except Exception as e:
                logger.warning(
                    "Failed to set %s = %s: %s",
                    param_name,
                    value,
                    e,
                )
        
        logger.info("Trial %d: %s", trial.number, suggestions)
        
        try:
            metric_value = train_fn(cfg)
            return metric_value
        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned()
    
    def optimize(
        self,
        base_cfg: DictConfig,
        train_fn: Callable[[DictConfig], float],
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run the optimization loop.
        
        Args:
            base_cfg: Base Hydra config
            train_fn: Training function that returns metric value
            n_jobs: Number of parallel jobs (requires distributed setup)
            show_progress: Whether to show progress bar
            
        Returns:
            Best trial's params dict
        """
        logger.info(
            "Starting optimization: n_trials=%d n_jobs=%d",
            self.n_trials,
            n_jobs,
        )
        
        self.study.optimize(
            lambda trial: self.objective(trial, base_cfg, train_fn),
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )
        
        best_trial = self.study.best_trial
        logger.info(
            "Optimization complete: best_value=%f best_trial=%d",
            best_trial.value,
            best_trial.number,
        )
        logger.info("Best hyperparameters: %s", best_trial.params)
        
        return best_trial.params
    
    def get_best_params(self) -> dict[str, Any]:
        """Get the best hyperparameters found so far."""
        if self.study.best_trial is None:
            return {}
        return self.study.best_trial.params
    
    def get_best_value(self) -> Optional[float]:
        """Get the best metric value found so far."""
        if self.study.best_trial is None:
            return None
        return self.study.best_trial.value


def run_optuna_trial_subprocess(
    base_cfg_path: str,
    trial_params: dict[str, Any],
    train_script: str = "main.py",
) -> float:
    """Run a single Optuna trial in a subprocess (for cluster integration).
    
    Args:
        base_cfg_path: Path to base config file
        trial_params: Hyperparameters for this trial
        train_script: Python script to run (default: main.py)
        
    Returns:
        Final metric value
    """
    # Build Hydra overrides from trial params
    overrides = []
    for key, value in trial_params.items():
        if isinstance(value, str):
            overrides.append(f"{key}='{value}'")
        else:
            overrides.append(f"{key}={value}")
    
    cmd = ["python", train_script] + overrides
    logger.info("Running trial: %s", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error("Trial failed:\n%s", result.stderr)
            raise RuntimeError(f"Trial process failed with code {result.returncode}")
        
        # Parse metric from output (assumes it's printed as "METRIC_VALUE: <float>")
        for line in result.stdout.split("\n"):
            if line.startswith("FINAL_METRIC:"):
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        
        logger.warning("Could not parse metric from trial output")
        return float("inf")
    
    except subprocess.TimeoutExpired:
        logger.error("Trial timeout")
        raise
