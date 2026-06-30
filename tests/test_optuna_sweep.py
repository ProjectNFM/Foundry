"""Integration tests for Optuna hyperparameter sweep functionality."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from foundry.optuna_integration import HydraOptunaOptimizer


class TestHydraOptunaOptimizer:
    """Test suite for HydraOptunaOptimizer."""
    
    @pytest.fixture
    def search_space(self):
        """Fixture providing a simple search space."""
        return {
            "hyperparameters.learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-3,
                "log": True,
            },
            "hyperparameters.batch_size": {
                "type": "categorical",
                "choices": [32, 64],
            },
        }
    
    @pytest.fixture
    def base_config(self):
        """Fixture providing a basic Hydra config."""
        return OmegaConf.create({
            "hyperparameters": {
                "learning_rate": 1e-3,
                "batch_size": 64,
            },
            "sweep": {
                "metric_name": "val_loss",
                "metric_direction": "minimize",
            },
        })
    
    def test_optimizer_initialization(self, search_space, tmp_path):
        """Test that optimizer initializes correctly."""
        storage_url = f"sqlite:///{tmp_path}/test_study.db"
        optimizer = HydraOptunaOptimizer(
            experiment_name="test_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            sampler_type="random",
            pruner_type="none",
            storage_url=storage_url,
            n_trials=3,
        )
        
        assert optimizer.experiment_name == "test_exp"
        assert optimizer.metric_name == "val_loss"
        assert optimizer.n_trials == 3
        assert optimizer.study is not None
    
    def test_suggest_trial(self, search_space, tmp_path):
        """Test trial suggestion generation."""
        storage_url = f"sqlite:///{tmp_path}/test_study.db"
        optimizer = HydraOptunaOptimizer(
            experiment_name="test_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            sampler_type="random",
            storage_url=storage_url,
            n_trials=1,
        )
        
        # Create a trial and get suggestions
        trial = optimizer.study.ask()
        suggestions = optimizer._suggest_trial(trial)
        
        # Verify structure
        assert "hyperparameters.learning_rate" in suggestions
        assert "hyperparameters.batch_size" in suggestions
        
        # Verify ranges
        lr = suggestions["hyperparameters.learning_rate"]
        assert 1e-5 <= lr <= 1e-3
        
        bs = suggestions["hyperparameters.batch_size"]
        assert bs in [32, 64]
    
    def test_objective_function(self, search_space, base_config, tmp_path):
        """Test the objective function wrapper."""
        storage_url = f"sqlite:///{tmp_path}/test_study.db"
        optimizer = HydraOptunaOptimizer(
            experiment_name="test_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            sampler_type="random",
            storage_url=storage_url,
            n_trials=1,
        )
        
        # Mock training function that returns a metric
        def mock_train_fn(cfg: DictConfig) -> float:
            return 0.5
        
        trial = optimizer.study.ask()
        result = optimizer.objective(trial, base_config, mock_train_fn)
        
        assert isinstance(result, float)
        assert result == 0.5
    
    def test_get_best_params_empty(self, search_space, tmp_path):
        """Test getting best params when no trials completed."""
        storage_url = f"sqlite:///{tmp_path}/test_study.db"
        optimizer = HydraOptunaOptimizer(
            experiment_name="test_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            storage_url=storage_url,
            n_trials=1,
        )
        
        # No trials yet
        assert optimizer.get_best_params() == {}
        assert optimizer.get_best_value() is None
    
    def test_resumable_storage(self, search_space, tmp_path):
        """Test that studies can be resumed from SQLite storage."""
        storage_url = f"sqlite:///{tmp_path}/resumable_study.db"
        
        # Create first optimizer and run one trial
        optimizer1 = HydraOptunaOptimizer(
            experiment_name="resumable_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            sampler_type="random",
            storage_url=storage_url,
            n_trials=1,
        )
        
        def mock_train_fn(cfg):
            return 0.5
        
        trial = optimizer1.study.ask()
        optimizer1.objective(trial, OmegaConf.create({}), mock_train_fn)
        optimizer1.study.tell(trial, 0.5)
        
        n_trials_before = len(optimizer1.study.trials)
        
        # Create second optimizer with same storage
        optimizer2 = HydraOptunaOptimizer(
            experiment_name="resumable_exp",
            search_space=search_space,
            metric_name="val_loss",
            metric_direction="minimize",
            sampler_type="random",
            storage_url=storage_url,
            n_trials=2,
        )
        
        # Should load existing study
        assert len(optimizer2.study.trials) == n_trials_before


class TestSweepMode:
    """Test WandB sweep mode detection in main.py."""
    
    def test_is_sweep_mode_false(self):
        """Test that sweep mode is False by default."""
        # Clean environment
        os.environ.pop("WANDB_SWEEP_ID", None)
        
        from main import _is_sweep_mode
        assert not _is_sweep_mode()
    
    def test_is_sweep_mode_true(self):
        """Test that sweep mode is True when WANDB_SWEEP_ID is set."""
        os.environ["WANDB_SWEEP_ID"] = "test_sweep_123"
        
        from main import _is_sweep_mode
        assert _is_sweep_mode()
        
        # Cleanup
        os.environ.pop("WANDB_SWEEP_ID", None)
    
    @patch("wandb.run")
    def test_inject_sweep_hyperparams_no_sweep(self, mock_run):
        """Test that injection does nothing when not in sweep mode."""
        mock_run.return_value = None
        os.environ.pop("WANDB_SWEEP_ID", None)
        
        from main import _inject_sweep_hyperparams
        cfg = OmegaConf.create({"hyperparameters": {"lr": 0.001}})
        
        # Should not raise, config unchanged
        _inject_sweep_hyperparams(cfg)
        assert cfg.hyperparameters.lr == 0.001
        
        # Cleanup
        os.environ.pop("WANDB_SWEEP_ID", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
