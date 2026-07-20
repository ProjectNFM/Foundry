"""Lightning callbacks for Foundry model training.

Re-exports all public callback classes so that existing Hydra ``_target_``
strings (e.g. ``foundry.training.callbacks.VocabInitializerCallback``)
continue to resolve after the single-file module was split into a package.
"""

from __future__ import annotations

from lightning import Trainer

from foundry.training.callbacks.diagnostics import ParameterWatcherCallback
from foundry.training.callbacks.lifecycle import (
    DeterministicSamplerCallback,
    VocabInitializerCallback,
)
from foundry.training.callbacks.metrics import (
    ConfusionMatrixCallback,
    SessionMetricsCallback,
)
from foundry.training.callbacks.tuning import EffectiveBatchSizeCallback
from foundry.training.callbacks.visualization import (
    ReconstructionVisualizationCallback,
)


def get_wandb_experiment(trainer: Trainer):
    """Return the W&B experiment object if a WandbLogger is active, else None."""
    from lightning.pytorch.loggers import WandbLogger

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger.experiment
    return None


__all__ = [
    "ConfusionMatrixCallback",
    "DeterministicSamplerCallback",
    "EffectiveBatchSizeCallback",
    "ParameterWatcherCallback",
    "ReconstructionVisualizationCallback",
    "SessionMetricsCallback",
    "VocabInitializerCallback",
    "get_wandb_experiment",
]
