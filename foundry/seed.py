"""Centralized RNG seeding for reproducible experiments.

Wraps ``lightning.seed_everything`` and optionally enables PyTorch's
deterministic mode for bitwise-reproducible runs (at a performance cost).
"""

import logging
import os

import torch
from lightning import seed_everything

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNG sources used during training.

    This seeds ``random``, ``numpy``, ``torch`` (CPU + all CUDA devices),
    and configures Lightning to seed DataLoader workers.

    Args:
        seed: Integer seed shared across all libraries.
        deterministic: If *True*, also set ``PYTHONHASHSEED``,
            ``CUBLAS_WORKSPACE_CONFIG``, ``cudnn.deterministic``, and
            ``torch.use_deterministic_algorithms`` so that results are
            bitwise reproducible (at a performance cost).
    """
    seed_everything(seed, workers=True)

    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic mode enabled (may reduce performance).")
