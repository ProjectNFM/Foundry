"""Utilities for deriving and cloning ModalitySpec objects.

This module provides helpers to create effective readout specs with modified
dimensions and loss functions while preserving routing identifiers.
"""

from dataclasses import replace
from torch_brain.registry import ModalitySpec
from torch_brain.nn.loss import Loss


def clone_readout_spec(
    base: ModalitySpec,
    *,
    dim: int | None = None,
    loss_fn: Loss | None = None,
) -> ModalitySpec:
    """Clone a ModalitySpec with optional overrides to dim and loss_fn.

    Preserves routing-critical fields (id, type, timestamp_key, value_key)
    so multitask decoding remains aligned.

    Args:
        base: Base ModalitySpec to clone from
        dim: Optional new dimension override
        loss_fn: Optional new loss function override

    Returns:
        New ModalitySpec with specified overrides applied
    """
    kwargs = {}
    if dim is not None:
        kwargs["dim"] = dim
    if loss_fn is not None:
        kwargs["loss_fn"] = loss_fn

    return replace(base, **kwargs)
