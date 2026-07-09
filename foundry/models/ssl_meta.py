"""Typed contracts for self-supervised learning output metadata.

These dataclasses replace the untyped dicts previously used to pass
reconstruction metadata between :class:`MaskedPOYOEEGModel` and
:class:`~foundry.training.module.FoundryModule`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SSLTaskMeta:
    """Per-task reconstruction targets and validity weights."""

    targets: torch.Tensor
    weights: torch.Tensor


@dataclass
class ReconstructionVizMeta:
    """Metadata for visualizing masked reconstruction during validation."""

    mask_indices: torch.Tensor
    validity_mask: torch.Tensor
    num_channels: int
    num_time_tokens: int


__all__ = ["SSLTaskMeta", "ReconstructionVizMeta"]
