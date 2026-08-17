"""Typed contracts for self-supervised learning output metadata.

These dataclasses replace the untyped dicts previously used to pass
reconstruction metadata between :class:`MaskedPOYOEEGModel` and
:class:`~foundry.training.module.FoundryModule`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class RepresentationPayload:
    """Optional representations captured for a scheduled validation event.

    Channel representations are the vectors fused into temporal tokens before
    the Perceiver. Backbone representations are mean-pooled final processed
    latents. A missing tensor means that representation family is unavailable.
    """

    channel_representations: torch.Tensor | None = None
    backbone_representations: torch.Tensor | None = None
    channel_mode: str | None = None
    channel_mask: torch.Tensor | None = None

    def detached(self) -> "RepresentationPayload":
        """Return a graph-free payload suitable for callback consumption."""

        return RepresentationPayload(
            channel_representations=(
                self.channel_representations.detach()
                if self.channel_representations is not None
                else None
            ),
            backbone_representations=(
                self.backbone_representations.detach()
                if self.backbone_representations is not None
                else None
            ),
            channel_mode=self.channel_mode,
            channel_mask=(
                self.channel_mask.detach()
                if self.channel_mask is not None
                else None
            ),
        )


@dataclass
class ModelOutput:
    """Structured output from model forward passes.

    Replaces the magic ``_ssl_meta`` / ``_reconstruction_viz`` dict keys
    that were previously stuffed into the task-output dict.
    """

    task_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    ssl_meta: dict[str, SSLTaskMeta] | None = None
    viz: ReconstructionVizMeta | None = None
    representations: RepresentationPayload | None = None


__all__ = [
    "SSLTaskMeta",
    "ReconstructionVizMeta",
    "RepresentationPayload",
    "ModelOutput",
]
