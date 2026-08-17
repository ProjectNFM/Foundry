"""Typed step-output contract for callback consumption.

Carries the data produced by each training/validation step so that
optional callbacks can observe training without attaching private
buffers to the Lightning module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from foundry.models.ssl_meta import ReconstructionVizMeta, RepresentationPayload


@dataclass
class SampleMetadata:
    """Explicit identity and validity metadata for a collated sample batch."""

    dataset_id: list[str] | None = None
    subject_id: list[str] | None = None
    session_id: list[str] | None = None
    absolute_start: torch.Tensor | None = None
    window_duration: torch.Tensor | None = None
    channel_index: torch.Tensor | None = None
    channel_mask: torch.Tensor | None = None

    def detached(self) -> "SampleMetadata":
        """Return metadata tensors detached from any computation graph."""

        def detach(value):
            return value.detach() if torch.is_tensor(value) else value

        return SampleMetadata(
            dataset_id=self.dataset_id,
            subject_id=self.subject_id,
            session_id=self.session_id,
            absolute_start=detach(self.absolute_start),
            window_duration=detach(self.window_duration),
            channel_index=detach(self.channel_index),
            channel_mask=detach(self.channel_mask),
        )


@dataclass
class StepOutput:
    """Data produced by a single training/validation step.

    Tensors remain on the original device; callbacks are responsible
    for detaching and moving to CPU when buffering across steps.
    """

    loss: torch.Tensor
    task_outputs: dict[str, torch.Tensor]
    target_values: dict[str, torch.Tensor]
    target_weights: dict[str, torch.Tensor | float]
    task_index: torch.Tensor
    session_id: list[str] | None = None
    sample_metadata: SampleMetadata | None = None
    representations: RepresentationPayload | None = None
    ssl_task_names: set[str] = field(default_factory=set)
    reconstruction_viz: ReconstructionVizMeta | None = None
    reconstruction_targets: torch.Tensor | None = None
    input_mask: torch.Tensor | None = None


def extract_step_output(outputs) -> StepOutput | None:
    """Extract a StepOutput from Lightning callback ``outputs`` parameter."""
    if isinstance(outputs, dict):
        return outputs.get("step_output")
    return None
