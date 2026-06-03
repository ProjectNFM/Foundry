"""Masking utilities for self-supervised pretraining."""

from __future__ import annotations

import torch


def generate_temporal_mask(
    num_time_tokens: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample a contiguous time-span mask.

    Returns the half-open interval ``[start, end)`` of time-token indices
    to mask.  The span length is ``round(mask_ratio * num_time_tokens)``,
    clamped to ``[1, num_time_tokens]``.

    Args:
        num_time_tokens: Number of time tokens per channel.
        mask_ratio: Fraction of time tokens to mask.
        generator: Optional RNG for reproducibility.

    Returns:
        ``(start, end)`` index pair.
    """
    span_length = max(
        1, min(num_time_tokens, round(mask_ratio * num_time_tokens))
    )
    max_start = num_time_tokens - span_length
    start = int(
        torch.randint(0, max_start + 1, (1,), generator=generator).item()
    )
    return start, start + span_length


def build_token_mask(
    num_channels: int,
    num_time_tokens: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Expand a time-span mask to all channels on a ``C × T`` token grid.

    Args:
        num_channels: Number of channels.
        num_time_tokens: Number of time tokens per channel.
        start: Inclusive start index of the masked time span.
        end: Exclusive end index of the masked time span.

    Returns:
        Bool tensor of shape ``(C * T,)`` where ``True`` = masked.
    """
    time_mask = torch.zeros(num_time_tokens, dtype=torch.bool)
    time_mask[start:end] = True
    return time_mask.unsqueeze(0).expand(num_channels, -1).reshape(-1)
