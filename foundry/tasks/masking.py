"""Token-level masking strategies for MAE-style self-supervised pretraining.

All strategies operate on the full ``(C_pad, N)`` token grid from
:class:`~foundry.models.embeddings.channel.PerChannelStrategy` and return a
fixed number of masked token indices per sample. The count is deterministic
from ``(C_pad, N, mask_ratio)`` — independent of per-sample ``C_real``.

A separate ``validity_mask`` ``(B, num_masked)`` indicates which masked
positions correspond to real channels (True) vs padded channels (False).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskingStrategy:
    """Base for token-level masking strategies."""

    mask_ratio: float

    def __call__(
        self,
        num_channels: int,
        num_time_tokens: int,
        channel_mask: torch.BoolTensor,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Args:
            num_channels: C_pad (total padded channels).
            num_time_tokens: N (temporal tokens per channel).
            channel_mask: (B, C_pad) which channels are real.

        Returns:
            mask_indices: (B, num_masked) indices into flattened (C_pad * N).
            validity_mask: (B, num_masked) True for real-channel positions.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RandomTokenMasking(MaskingStrategy):
    """I.i.d. random masking of individual tokens (MAE default).

    A fixed count of ``floor(mask_ratio * C_pad * N)`` tokens is selected per
    sample via random permutation.
    """

    def __call__(self, num_channels, num_time_tokens, channel_mask):
        B = channel_mask.shape[0]
        C, N = num_channels, num_time_tokens
        total_tokens = C * N
        num_masked = max(1, int(self.mask_ratio * total_tokens))

        noise = torch.rand(B, total_tokens, device=channel_mask.device)
        indices = noise.argsort(dim=1)
        mask_indices = indices[:, :num_masked]

        channel_of_token = mask_indices // N
        validity_mask = torch.gather(channel_mask, 1, channel_of_token)

        return mask_indices, validity_mask


@dataclass(frozen=True)
class TemporalBlockMasking(MaskingStrategy):
    """Masks multiple non-overlapping time blocks across ALL channels.

    The time axis is divided into slots of ``block_size`` tokens. A random
    subset of slots is selected until ``floor(mask_ratio * N)`` time positions
    are masked. Each selected slot spans ALL ``C_pad`` channels.

    Total ``num_masked = num_time_masked * C_pad`` (fixed).
    """

    block_size: int = 5

    def __call__(self, num_channels, num_time_tokens, channel_mask):
        B = channel_mask.shape[0]
        device = channel_mask.device
        C, N = num_channels, num_time_tokens

        num_slots = N // self.block_size

        if num_slots == 0:
            # Not enough time tokens for a full block; mask individual
            # time positions across all channels instead.
            num_time_masked = max(1, int(self.mask_ratio * N))
            num_masked = num_time_masked * C

            noise = torch.rand(B, N, device=device)
            time_indices = noise.argsort(dim=1)[:, :num_time_masked]
        else:
            num_time_masked_raw = max(1, int(self.mask_ratio * N))
            num_blocks = max(
                1, min(num_time_masked_raw // self.block_size, num_slots)
            )
            num_time_masked = num_blocks * self.block_size
            num_masked = num_time_masked * C

            noise = torch.rand(B, num_slots, device=device)
            selected_slots = noise.argsort(dim=1)[:, :num_blocks]
            slot_starts = selected_slots * self.block_size

            offsets = torch.arange(self.block_size, device=device)
            time_indices = (
                slot_starts.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)
            ).reshape(B, num_time_masked)

        channel_offsets = torch.arange(C, device=device).unsqueeze(1) * N
        mask_indices = (
            time_indices.unsqueeze(1) + channel_offsets.unsqueeze(0)
        ).reshape(B, num_masked)

        validity_mask = (
            channel_mask.unsqueeze(2)
            .expand(B, C, num_time_masked)
            .reshape(B, num_masked)
        )

        return mask_indices, validity_mask


@dataclass(frozen=True)
class ChannelMasking(MaskingStrategy):
    """Masks entire channels (all time positions).

    Selects ``floor(mask_ratio * C_pad)`` channels and masks all ``N`` time
    positions for each. Channel selection is biased toward real channels
    via noise offset (fully vectorized).

    Total ``num_masked = num_channels_masked * N`` (fixed).
    """

    def __call__(self, num_channels, num_time_tokens, channel_mask):
        B = channel_mask.shape[0]
        device = channel_mask.device
        C, N = num_channels, num_time_tokens
        num_channels_masked = max(1, int(self.mask_ratio * C))
        num_masked = num_channels_masked * N

        noise = torch.rand(B, C, device=device)
        noise = noise + (~channel_mask).float() * 2.0
        selected = noise.argsort(dim=1)[:, :num_channels_masked]

        time_offsets = torch.arange(N, device=device)
        channel_starts = selected.unsqueeze(2) * N
        mask_indices = (
            channel_starts + time_offsets.unsqueeze(0).unsqueeze(0)
        ).reshape(B, num_masked)

        selected_real = torch.gather(channel_mask, 1, selected)
        validity_mask = (
            selected_real.unsqueeze(2)
            .expand(B, num_channels_masked, N)
            .reshape(B, num_masked)
        )

        return mask_indices, validity_mask
