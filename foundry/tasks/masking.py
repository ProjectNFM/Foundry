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
from typing import Optional

import torch


def build_token_validity_mask(
    channel_mask: torch.BoolTensor,
    num_time_tokens: int,
    input_seq_len: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    """Build a ``(B, C_pad * N)`` validity mask over the full token grid.

    A token is valid when its channel is real (from ``channel_mask``) **and**
    its time position is within the sample's true sequence length.

    Args:
        channel_mask: ``(B, C_pad)`` which channels are real.
        num_time_tokens: ``N`` temporal tokens per channel.
        input_seq_len: ``(B,)`` per-item true time-token count.
            When ``None``, all time positions are considered valid
            (fixed-length grid).

    Returns:
        ``(B, C_pad * N)`` boolean tensor on the same device as
        ``channel_mask``.
    """
    B, C_pad = channel_mask.shape
    N = num_time_tokens
    device = channel_mask.device

    # Channel validity: (B, C_pad) → (B, C_pad, N) → (B, C_pad * N)
    validity = (
        channel_mask.unsqueeze(2).expand(B, C_pad, N).reshape(B, C_pad * N)
    )

    if input_seq_len is not None:
        # Time validity: token index within each channel block < seq_len
        time_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        time_valid = time_idx < input_seq_len.unsqueeze(1)  # (B, N)
        # Tile across channels: (B, N) → (B, C_pad * N)
        time_valid = (
            time_valid.unsqueeze(1).expand(B, C_pad, N).reshape(B, C_pad * N)
        )
        validity = validity & time_valid

    return validity


@dataclass(frozen=True)
class MaskingStrategy:
    """Base for token-level masking strategies."""

    mask_ratio: float

    def __call__(
        self,
        num_channels: int,
        num_time_tokens: int,
        channel_mask: torch.BoolTensor,
        device: torch.device | None = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Args:
            num_channels: C_pad (total padded channels).
            num_time_tokens: N (temporal tokens per channel).
            channel_mask: (B, C_pad) which channels are real.
            device: Target device for output tensors. Defaults to
                ``channel_mask.device``.

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

    def __post_init__(self):
        if not (0 < self.mask_ratio < 1):
            raise ValueError(
                f"mask_ratio must be in (0, 1), got {self.mask_ratio}"
            )

    def __call__(
        self, num_channels, num_time_tokens, channel_mask, device=None
    ):
        device = device if device is not None else channel_mask.device
        B = channel_mask.shape[0]
        C, N = num_channels, num_time_tokens
        total_tokens = C * N
        num_masked = max(1, int(self.mask_ratio * total_tokens))

        noise = torch.rand(B, total_tokens, device=device)
        indices = noise.argsort(dim=1)
        mask_indices = indices[:, :num_masked]

        channel_of_token = mask_indices // N
        validity_mask = torch.gather(
            channel_mask.to(device), 1, channel_of_token
        )

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

    def __post_init__(self):
        if not (0 < self.mask_ratio < 1):
            raise ValueError(
                f"mask_ratio must be in (0, 1), got {self.mask_ratio}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

    def __call__(
        self, num_channels, num_time_tokens, channel_mask, device=None
    ):
        B = channel_mask.shape[0]
        device = device if device is not None else channel_mask.device
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
            channel_mask.to(device)
            .unsqueeze(2)
            .expand(B, C, num_time_masked)
            .reshape(B, num_masked)
        )

        return mask_indices, validity_mask


@dataclass(frozen=True)
class ChannelMasking(MaskingStrategy):
    """Masks entire channels (all time positions).

    Selects ``min(floor(mask_ratio * C_pad), C_pad - 1)`` channels and masks
    all ``N`` time positions for each. Channel selection is biased toward real
    channels via noise offset (fully vectorized). One real channel per sample
    is always protected from masking so the encoder never receives an input
    with zero real channels.

    Total ``num_masked = num_channels_masked * N`` (fixed).

    Raises:
        ValueError: If ``num_channels == 1``, because masking the only channel
            would leave zero visible channels.  Use :class:`RandomTokenMasking`
            for single-channel data.
    """

    def __post_init__(self):
        if not (0 < self.mask_ratio < 1):
            raise ValueError(
                f"mask_ratio must be in (0, 1), got {self.mask_ratio}"
            )

    def __call__(
        self, num_channels, num_time_tokens, channel_mask, device=None
    ):
        if num_channels < 2:
            raise ValueError(
                f"ChannelMasking requires num_channels >= 2, got "
                f"{num_channels}. Use RandomTokenMasking for single-channel "
                f"data."
            )
        B = channel_mask.shape[0]
        device = device if device is not None else channel_mask.device
        C, N = num_channels, num_time_tokens
        num_channels_masked = max(1, int(self.mask_ratio * C))
        num_channels_masked = min(num_channels_masked, C - 1)
        num_masked = num_channels_masked * N

        channel_mask_dev = channel_mask.to(device)
        noise = torch.rand(B, C, device=device)

        # Protect one real channel per sample so it is never masked.
        # Pick the real channel with the highest random noise (arbitrary
        # but uniform), then boost it above all others so it sorts last.
        protect_noise = noise.clone()
        protect_noise[~channel_mask_dev] = -float("inf")
        _, protected_idx = protect_noise.max(dim=1)
        noise.scatter_add_(
            1,
            protected_idx.unsqueeze(1),
            torch.full((B, 1), 4.0, device=device),
        )

        noise = noise + (~channel_mask_dev).float() * 2.0
        selected = noise.argsort(dim=1)[:, :num_channels_masked]

        time_offsets = torch.arange(N, device=device)
        channel_starts = selected.unsqueeze(2) * N
        mask_indices = (
            channel_starts + time_offsets.unsqueeze(0).unsqueeze(0)
        ).reshape(B, num_masked)

        selected_real = torch.gather(channel_mask_dev, 1, selected)
        validity_mask = (
            selected_real.unsqueeze(2)
            .expand(B, num_channels_masked, N)
            .reshape(B, num_masked)
        )

        return mask_indices, validity_mask
