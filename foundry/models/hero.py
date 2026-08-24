"""HERO: Hierarchical EEG Representation model.

Implements a three-level temporal hierarchy (fine 128 Hz, mid 32 Hz,
coarse 8 Hz) with shared local channel encoding, spatial-slot fusion,
local window attention, temporal reduction, aligned gated top-down residuals,
and task-specific cross-attention readout. Exposes a public ``encode()`` API
returning
:class:`Representation` and a ``forward()`` adapter returning
:class:`~foundry.models.ssl_meta.ModelOutput` for the standard Foundry
training loop.

This module does **not** reuse :class:`EEGTokenizer` or
:class:`PerceiverIOBackbone` -- those are POYO-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_brain.batching import chain, pad8

from foundry.models.readout import build_readout_router
from foundry.models.ssl_meta import ModelOutput
from foundry.tasks.targets import extract_multitask_targets

if TYPE_CHECKING:
    from torch_brain.data import Data
    from foundry.tasks.config import TaskConfig

__all__ = ["HEROModel", "Representation", "TaskQueryCrossAttention"]


TemporalMode = Literal["hierarchical", "flat"]
TemporalReductionMode = Literal["slots", "mean"]
TopDownFusionMode = Literal["gated", "add"]


# ---------------------------------------------------------------------------
# Public output types
# ---------------------------------------------------------------------------


@dataclass
class CoverageInfo:
    """Observation metadata propagated through the hierarchy.

    All masks use ``True`` for valid / observed positions.
    """

    fine_valid: torch.Tensor
    mid_valid: torch.Tensor
    coarse_valid: torch.Tensor
    sample_support: torch.Tensor
    channel_count: torch.Tensor
    channel_fraction: torch.Tensor
    fine_rf_intervals: torch.Tensor
    mid_rf_intervals: torch.Tensor
    coarse_rf_intervals: torch.Tensor


@dataclass
class Representation:
    """Public encoder output contract.

    Attributes:
        content: ``[B, T_fine, D]`` fused fine-level features.
        content_timestamps: ``[B, T_fine]`` physical timestamps.
        coverage: Per-level validity masks and receptive-field metadata.
    """

    content: torch.Tensor
    content_timestamps: torch.Tensor
    coverage: CoverageInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_kaiser_lowpass_kernel(
    num_taps: int = 33,
    cutoff: float = 0.1125,
    beta: float = 6.0,
) -> torch.Tensor:
    """Fixed symmetric Kaiser-windowed sinc low-pass filter."""
    half = (num_taps - 1) / 2.0
    n = torch.arange(num_taps, dtype=torch.float64)
    sinc_arg = 2.0 * cutoff * (n - half)
    # Ideal discrete-time low-pass: 2 fc sinc(2 fc n).  The 2 fc factor is
    # required on every tap, not only at the removable center singularity.
    h = 2.0 * cutoff * torch.sinc(sinc_arg)
    window = torch.kaiser_window(num_taps, periodic=False, beta=beta).to(
        torch.float64
    )
    h = h * window
    h = h / h.sum()
    return h.float()


def _timestamp_intervals(
    timestamps: torch.Tensor,
    fallback_step: float,
) -> torch.Tensor:
    """Return sample-bin intervals centered on monotonically increasing times."""
    if timestamps.shape[1] == 1:
        half_step = fallback_step / 2
        return torch.stack(
            [timestamps - half_step, timestamps + half_step], dim=-1
        )

    boundaries = (timestamps[:, 1:] + timestamps[:, :-1]) / 2
    first = timestamps[:, :1] - (boundaries[:, :1] - timestamps[:, :1])
    last = timestamps[:, -1:] + (timestamps[:, -1:] - boundaries[:, -1:])
    starts = torch.cat([first, boundaries], dim=1)
    ends = torch.cat([boundaries, last], dim=1)
    return torch.stack([starts, ends], dim=-1)


def _expand_receptive_fields(
    intervals: torch.Tensor,
    left: int,
    right: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Union neighboring receptive fields without extending beyond the input."""
    if left == 0 and right == 0:
        return intervals

    window = left + right + 1
    starts = intervals[..., 0]
    ends = intervals[..., 1]
    if valid_mask is not None:
        starts = starts.masked_fill(~valid_mask, float("inf"))
        ends = ends.masked_fill(~valid_mask, float("-inf"))
    starts = F.pad(starts, (left, right), value=float("inf"))
    ends = F.pad(ends, (left, right), value=float("-inf"))
    starts = starts.unfold(1, window, 1).amin(dim=-1)
    ends = ends.unfold(1, window, 1).amax(dim=-1)
    expanded = torch.stack([starts, ends], dim=-1)
    if valid_mask is not None:
        expanded = torch.where(valid_mask.unsqueeze(-1), expanded, intervals)
    return expanded


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class SharedLocalChannelEncoder(nn.Module):
    """Three-layer 1-D causal conv stack applied identically per channel.

    Input:  ``(B*C, 1, T)``
    Output: ``(B*C, D, T)`` reshaped externally to ``(B, C, T, D)``.

    Each layer uses left-padding of ``kernel_size - 1`` for causal
    alignment, followed by per-time-step LayerNorm and GELU. LayerNorm is
    intentionally applied only across features so masked padding and future
    samples cannot change an earlier representation.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

        layers = nn.ModuleList()
        in_ch = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, embed_dim, kernel_size, bias=False))
            layers.append(nn.LayerNorm(embed_dim))
            layers.append(nn.GELU())
            in_ch = embed_dim
        self.layers = layers

    def forward(
        self, x: torch.Tensor, sample_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: ``(BC, 1, T)`` raw channel signals.
            sample_mask: ``(BC, T)`` boolean validity mask. Invalid samples
                are zeroed before every conv and invalid outputs zeroed after
                every block.

        Returns:
            ``(BC, D, T)``
        """
        for i in range(0, len(self.layers), 3):
            conv = self.layers[i]
            norm = self.layers[i + 1]
            act = self.layers[i + 2]

            if sample_mask is not None:
                mask_f = sample_mask.unsqueeze(1).float()
                x = x * mask_f

            x = F.pad(x, (self.pad, 0))
            x = conv(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = act(x)

            if sample_mask is not None:
                x = x * mask_f
        return x


class SpatialSlotMixer(nn.Module):
    """Fuse an unordered set of channel-local features at each time bin.

    Uses ``num_slots`` learned queries that cross-attend to the channel
    dimension, concatenates the slot outputs, and applies a gated linear
    projection to produce one D-dim fused token per time bin.

    The operation is permutation-invariant in the channel dimension.
    """

    def __init__(
        self, embed_dim: int = 256, num_slots: int = 8, num_heads: int = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.queries = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.gate_proj = nn.Linear(num_slots * embed_dim, embed_dim)
        self.out_proj = nn.Linear(num_slots * embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, C, T, D)`` channel-local features.
            channel_mask: ``(B, C)`` boolean mask (True = valid).

        Returns:
            Tuple of fused ``(B, T, D)`` and token validity ``(B, T)``.
        """
        B, C, T, D = x.shape
        S = self.num_slots
        H = self.num_heads
        Dh = self.head_dim

        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C, D)

        q = self.q_proj(self.queries).unsqueeze(0).expand(B * T, -1, -1)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.view(B * T, S, H, Dh).transpose(1, 2)
        k = k.view(B * T, C, H, Dh).transpose(1, 2)
        v = v.view(B * T, C, H, Dh).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if channel_mask is not None:
            mask_bt = channel_mask.unsqueeze(2).expand(B, C, T)
            mask_bt = mask_bt.permute(0, 2, 1).reshape(B * T, C)
            mask_4d = mask_bt.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask_4d, float("-inf"))

            any_valid = mask_bt.any(dim=1)
        else:
            any_valid = torch.ones(B * T, dtype=torch.bool, device=x.device)

        safe_rows = any_valid.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attn_weights = torch.where(
            safe_rows.expand_as(attn_weights),
            attn_weights,
            torch.zeros_like(attn_weights),
        )
        attn_probs = torch.where(
            safe_rows.expand_as(attn_weights),
            F.softmax(attn_weights, dim=-1),
            torch.zeros_like(attn_weights),
        )

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).reshape(B * T, S, D)
        concat = attn_out.reshape(B * T, S * D)

        gate = torch.sigmoid(self.gate_proj(concat))
        out = self.out_proj(concat) * gate

        out = out.where(
            any_valid.unsqueeze(1).expand_as(out),
            torch.zeros_like(out),
        )

        out = self.layer_norm(out)
        out = out.where(any_valid.unsqueeze(1), torch.zeros_like(out))
        out = out.view(B, T, D)
        token_valid = any_valid.view(B, T)
        return out, token_valid


class LocalWindowAttention(nn.Module):
    """Bidirectional multi-head self-attention within a sliding window.

    Window: ``[t - half, t + half]`` inclusive, truncated at edges.
    Uses learned relative-time bias computed from physical timestamps.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.half_w = window_size // 2

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.rel_time_proj = nn.Linear(1, num_heads, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, D)``
            timestamps: ``(B, T)`` physical timestamps.
            valid_mask: ``(B, T)`` boolean (True = valid).

        Returns:
            ``(B, T, D)`` with invalid positions zeroed.
        """
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        residual = x
        x_n = self.norm1(x)
        qkv = self.qkv(x_n).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        left_pad = self.half_w - 1
        right_pad = self.half_w
        padded_k = F.pad(k, (0, 0, left_pad, right_pad), value=0.0)
        padded_v = F.pad(v, (0, 0, left_pad, right_pad), value=0.0)

        padded_ts = F.pad(timestamps, (left_pad, right_pad), value=0.0)
        if valid_mask is not None:
            padded_valid = F.pad(
                valid_mask.float(), (left_pad, right_pad), value=0.0
            )
        else:
            padded_valid = F.pad(
                torch.ones(B, T, device=x.device),
                (left_pad, right_pad),
                value=0.0,
            )

        W = self.window_size
        k_windows = padded_k.unfold(2, W, 1)
        v_windows = padded_v.unfold(2, W, 1)
        ts_windows = padded_ts.unfold(1, W, 1)
        valid_windows = padded_valid.unfold(1, W, 1)

        k_windows = k_windows.permute(0, 1, 2, 4, 3)
        v_windows = v_windows.permute(0, 1, 2, 4, 3)

        attn_logits = torch.einsum(
            "bhqd,bhqwd->bhqw", q, k_windows
        ) / math.sqrt(Dh)

        dt = ts_windows - timestamps.unsqueeze(-1)
        rel_bias = self.rel_time_proj(dt.unsqueeze(-1))
        rel_bias = rel_bias.permute(0, 3, 1, 2)
        attn_logits = attn_logits + rel_bias

        key_mask = valid_windows.unsqueeze(1).bool()
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

        any_key_valid = key_mask.any(dim=-1, keepdim=True)
        safe_logits = torch.where(
            any_key_valid.expand_as(attn_logits),
            attn_logits,
            torch.zeros_like(attn_logits),
        )
        attn_probs = torch.where(
            any_key_valid.expand_as(attn_logits),
            F.softmax(safe_logits, dim=-1),
            torch.zeros_like(attn_logits),
        )

        out = torch.einsum("bhqw,bhqwd->bhqd", attn_probs, v_windows)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        x = residual + out
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).float()

        residual = x
        x = residual + self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).float()

        return x

    def propagate_receptive_fields(
        self,
        intervals: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand intervals by exactly the keys visible to each query."""
        return _expand_receptive_fields(
            intervals,
            left=self.half_w - 1,
            right=self.half_w,
            valid_mask=valid_mask,
        )


class TemporalReduction(nn.Module):
    """Local anti-aliased 4x downsampling with selectable aggregation.

    Steps:
    1. Fixed 33-tap Kaiser-windowed-sinc depthwise low-pass.
    2. Compute coverage from validity mask convolved with absolute kernel.
    3. Group into 8-token neighborhoods with stride 4.
    4. Aggregate each group with learned temporal slots (reference) or a
       mask-aware mean (no-temporal-slots control).
    5. Project and normalize to D dims.

    Drops partial tails: ``T_out = T_in // 4``.
    """

    SUPPORT_THRESHOLD = 0.999

    def __init__(
        self,
        embed_dim: int = 256,
        num_temporal_slots: int = 4,
        num_heads: int = 8,
        stride: int = 4,
        neighborhood: int = 8,
        aggregation: TemporalReductionMode = "slots",
    ):
        super().__init__()
        if aggregation not in ("slots", "mean"):
            raise ValueError(
                "aggregation must be either 'slots' or 'mean', "
                f"got {aggregation!r}."
            )
        self.embed_dim = embed_dim
        self.num_temporal_slots = num_temporal_slots
        self.stride = stride
        self.neighborhood = neighborhood
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.aggregation = aggregation

        kernel = _build_kaiser_lowpass_kernel()
        self.register_buffer("lp_kernel", kernel.unsqueeze(0).unsqueeze(0))
        abs_kernel = kernel.abs()
        abs_kernel = abs_kernel / abs_kernel.sum()
        self.register_buffer("abs_kernel", abs_kernel.unsqueeze(0).unsqueeze(0))

        if aggregation == "slots":
            if num_temporal_slots <= 0:
                raise ValueError(
                    "num_temporal_slots must be positive when aggregation='slots'."
                )
            self.slot_queries = nn.Parameter(
                torch.randn(num_temporal_slots, embed_dim) * 0.02
            )
            self.slot_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.slot_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.slot_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.gate_proj = nn.Linear(
                num_temporal_slots * embed_dim, embed_dim
            )
            self.out_proj = nn.Linear(num_temporal_slots * embed_dim, embed_dim)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _compute_support(
        self, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return filter support fractions and fully supported positions."""
        valid_f = valid_mask.float().unsqueeze(1)
        pad_size = self.abs_kernel.shape[-1] // 2
        support = F.conv1d(
            F.pad(valid_f, (pad_size, pad_size), mode="constant", value=0.0),
            self.abs_kernel.to(device=valid_f.device, dtype=valid_f.dtype),
        ).squeeze(1)
        return support, support >= self.SUPPORT_THRESHOLD

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, T, D)``
            timestamps: ``(B, T)``
            valid_mask: ``(B, T)`` boolean.

        Returns:
            Tuple of:
            - reduced features ``(B, T_out, D)``
            - reduced timestamps ``(B, T_out)``
            - reduced validity ``(B, T_out)``
            - support weights ``(B, T_out)``
        """
        B, T, D = x.shape
        device = x.device
        T_out = T // self.stride

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        x_lp = self._apply_lowpass(x, valid_mask)

        support, out_valid = self._compute_support(valid_mask)

        usable_T = T_out * self.stride
        pad_needed = self.neighborhood - self.stride
        if pad_needed > 0:
            half_pad = pad_needed // 2
            x_padded = F.pad(
                x_lp[:, :usable_T].transpose(1, 2),
                (half_pad, pad_needed - half_pad),
            ).transpose(1, 2)
            valid_padded = F.pad(
                out_valid[:, :usable_T].float(),
                (half_pad, pad_needed - half_pad),
            )
        else:
            x_padded = x_lp[:, :usable_T]
            valid_padded = out_valid[:, :usable_T].float()

        neighborhoods = x_padded.unfold(1, self.neighborhood, self.stride)
        neighborhoods = neighborhoods.permute(0, 1, 3, 2)
        neigh_valid = valid_padded.unfold(
            1, self.neighborhood, self.stride
        ).bool()

        if self.aggregation == "slots":
            out = self._aggregate_with_slots(neighborhoods, neigh_valid)
        else:
            weights = neigh_valid.unsqueeze(-1).to(neighborhoods.dtype)
            pooled = (neighborhoods * weights).sum(dim=2)
            pooled = pooled / weights.sum(dim=2).clamp(min=1.0)
            out = self.layer_norm(self.out_proj(pooled))

        out_timestamps = timestamps[
            :, self.stride // 2 : usable_T : self.stride
        ]
        out_valid = out_valid[:, self.stride // 2 : usable_T : self.stride]
        out_support = support[:, self.stride // 2 : usable_T : self.stride]

        if out_timestamps.shape[1] > T_out:
            out_timestamps = out_timestamps[:, :T_out]
            out_valid = out_valid[:, :T_out]
            out_support = out_support[:, :T_out]
        elif out_timestamps.shape[1] < T_out:
            deficit = T_out - out_timestamps.shape[1]
            out_timestamps = F.pad(out_timestamps, (0, deficit))
            out_valid = F.pad(out_valid, (0, deficit), value=False)
            out_support = F.pad(out_support, (0, deficit))

        out = out * out_valid.unsqueeze(-1).float()

        return out, out_timestamps, out_valid, out_support

    def _aggregate_with_slots(
        self,
        neighborhoods: torch.Tensor,
        neigh_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the reference learned-slot aggregation to local groups."""
        B, T_out, N, D = neighborhoods.shape
        S = self.num_temporal_slots
        H = self.num_heads
        Dh = self.head_dim
        BT = B * T_out
        neigh_flat = neighborhoods.reshape(BT, N, D)
        neigh_valid_flat = neigh_valid.reshape(BT, N)

        q = self.slot_q_proj(self.slot_queries).unsqueeze(0).expand(BT, -1, -1)
        k = self.slot_k_proj(neigh_flat)
        v = self.slot_v_proj(neigh_flat)

        q = q.view(BT, S, H, Dh).transpose(1, 2)
        k = k.view(BT, N, H, Dh).transpose(1, 2)
        v = v.view(BT, N, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        kv_mask = neigh_valid_flat.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~kv_mask, float("-inf"))

        any_valid = neigh_valid_flat.any(dim=-1, keepdim=True)
        any_valid_4d = any_valid.unsqueeze(1).unsqueeze(-1).expand_as(attn)
        safe_attn = torch.where(any_valid_4d, attn, torch.zeros_like(attn))
        attn_probs = torch.where(
            any_valid_4d,
            F.softmax(safe_attn, dim=-1),
            torch.zeros_like(attn),
        )

        slot_out = torch.matmul(attn_probs, v)
        slot_out = slot_out.transpose(1, 2).reshape(BT, S, D)
        concat = slot_out.reshape(BT, S * D)
        gate = torch.sigmoid(self.gate_proj(concat))
        out = self.out_proj(concat) * gate
        return self.layer_norm(out).view(B, T_out, D)

    def _apply_lowpass(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Depthwise low-pass filter with zero-fill for invalid samples."""
        B, T, D = x.shape
        x_filt = x * valid_mask.unsqueeze(-1).float()

        x_filt = x_filt.transpose(1, 2).reshape(B * D, 1, T)
        pad_size = self.lp_kernel.shape[-1] // 2
        x_filt = F.pad(x_filt, (pad_size, pad_size), mode="constant", value=0.0)
        kernel = self.lp_kernel.to(device=x_filt.device, dtype=x_filt.dtype)
        x_filt = F.conv1d(x_filt, kernel)
        x_filt = x_filt.view(B, D, T).transpose(1, 2)
        return x_filt

    def propagate_receptive_fields(
        self,
        intervals: torch.Tensor,
        input_valid: torch.Tensor,
        output_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate dependency bounds through filtering and local groups."""
        lowpass_radius = self.lp_kernel.shape[-1] // 2
        filtered = _expand_receptive_fields(
            intervals,
            left=lowpass_radius,
            right=lowpass_radius,
            valid_mask=input_valid,
        )
        _, filtered_valid = self._compute_support(input_valid)

        output_length = intervals.shape[1] // self.stride
        usable_length = output_length * self.stride
        pad_needed = self.neighborhood - self.stride
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad

        starts = filtered[:, :usable_length, 0].masked_fill(
            ~filtered_valid[:, :usable_length], float("inf")
        )
        ends = filtered[:, :usable_length, 1].masked_fill(
            ~filtered_valid[:, :usable_length], float("-inf")
        )
        starts = F.pad(
            starts,
            (left_pad, right_pad),
            value=float("inf"),
        )
        ends = F.pad(
            ends,
            (left_pad, right_pad),
            value=float("-inf"),
        )
        starts = starts.unfold(1, self.neighborhood, self.stride).amin(dim=-1)
        ends = ends.unfold(1, self.neighborhood, self.stride).amax(dim=-1)
        reduced = torch.stack([starts, ends], dim=-1)
        fallback = intervals[:, self.stride // 2 : usable_length : self.stride]
        return torch.where(output_valid.unsqueeze(-1), reduced, fallback)


class AlignedGatedResidual(nn.Module):
    """Align and fuse coarser features via receptive-field overlap.

    For each fine token, reads the bounded set of coarse tokens whose
    receptive-field intervals overlap that fine token's interval, uses
    normalized overlap weights, and projects before gated (reference) or
    direct addition. The fine residual is always preserved.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        fusion: TopDownFusionMode = "gated",
    ):
        super().__init__()
        if fusion not in ("gated", "add"):
            raise ValueError(
                f"fusion must be either 'gated' or 'add', got {fusion!r}."
            )
        self.fusion = fusion
        self.proj = nn.Linear(embed_dim, embed_dim)
        if fusion == "gated":
            self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        fine_rf_intervals: torch.Tensor,
        coarse_rf_intervals: torch.Tensor,
        coarse_valid: torch.Tensor | None = None,
        fine_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fine: ``(B, T_fine, D)``
            coarse: ``(B, T_coarse, D)``
            fine_rf_intervals: ``(B, T_fine, 2)`` dependency bounds.
            coarse_rf_intervals: ``(B, T_coarse, 2)`` dependency bounds.
            coarse_valid: ``(B, T_coarse)`` boolean.
            fine_valid: ``(B, T_fine)`` boolean.

        Returns:
            Updated fine features and their expanded receptive-field intervals.
        """
        B, T_f, D = fine.shape
        T_c = coarse.shape[1]
        if T_c == 0:
            return fine, fine_rf_intervals

        fine_starts = fine_rf_intervals[..., 0].contiguous()
        fine_ends = fine_rf_intervals[..., 1].contiguous()
        coarse_starts = coarse_rf_intervals[..., 0].contiguous()
        coarse_ends = coarse_rf_intervals[..., 1].contiguous()

        # Receptive fields are ordered in time. Search their boundaries to find
        # only the bounded candidate range for each fine token instead of
        # materializing a T_fine x T_coarse matrix.
        first = torch.searchsorted(coarse_ends, fine_starts, right=False)
        stop = torch.searchsorted(coarse_starts, fine_ends, right=True)
        candidate_counts = (stop - first).clamp(min=0)
        max_candidates = max(1, int(candidate_counts.max().item()))

        offsets = torch.arange(max_candidates, device=fine.device)
        candidate_mask = offsets.view(1, 1, -1) < candidate_counts.unsqueeze(-1)
        candidate_indices = first.unsqueeze(-1) + offsets.view(1, 1, -1)
        candidate_indices = candidate_indices.clamp(max=T_c - 1)
        batch_indices = torch.arange(B, device=fine.device).view(B, 1, 1)

        candidate_intervals = coarse_rf_intervals[
            batch_indices, candidate_indices
        ]
        overlap_starts = torch.maximum(
            fine_starts.unsqueeze(-1), candidate_intervals[..., 0]
        )
        overlap_ends = torch.minimum(
            fine_ends.unsqueeze(-1), candidate_intervals[..., 1]
        )
        overlap = (overlap_ends - overlap_starts).clamp(min=0)
        overlap = overlap * candidate_mask.to(overlap.dtype)

        if coarse_valid is not None:
            candidate_valid = coarse_valid[batch_indices, candidate_indices]
            overlap = overlap * candidate_valid.to(overlap.dtype)
        if fine_valid is not None:
            overlap = overlap * fine_valid.unsqueeze(-1).to(overlap.dtype)

        overlap_sum = overlap.sum(dim=-1, keepdim=True)
        weights = overlap / overlap_sum.clamp(min=1e-8)
        coarse_aligned = torch.zeros_like(fine)
        # Bound temporary feature memory even when broad coarse receptive fields
        # overlap many fine tokens. Candidate count depends on architecture, not
        # bounded by the architecture's finite context rather than the total
        # recording duration, so this remains linear for long sequences.
        candidate_chunk_size = 8
        for chunk_start in range(0, max_candidates, candidate_chunk_size):
            chunk_stop = min(chunk_start + candidate_chunk_size, max_candidates)
            chunk_indices = candidate_indices[..., chunk_start:chunk_stop]
            chunk_features = coarse[batch_indices, chunk_indices]
            chunk_weights = weights[..., chunk_start:chunk_stop].unsqueeze(-1)
            coarse_aligned = coarse_aligned + (
                chunk_weights * chunk_features
            ).sum(dim=2)
        coarse_aligned = self.layer_norm(coarse_aligned)

        projected = self.proj(coarse_aligned)

        has_overlap = (overlap_sum > 0).to(fine.dtype)
        if self.fusion == "gated":
            projected = (
                torch.sigmoid(self.gate_proj(coarse_aligned)) * projected
            )
        result = fine + projected * has_overlap

        if fine_valid is not None:
            result = result * fine_valid.unsqueeze(-1).float()

        contributes = overlap > 0
        candidate_starts = candidate_intervals[..., 0].masked_fill(
            ~contributes, float("inf")
        )
        candidate_ends = candidate_intervals[..., 1].masked_fill(
            ~contributes, float("-inf")
        )
        aligned_starts = candidate_starts.amin(dim=-1)
        aligned_ends = candidate_ends.amax(dim=-1)
        result_starts = torch.where(
            has_overlap.squeeze(-1).bool(),
            torch.minimum(fine_starts, aligned_starts),
            fine_starts,
        )
        result_ends = torch.where(
            has_overlap.squeeze(-1).bool(),
            torch.maximum(fine_ends, aligned_ends),
            fine_ends,
        )
        result_intervals = torch.stack([result_starts, result_ends], dim=-1)
        return result, result_intervals


class HEROEncoder(nn.Module):
    """Orchestrates the three-level hierarchy.

    Fine (128 Hz) -> LocalWindowAttention x2
    -> TemporalReduction -> Mid (32 Hz) -> LocalWindowAttention x2
    -> TemporalReduction -> Coarse (8 Hz) -> LocalWindowAttention x2
    -> AlignedGatedResidual coarse->mid -> AlignedGatedResidual mid->fine
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        num_local_attn_blocks: int = 2,
        local_window: int = 32,
        num_temporal_slots: int = 4,
        temporal_reduction: TemporalReductionMode = "slots",
        top_down_fusion: TopDownFusionMode = "gated",
    ):
        super().__init__()
        self.fine_attns = nn.ModuleList(
            [
                LocalWindowAttention(embed_dim, num_attn_heads, local_window)
                for _ in range(num_local_attn_blocks)
            ]
        )
        self.fine_to_mid = TemporalReduction(
            embed_dim,
            num_temporal_slots,
            num_attn_heads,
            aggregation=temporal_reduction,
        )
        self.mid_attns = nn.ModuleList(
            [
                LocalWindowAttention(embed_dim, num_attn_heads, local_window)
                for _ in range(num_local_attn_blocks)
            ]
        )
        self.mid_to_coarse = TemporalReduction(
            embed_dim,
            num_temporal_slots,
            num_attn_heads,
            aggregation=temporal_reduction,
        )
        self.coarse_attns = nn.ModuleList(
            [
                LocalWindowAttention(embed_dim, num_attn_heads, local_window)
                for _ in range(num_local_attn_blocks)
            ]
        )

        self.coarse_to_mid_align = AlignedGatedResidual(
            embed_dim, fusion=top_down_fusion
        )
        self.mid_to_fine_align = AlignedGatedResidual(
            embed_dim, fusion=top_down_fusion
        )

    def forward(
        self,
        x_fine: torch.Tensor,
        fine_timestamps: torch.Tensor,
        fine_rf_intervals: torch.Tensor,
        fine_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x_fine: ``(B, T_fine, D)`` from spatial-slot mixer.
            fine_timestamps: ``(B, T_fine)``
            fine_rf_intervals: ``(B, T_fine, 2)`` dependency bounds from the
                local channel encoder.
            fine_valid: ``(B, T_fine)`` boolean mask.

        Returns:
            Tuple of fused fine features ``(B, T_fine, D)`` and a dict of
            per-level metadata (timestamps, validity, support, features).
        """
        for attn in self.fine_attns:
            x_fine = attn(x_fine, fine_timestamps, fine_valid)
            fine_rf_intervals = attn.propagate_receptive_fields(
                fine_rf_intervals, fine_valid
            )

        x_mid, mid_ts, mid_valid, mid_support = self.fine_to_mid(
            x_fine, fine_timestamps, fine_valid
        )
        mid_rf_intervals = self.fine_to_mid.propagate_receptive_fields(
            fine_rf_intervals,
            fine_valid,
            mid_valid,
        )

        for attn in self.mid_attns:
            x_mid = attn(x_mid, mid_ts, mid_valid)
            mid_rf_intervals = attn.propagate_receptive_fields(
                mid_rf_intervals, mid_valid
            )

        x_coarse, coarse_ts, coarse_valid, coarse_support = self.mid_to_coarse(
            x_mid, mid_ts, mid_valid
        )
        coarse_rf_intervals = self.mid_to_coarse.propagate_receptive_fields(
            mid_rf_intervals,
            mid_valid,
            coarse_valid,
        )

        for attn in self.coarse_attns:
            x_coarse = attn(x_coarse, coarse_ts, coarse_valid)
            coarse_rf_intervals = attn.propagate_receptive_fields(
                coarse_rf_intervals, coarse_valid
            )

        x_mid, mid_rf_intervals = self.coarse_to_mid_align(
            x_mid,
            x_coarse,
            mid_rf_intervals,
            coarse_rf_intervals,
            coarse_valid=coarse_valid,
            fine_valid=mid_valid,
        )

        x_fine, fine_rf_intervals = self.mid_to_fine_align(
            x_fine,
            x_mid,
            fine_rf_intervals,
            mid_rf_intervals,
            coarse_valid=mid_valid,
            fine_valid=fine_valid,
        )

        meta = {
            "mid_timestamps": mid_ts,
            "mid_valid": mid_valid,
            "mid_support": mid_support,
            "fine_rf_intervals": fine_rf_intervals,
            "mid_rf_intervals": mid_rf_intervals,
            "coarse_timestamps": coarse_ts,
            "coarse_valid": coarse_valid,
            "coarse_support": coarse_support,
            "coarse_rf_intervals": coarse_rf_intervals,
        }
        return x_fine, meta


class FlatTemporalEncoder(nn.Module):
    """Flat control over the fine stream with no temporal rate hierarchy.

    This control deliberately starts after the same channel encoder and
    spatial-slot mixer as the reference model. ``num_local_attn_blocks`` is
    independent so an experiment can select a parameter- or compute-matched
    flat depth without altering the shared pre-fusion path.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        num_local_attn_blocks: int = 2,
        local_window: int = 32,
    ):
        super().__init__()
        self.fine_attns = nn.ModuleList(
            [
                LocalWindowAttention(embed_dim, num_attn_heads, local_window)
                for _ in range(num_local_attn_blocks)
            ]
        )

    def forward(
        self,
        x_fine: torch.Tensor,
        fine_timestamps: torch.Tensor,
        fine_rf_intervals: torch.Tensor,
        fine_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for attn in self.fine_attns:
            x_fine = attn(x_fine, fine_timestamps, fine_valid)
            fine_rf_intervals = attn.propagate_receptive_fields(
                fine_rf_intervals, fine_valid
            )

        B = x_fine.shape[0]
        empty = fine_timestamps.new_empty(B, 0)
        empty_valid = torch.empty(B, 0, dtype=torch.bool, device=x_fine.device)
        empty_intervals = fine_rf_intervals.new_empty(B, 0, 2)
        meta = {
            "mid_timestamps": empty,
            "mid_valid": empty_valid,
            "mid_support": empty,
            "fine_rf_intervals": fine_rf_intervals,
            "mid_rf_intervals": empty_intervals,
            "coarse_timestamps": empty,
            "coarse_valid": empty_valid.clone(),
            "coarse_support": empty.clone(),
            "coarse_rf_intervals": empty_intervals.clone(),
        }
        return x_fine, meta


class TaskQueryCrossAttention(nn.Module):
    """Read encoded content with learned task- and time-specific queries.

    Every requested output starts from an embedding for its task and attends
    over the complete fine-resolution representation. When output timestamps
    are available, each task/head also learns a temporal offset and span. A
    head can therefore specialize in anything from the immediate neighborhood
    of an output to long-range context elsewhere in the recording.

    Query chunks bound the temporary ``N_out x T_content`` attention matrix
    without changing the result.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tasks: int,
        num_heads: int = 8,
        query_chunk_size: int = 256,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive.")

        self.embed_dim = embed_dim
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_chunk_size = query_chunk_size

        # Keep an unused row when there are no downstream tasks so encoder-only
        # HERO instances remain constructible.
        self.task_queries = nn.Embedding(max(1, num_tasks), embed_dim)
        nn.init.normal_(self.task_queries.weight, std=0.02)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.content_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # softplus(-4) gives a weak, broad initial preference around the output
        # time. Tasks can increase it for short-range reads or drive it toward
        # zero for effectively global attention. Offsets allow heads to seek
        # task-relevant information before or after the requested timestamp.
        temporal_shape = (max(1, num_tasks), num_heads)
        self.log_time_decay = nn.Parameter(torch.full(temporal_shape, -4.0))
        self.time_offset = nn.Parameter(torch.zeros(temporal_shape))

    def forward(
        self,
        content: torch.Tensor,
        task_index: torch.Tensor,
        *,
        content_timestamps: torch.Tensor | None = None,
        output_timestamps: torch.Tensor | None = None,
        content_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one decoded embedding for every padded output query.

        Args:
            content: ``(B, T, D)`` encoder content.
            task_index: ``(B, N_out)`` padded 1-based task indices.
            content_timestamps: Optional ``(B, T)`` physical timestamps.
            output_timestamps: Optional ``(B, N_out)`` physical timestamps.
                Temporal bias is used only when both timestamp tensors are
                supplied.
            content_valid: Optional ``(B, T)`` validity mask.

        Returns:
            Decoded task embeddings with shape ``(B, N_out, D)``.
        """
        B, T, D = content.shape
        if D != self.embed_dim:
            raise ValueError(f"Expected content dim {self.embed_dim}, got {D}.")
        if task_index.ndim != 2 or task_index.shape[0] != B:
            raise ValueError("task_index must have shape [batch, outputs].")

        N_out = task_index.shape[1]
        if N_out == 0:
            return content.new_zeros(B, 0, D)

        use_time = (
            content_timestamps is not None or output_timestamps is not None
        )
        if use_time and (
            content_timestamps is None or output_timestamps is None
        ):
            raise ValueError(
                "content_timestamps and output_timestamps must be provided together."
            )
        if content_timestamps is not None and content_timestamps.shape != (
            B,
            T,
        ):
            raise ValueError(
                "content_timestamps must have shape [batch, content]."
            )
        if output_timestamps is not None and output_timestamps.shape != (
            B,
            N_out,
        ):
            raise ValueError(
                "output_timestamps must have shape [batch, outputs]."
            )

        if content_valid is None:
            content_valid = torch.ones(
                B, T, dtype=torch.bool, device=content.device
            )
        elif content_valid.shape != (B, T):
            raise ValueError("content_valid must have shape [batch, content].")

        task_ids = (task_index - 1).clamp(min=0)
        queries = self.task_queries(task_ids)

        H = self.num_heads
        Dh = self.head_dim
        normalized_content = self.content_norm(content)
        keys = self.k_proj(normalized_content)
        values = self.v_proj(normalized_content)
        keys = keys.view(B, T, H, Dh).transpose(1, 2)
        values = values.view(B, T, H, Dh).transpose(1, 2)

        any_content = content_valid.any(dim=1)
        key_mask = content_valid[:, None, None, :]
        decoded_chunks = []
        for start in range(0, N_out, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, N_out)
            query_chunk = queries[:, start:stop]
            chunk_task_ids = task_ids[:, start:stop]

            q = self.q_proj(self.query_norm(query_chunk))
            q = q.view(B, stop - start, H, Dh).transpose(1, 2)
            logits = torch.matmul(q, keys.transpose(-2, -1)) / math.sqrt(Dh)

            if content_timestamps is not None and output_timestamps is not None:
                query_timestamps = output_timestamps[:, start:stop].to(
                    content_timestamps
                )
                dt = (
                    content_timestamps[:, None, :]
                    - query_timestamps[:, :, None]
                )
                decay = F.softplus(self.log_time_decay[chunk_task_ids])
                offset = self.time_offset[chunk_task_ids]
                temporal_bias = (
                    -decay.permute(0, 2, 1).unsqueeze(-1)
                    * (
                        dt.unsqueeze(1) - offset.permute(0, 2, 1).unsqueeze(-1)
                    ).abs()
                )
                logits = logits + temporal_bias.to(logits.dtype)

            logits = logits.masked_fill(~key_mask, float("-inf"))
            safe_rows = any_content[:, None, None, None]
            safe_logits = torch.where(
                safe_rows.expand_as(logits), logits, torch.zeros_like(logits)
            )
            probabilities = torch.where(
                safe_rows.expand_as(logits),
                F.softmax(safe_logits, dim=-1),
                torch.zeros_like(logits),
            )

            attended = torch.matmul(probabilities, values)
            attended = attended.transpose(1, 2).reshape(B, stop - start, D)
            decoded = query_chunk + self.out_proj(attended)
            decoded = decoded + self.ffn(self.ffn_norm(decoded))
            decoded = decoded * any_content[:, None, None].to(decoded.dtype)
            decoded_chunks.append(decoded)

        return torch.cat(decoded_chunks, dim=1)


class MaskAwareTemporalPool(nn.Module):
    """Pool fine content over time, respecting validity masks."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, D)`` fine content.
            valid_mask: ``(B, T)`` boolean (True = valid).

        Returns:
            ``(B, D)`` pooled embedding.
        """
        if valid_mask is None:
            return self.norm(x.mean(dim=1))

        mask_f = valid_mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask_f).sum(dim=1) / count
        return self.norm(pooled)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class HEROModel(nn.Module):
    """Hierarchical EEG Representation model.

    Constructor signature follows the ``BaselineEEGModel`` pattern:
    ``task_configs`` and ``num_channels`` are required for training integration.

    Public API:
        ``encode()``:  pure representation, no task predictions.
        ``forward()``: training adapter returning ``ModelOutput``.
        ``tokenize()``: CPU-side data preparation.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}

    def __init__(
        self,
        task_configs: dict[str, TaskConfig] | None = None,
        num_channels: int = 1,
        embed_dim: int = 256,
        canonical_rate: int = 128,
        num_spatial_slots: int = 8,
        num_temporal_slots: int = 4,
        local_window: int = 32,
        num_local_attn_blocks: int = 2,
        num_attn_heads: int = 8,
        channel_encoder_layers: int = 3,
        channel_encoder_kernel_size: int = 7,
        task_query_chunk_size: int = 256,
        temporal_mode: TemporalMode = "hierarchical",
        flat_num_local_attn_blocks: int | None = None,
        temporal_reduction: TemporalReductionMode = "slots",
        top_down_fusion: TopDownFusionMode = "gated",
    ):
        super().__init__()

        from foundry.tasks.config import TaskConfig as TC

        if canonical_rate != 128:
            raise ValueError("HERO v1 has a fixed 128 Hz canonical rate.")
        if temporal_mode not in ("hierarchical", "flat"):
            raise ValueError(
                "temporal_mode must be either 'hierarchical' or 'flat', "
                f"got {temporal_mode!r}."
            )
        if temporal_reduction not in ("slots", "mean"):
            raise ValueError(
                "temporal_reduction must be either 'slots' or 'mean', "
                f"got {temporal_reduction!r}."
            )
        if top_down_fusion not in ("gated", "add"):
            raise ValueError(
                "top_down_fusion must be either 'gated' or 'add', "
                f"got {top_down_fusion!r}."
            )
        if flat_num_local_attn_blocks is not None and (
            flat_num_local_attn_blocks < 0
        ):
            raise ValueError("flat_num_local_attn_blocks must be non-negative.")

        self.embed_dim = embed_dim
        self.canonical_rate = canonical_rate
        self.num_channels = num_channels
        self.temporal_mode = temporal_mode
        self._task_configs = TC.normalize_task_configs(task_configs or {})

        self.channel_encoder = SharedLocalChannelEncoder(
            embed_dim=embed_dim,
            num_layers=channel_encoder_layers,
            kernel_size=channel_encoder_kernel_size,
        )
        self.spatial_mixer = SpatialSlotMixer(
            embed_dim=embed_dim,
            num_slots=num_spatial_slots,
            num_heads=num_attn_heads,
        )
        if temporal_mode == "hierarchical":
            self.encoder = HEROEncoder(
                embed_dim=embed_dim,
                num_attn_heads=num_attn_heads,
                num_local_attn_blocks=num_local_attn_blocks,
                local_window=local_window,
                num_temporal_slots=num_temporal_slots,
                temporal_reduction=temporal_reduction,
                top_down_fusion=top_down_fusion,
            )
        else:
            flat_depth = (
                num_local_attn_blocks
                if flat_num_local_attn_blocks is None
                else flat_num_local_attn_blocks
            )
            self.encoder = FlatTemporalEncoder(
                embed_dim=embed_dim,
                num_attn_heads=num_attn_heads,
                num_local_attn_blocks=flat_depth,
                local_window=local_window,
            )
        self.task_decoder = TaskQueryCrossAttention(
            embed_dim=embed_dim,
            num_tasks=len(self._task_configs),
            num_heads=num_attn_heads,
            query_chunk_size=task_query_chunk_size,
        )
        # Retained as part of the public module structure for downstream-code
        # compatibility. Forward now uses ``task_decoder``.
        self.temporal_pool = MaskAwareTemporalPool(embed_dim)
        self.router = build_readout_router(self._task_configs, embed_dim)

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    # ------------------------------------------------------------------
    # tokenize  (CPU / DataLoader workers)
    # ------------------------------------------------------------------

    def tokenize(self, data: Data) -> dict:
        """CPU-side data preparation.

        Resolves modality, filters channels, sanitizes non-finites,
        z-scores each valid channel, resamples to ``canonical_rate``,
        pads, and extracts multitask targets.

        Returns:
            Dict ready for collation with ``input_values``,
            ``input_timestamps``, ``output_timestamps``, ``channel_mask``,
            ``sample_mask``, ``task_index``, ``target_values``,
            ``target_weights``, and provenance fields.
        """
        signal, default_type, sampling_rate = self._resolve_signal_source(data)

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )
        sig = np.asarray(signal.signal[:, modality_mask], dtype=np.float32)

        C_real = sig.shape[1]

        non_finite = ~np.isfinite(sig)
        sample_valid = np.ones(sig.shape, dtype=bool)
        if non_finite.any():
            sample_valid[non_finite] = False
            sig = np.where(non_finite, 0.0, sig)

        for c in range(C_real):
            ch_valid = sample_valid[:, c]
            if ch_valid.sum() < 2:
                continue
            vals = sig[ch_valid, c]
            mu = vals.mean()
            std = vals.std()
            if std > 1e-8:
                sig[:, c] = (sig[:, c] - mu) / std
            sig[~ch_valid, c] = 0.0

        sig_ct = sig.T

        was_resampled = abs(sampling_rate - self.canonical_rate) > 0.5
        if was_resampled:
            resampled, resampled_mask = self._resample_signal(
                torch.from_numpy(sig_ct).unsqueeze(0),
                torch.from_numpy(sample_valid.all(axis=1)).unsqueeze(0),
                sampling_rate,
                self.canonical_rate,
            )
            sig_ct = resampled.squeeze(0).numpy()
            T_out = sig_ct.shape[1]
            sample_mask_1d = resampled_mask.squeeze(0).numpy()
        else:
            T_out = sig_ct.shape[1]
            sample_mask_1d = sample_valid.all(axis=1)[:T_out]

        domain_start = float(getattr(signal, "domain_start", 0.0))
        output_rate = self.canonical_rate if was_resampled else sampling_rate
        timestamps_1d = (
            domain_start
            + (np.arange(T_out, dtype=np.float32) + 0.5) / output_rate
        )

        channel_mask_1d = np.ones(C_real, dtype=bool)

        C_pad = self.num_channels
        if C_real < C_pad:
            pad_c = C_pad - C_real
            sig_ct = np.pad(sig_ct, ((0, pad_c), (0, 0)), constant_values=0.0)
            channel_mask_1d = np.concatenate(
                [channel_mask_1d, np.zeros(pad_c, dtype=bool)]
            )
        elif C_real > C_pad:
            sig_ct = sig_ct[:C_pad]
            channel_mask_1d = channel_mask_1d[:C_pad]
            C_real = C_pad

        output_timestamps, output_values, output_task_index, output_weights = (
            extract_multitask_targets(self._task_configs, data)
        )

        return {
            "input_values": torch.from_numpy(sig_ct),
            "input_timestamps": torch.from_numpy(timestamps_1d),
            "output_timestamps": pad8(output_timestamps),
            "channel_mask": torch.from_numpy(channel_mask_1d),
            "sample_mask": torch.from_numpy(sample_mask_1d),
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

    # ------------------------------------------------------------------
    # encode  (GPU -- public representation API)
    # ------------------------------------------------------------------

    def encode(
        self,
        *,
        signal: torch.Tensor,
        sampling_rate: float | None = None,
        timestamps: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> Representation:
        """Public representation entry point.

        Args:
            signal: ``(B, C, T)`` raw or pre-resampled EEG.
            sampling_rate: Source sampling rate if ``signal`` is not at
                ``canonical_rate``. Mutually exclusive with ``timestamps``.
            timestamps: ``(B, T)`` explicit physical timestamps. Mutually
                exclusive with ``sampling_rate``.
            channel_mask: ``(B, C)`` boolean (True = valid channel).
            sample_mask: ``(B, T)`` boolean (True = valid sample).

        Returns:
            :class:`Representation` with fused fine content, timestamps,
            and coverage metadata.
        """
        B, C, T = signal.shape
        device = signal.device

        if (timestamps is None) == (sampling_rate is None):
            raise ValueError(
                "Provide exactly one of sampling_rate or timestamps."
            )

        if timestamps is None:
            rate = float(sampling_rate)
            domain_start = torch.zeros(B, 1, device=device, dtype=signal.dtype)
            explicit_timestamps = None
        else:
            timestamps = timestamps.to(device=device, dtype=signal.dtype)
            if timestamps.shape != (B, T):
                raise ValueError("timestamps must have shape [batch, samples].")
            steps = timestamps[:, 1:] - timestamps[:, :-1]
            if not torch.all(steps > 0):
                raise ValueError("timestamps must be strictly increasing.")
            rate = float((1 / steps.median()).item())
            tolerance = max(1e-6, 1e-3 / rate)
            if not torch.allclose(
                steps, torch.full_like(steps, 1 / rate), atol=tolerance
            ):
                raise ValueError(
                    "Irregular timestamps are not supported by HERO v1."
                )
            domain_start = timestamps[:, :1] - 0.5 / rate
            explicit_timestamps = timestamps

        if channel_mask is None:
            channel_mask = torch.ones(B, C, dtype=torch.bool, device=device)
        if sample_mask is None:
            sample_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        was_resampled = not math.isclose(
            rate, self.canonical_rate, rel_tol=0, abs_tol=0.5
        )
        if was_resampled:
            signal, sample_mask = self._resample_signal(
                signal, sample_mask, rate, self.canonical_rate
            )
            T = signal.shape[-1]
            ts = (
                domain_start
                + (
                    torch.arange(
                        T, device=device, dtype=signal.dtype
                    ).unsqueeze(0)
                    + 0.5
                )
                / self.canonical_rate
            )
        elif explicit_timestamps is not None:
            ts = explicit_timestamps
        else:
            ts = (
                domain_start
                + (
                    torch.arange(
                        T, device=device, dtype=signal.dtype
                    ).unsqueeze(0)
                    + 0.5
                )
                / rate
            )

        input_rf_intervals = _timestamp_intervals(ts, fallback_step=1.0 / rate)

        x = self._channel_encode(signal, channel_mask, sample_mask)
        fused, token_valid = self.spatial_mixer(x, channel_mask)

        combined_valid = token_valid & sample_mask
        fine_rf_intervals = input_rf_intervals
        channel_layers = len(self.channel_encoder.layers) // 3
        for _ in range(channel_layers):
            fine_rf_intervals = _expand_receptive_fields(
                fine_rf_intervals,
                left=self.channel_encoder.pad,
                right=0,
                valid_mask=combined_valid,
            )

        fused_fine, meta = self.encoder(
            fused,
            ts,
            fine_rf_intervals,
            combined_valid,
        )

        B_out, T_fine = fused_fine.shape[:2]

        coverage = CoverageInfo(
            fine_valid=combined_valid,
            mid_valid=meta["mid_valid"],
            coarse_valid=meta["coarse_valid"],
            sample_support=sample_mask.float(),
            channel_count=(
                channel_mask.sum(dim=1).unsqueeze(1).expand(B_out, T_fine)
            ),
            channel_fraction=(
                channel_mask.float()
                .mean(dim=1)
                .unsqueeze(1)
                .expand(B_out, T_fine)
            ),
            fine_rf_intervals=meta["fine_rf_intervals"],
            mid_rf_intervals=meta["mid_rf_intervals"],
            coarse_rf_intervals=meta["coarse_rf_intervals"],
        )

        return Representation(
            content=fused_fine,
            content_timestamps=ts,
            coverage=coverage,
        )

    @staticmethod
    def _resample_signal(
        signal: torch.Tensor,
        sample_mask: torch.Tensor,
        source_rate: float,
        target_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Band-limit resample zero-filled values and conservatively map masks."""
        import torchaudio.functional as audio

        source = int(round(source_rate))
        target = int(round(target_rate))
        if not math.isclose(source_rate, source, abs_tol=1e-6):
            raise ValueError(
                "HERO v1 resampling requires an integral sampling rate."
            )

        B, C, T = signal.shape
        masked_signal = signal * sample_mask[:, None].to(signal.dtype)
        values = audio.resample(masked_signal.reshape(B * C, T), source, target)
        values = values.reshape(B, C, -1)

        invalid_source = (~sample_mask).float().unsqueeze(1)
        invalid = (
            F.interpolate(
                invalid_source,
                size=values.shape[-1],
                mode="area",
            ).squeeze(1)
            > 0
        )
        # A zero-filled input hole influences nearby filtered values.  Expand
        # it by the fixed 33-tap support before declaring an output valid.
        invalid = (
            F.max_pool1d(
                invalid.float().unsqueeze(1),
                kernel_size=33,
                stride=1,
                padding=16,
            )
            .squeeze(1)
            .bool()
        )
        valid = ~invalid
        return values, valid

    # ------------------------------------------------------------------
    # forward  (GPU -- training adapter)
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        task_index: torch.Tensor,
        output_timestamps: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        unpack_output: bool = True,
    ) -> ModelOutput:
        """Training adapter: encode + task-query decode + route to heads.

        Args:
            input_values: ``(B, C, T)`` signal from ``tokenize()``.
            input_timestamps: ``(B, T)`` physical timestamps.
            task_index: ``(B, N_out)`` padded task indices (0 = padding).
            output_timestamps: Optional ``(B, N_out)`` target timestamps. When
                omitted, task queries attend globally without temporal bias.
            channel_mask: ``(B, C)`` boolean.
            sample_mask: ``(B, T)`` boolean.
            unpack_output: Ignored (kept for interface compat).

        Returns:
            :class:`ModelOutput` with ``task_outputs``.
        """
        del unpack_output

        rep = self.encode(
            signal=input_values,
            timestamps=input_timestamps,
            channel_mask=channel_mask,
            sample_mask=sample_mask,
        )

        output_embs = self.task_decoder(
            rep.content,
            task_index,
            content_timestamps=(
                rep.content_timestamps
                if output_timestamps is not None
                else None
            ),
            output_timestamps=output_timestamps,
            content_valid=rep.coverage.fine_valid,
        )

        B, N_out = task_index.shape
        flat_embs = output_embs.reshape(B * N_out, -1)
        flat_task = task_index.reshape(B * N_out)
        valid = flat_task > 0

        task_outputs = self.router(
            flat_embs[valid], (flat_task[valid] - 1).long()
        )

        return ModelOutput(task_outputs=task_outputs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _channel_encode(
        self,
        signal: torch.Tensor,
        channel_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run shared local channel encoder.

        Args:
            signal: ``(B, C, T)``
            channel_mask: ``(B, C)``
            sample_mask: ``(B, T)``

        Returns:
            ``(B, C, T, D)`` channel-local features.
        """
        B, C, T = signal.shape
        x = signal.reshape(B * C, 1, T)

        sm_expanded = sample_mask.unsqueeze(1).expand(B, C, T).reshape(B * C, T)
        cm_expanded = (
            channel_mask.unsqueeze(2).expand(B, C, T).reshape(B * C, T)
        )
        combined_mask = sm_expanded & cm_expanded

        x = self.channel_encoder(x, combined_mask)
        x = x.view(B, C, self.embed_dim, T).permute(0, 1, 3, 2)
        return x

    def _resolve_signal_source(self, data: Data):
        """Find signal source, default modality type, and sampling rate."""
        for modality in ["eeg", "ecog", "seeg", "ieeg"]:
            sig = getattr(data, modality, None)
            if sig is not None:
                if (
                    hasattr(sig, "sampling_rate")
                    and sig.sampling_rate is not None
                ):
                    sr = float(sig.sampling_rate)
                else:
                    ts = sig.timestamps
                    diffs = np.diff(ts).astype(np.float64)
                    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if valid.size == 0:
                        raise ValueError("Cannot infer sampling rate.")
                    sr = 1.0 / float(np.median(valid))
                return sig, modality.upper(), sr
        raise ValueError(
            "Data must have an 'eeg', 'ecog', 'seeg', or 'ieeg' field"
        )
