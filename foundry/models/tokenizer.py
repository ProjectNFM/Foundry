from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn

from foundry.models.embeddings.channel import (
    ChannelStrategy,
    PerChannelStrategy,
)
from foundry.models.embeddings.patching import (
    compute_patch_timestamps,
    patch_signal,
)
from foundry.models.embeddings.temporal.base import TemporalEmbedding
from foundry.models.signal_preparation import (
    PreparedSignal,
    compute_num_patches,
    normalize_signal_length,
)


class EEGTokenizer(nn.Module):
    """Composable EEG tokenizer combining channel strategy and temporal embedding.

    Orchestrates three independent concerns:
      1. **Channel strategy** -- spatial transform of the channel dimension.
      2. **Optional GPU patching** -- when ``patch_duration`` is set, the
         signal is patched via ``torch.unfold`` on GPU.
      3. **Temporal embedding** -- converts (patched or raw) signal into
         token embeddings.

    An optional fourth step reassembles per-channel tokens and adds
    channel identity embeddings when using :class:`PerChannelStrategy`.

    Args:
        channel_strategy: Module handling channel dimension transform.
        temporal_embedding: Module converting (patched or raw) signal to
            token embeddings.
        embed_dim: Output embedding dimension.
        patch_duration: If set, patches the signal with this duration in
            seconds on GPU.  If ``None``, no patching is applied.
        stride: Stride between patches in seconds.  Defaults to
            ``patch_duration`` (non-overlapping patches).
        channel_fusion: How channel identity embeddings are combined with
            token embeddings in per-channel mode.  ``"add"`` adds them
            (both must be ``embed_dim``).  ``"concat"`` concatenates them
            along the feature axis -- the temporal embedding produces
            tokens of size ``embed_dim - channel_emb_dim`` and the
            channel embedding fills the remaining dimensions.
        channel_emb_dim: Dimension of channel identity embeddings.
            Required when ``channel_fusion="concat"``.  Ignored (defaults
            to ``embed_dim``) when ``channel_fusion="add"``.
    """

    def __init__(
        self,
        channel_strategy: ChannelStrategy,
        temporal_embedding: TemporalEmbedding,
        embed_dim: int,
        patch_duration: float | None = None,
        stride: float | None = None,
        channel_fusion: Literal["add", "concat"] = "add",
        channel_emb_dim: int | None = None,
    ):
        super().__init__()
        self.channel_strategy = channel_strategy
        self.temporal_embedding = temporal_embedding
        self.embed_dim = embed_dim
        self.patch_duration = patch_duration
        self.stride = stride if stride is not None else patch_duration
        self._do_patching = patch_duration is not None
        self.channel_fusion = channel_fusion

        if channel_fusion == "concat":
            if channel_emb_dim is None:
                raise ValueError(
                    "channel_emb_dim is required when channel_fusion='concat'"
                )
            if channel_emb_dim >= embed_dim:
                raise ValueError(
                    f"channel_emb_dim ({channel_emb_dim}) must be less than "
                    f"embed_dim ({embed_dim})"
                )
            self._channel_emb_dim = channel_emb_dim
        else:
            self._channel_emb_dim = embed_dim

        self.post_proj_norm = nn.LayerNorm(self.token_embed_dim)

    @property
    def does_patching(self) -> bool:
        """Whether this tokenizer applies GPU-side signal patching."""
        return self._do_patching

    @property
    def has_fixed_token_count(self) -> bool:
        """Whether the temporal embedding produces a fixed token count."""
        return self.temporal_embedding.has_fixed_token_count

    @property
    def channel_emb_dim(self) -> int:
        """Dimension expected for channel identity embeddings."""
        return self._channel_emb_dim

    @property
    def token_embed_dim(self) -> int:
        """Dimension the temporal embedding should produce.

        Equals ``embed_dim`` for add fusion, or
        ``embed_dim - channel_emb_dim`` for concat fusion.
        """
        if self.channel_fusion == "concat":
            return self.embed_dim - self._channel_emb_dim
        return self.embed_dim

    @property
    def uses_per_channel(self) -> bool:
        return isinstance(self.channel_strategy, PerChannelStrategy)

    def get_patch_samples(self, sampling_rate: float) -> int:
        """Number of time samples per patch for the given sampling rate."""
        if not self._do_patching:
            raise ValueError(
                "get_patch_samples() requires patch_duration to be set"
            )
        return max(1, round(self.patch_duration * sampling_rate))

    def get_stride_samples(self, sampling_rate: float) -> int:
        """Number of time samples between patch starts for the given sampling rate."""
        if not self._do_patching:
            raise ValueError(
                "get_stride_samples() requires patch_duration to be set"
            )
        return max(1, round(self.stride * sampling_rate))

    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        """Canonical token count for a given window duration and rate.

        Delegates to either the patch-count formula or the temporal embedding.
        """
        if self._do_patching:
            num_samples = round(sampling_rate * sequence_length)
            patch_samples = self.get_patch_samples(sampling_rate)
            stride_samples = self.get_stride_samples(sampling_rate)
            return compute_num_patches(
                num_samples, patch_samples, stride_samples
            )
        return self.temporal_embedding.get_num_time_tokens(
            sequence_length, sampling_rate
        )

    def prepare_signal(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        sequence_length: float,
        modality_mask: np.ndarray,
    ) -> PreparedSignal:
        """Normalize signal length and produce the token-grid contract.

        This is the single entry point that ensures encoder inputs and
        reconstruction targets share the same prepared signal.

        Args:
            signal: (T, C_filtered) modality-filtered, sanitized signal.
            sampling_rate: Sampling rate in Hz.
            sequence_length: Target duration in seconds.
            modality_mask: Boolean mask over original channels.

        Returns:
            Immutable :class:`PreparedSignal` with normalized signal
            and token-grid metadata.
        """
        original_T = signal.shape[0]
        normalized = normalize_signal_length(
            signal, sampling_rate, sequence_length
        )
        return PreparedSignal(
            signal=normalized,
            sampling_rate=sampling_rate,
            num_samples=normalized.shape[0],
            original_num_samples=original_T,
            num_channels=normalized.shape[1],
            modality_mask=modality_mask,
        )

    def pretokenize(
        self,
        signal: np.ndarray,
        channel_tokens: np.ndarray,
        sampling_rate: float,
        sequence_length: float,
    ) -> dict:
        """CPU-side per-sample preparation.

        Called from ``POYOEEGModel.tokenize()`` during data loading.
        Expects a signal that has already been length-normalized via
        :func:`~foundry.models.signal_preparation.normalize_signal_length`.

        Args:
            signal: (T, C_actual) length-normalized signal.
            channel_tokens: (C_actual,) channel-token indices.
            sampling_rate: Sampling rate in Hz.
            sequence_length: Duration of the context window in seconds.

        Returns:
            dict with model input tensors including ``input_timestamps``.
        """
        result = self.channel_strategy.prepare_pretokenize(
            signal,
            channel_tokens,
            sampling_rate,
        )

        if self._do_patching:
            num_samples = signal.shape[0]
            patch_samples = self.get_patch_samples(sampling_rate)
            stride_samples = self.get_stride_samples(sampling_rate)
            num_patches = compute_num_patches(
                num_samples, patch_samples, stride_samples
            )

            patch_timestamps = compute_patch_timestamps(
                start_time=0.0,
                num_patches=num_patches,
                patch_duration=self.patch_duration,
                stride=self.stride,
            )

            if self.uses_per_channel:
                C = result["input_mask"].shape[0]
                result["input_timestamps"] = patch_timestamps.repeat(C)
            else:
                result["input_timestamps"] = patch_timestamps
        else:
            if self.temporal_embedding.has_fixed_token_count:
                num_time_tokens = self.temporal_embedding.get_num_time_tokens(
                    sequence_length, sampling_rate
                )
            else:
                num_time_tokens = signal.shape[0]

            sample_timestamps = torch.linspace(
                0,
                sequence_length,
                num_time_tokens,
                dtype=torch.float32,
            )

            if self.uses_per_channel:
                C = result["input_mask"].shape[0]
                result["input_timestamps"] = sample_timestamps.repeat(C)
            else:
                result["input_timestamps"] = sample_timestamps

        return result

    def compute_reconstruction_targets(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        sequence_length: float,
    ) -> torch.Tensor:
        """Compute per-token z-scored reconstruction targets.

        Uses :meth:`get_num_time_tokens` and the channel strategy to produce
        targets that match the tokenizer's token grid exactly.

        The signal must already be length-normalized (via
        :func:`~foundry.models.signal_preparation.normalize_signal_length`)
        so that its grid aligns with the encoder input.

        Args:
            signal: ``(T, C_actual)`` raw signal, already modality-filtered
                and length-normalized.
            sampling_rate: Sampling rate in Hz.
            sequence_length: Duration of the context window in seconds.

        Returns:
            For patching mode: ``(C_pad * num_patches, patch_samples)``
            Otherwise: ``(C_pad, N)`` where N is the number of time tokens.
        """
        if self._do_patching:
            return self._compute_patch_targets(signal, sampling_rate)

        C_pad = self.channel_strategy.max_channels
        C_actual = min(signal.shape[1], C_pad)
        N = self.temporal_embedding.get_num_time_tokens(
            sequence_length, sampling_rate
        )

        T = signal.shape[0]
        if self.temporal_embedding.has_fixed_token_count:
            token_times = np.linspace(0, sequence_length, N)
            raw_times = np.linspace(0, sequence_length, T)
            from scipy.interpolate import interp1d

            f = interp1d(
                raw_times,
                signal[:, :C_actual],
                axis=0,
                kind="linear",
                assume_sorted=True,
                copy=False,
            )
            resampled = f(token_times).T.astype(np.float32)
        else:
            N = T
            resampled = signal[:, :C_actual].T.astype(np.float32)

        targets = np.zeros((C_pad, N), dtype=np.float32)
        mu = resampled.mean(axis=1, keepdims=True)
        std = resampled.std(axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        targets[:C_actual] = (resampled - mu) / std

        return torch.from_numpy(targets)

    def _compute_patch_targets(
        self, signal: np.ndarray, sampling_rate: float
    ) -> torch.Tensor:
        """Compute reconstruction targets for patching mode.

        Each token's target is the z-scored raw signal within its patch window.
        """
        patch_samples = self.get_patch_samples(sampling_rate)
        stride_samples = self.get_stride_samples(sampling_rate)
        T, C_actual = signal.shape
        C_pad = self.channel_strategy.max_channels

        num_patches = compute_num_patches(T, patch_samples, stride_samples)

        C_used = min(C_actual, C_pad)
        starts = np.arange(num_patches) * stride_samples
        idx = starts[:, None] + np.arange(patch_samples)[None, :]
        patches = signal[idx, :C_used].transpose(2, 0, 1).astype(np.float32)

        mu = patches.mean(axis=2, keepdims=True)
        std = patches.std(axis=2, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        normalized = (patches - mu) / std

        targets = np.zeros(
            (C_pad, num_patches, patch_samples), dtype=np.float32
        )
        targets[:C_used] = normalized

        return torch.from_numpy(targets.reshape(C_pad * num_patches, -1))

    def forward(
        self,
        input_values: torch.Tensor,
        *,
        input_channel_index: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        input_sampling_rate: torch.Tensor | None = None,
        channel_emb_fn: Callable | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """GPU-side embedding.

        Called from ``POYOEEGModel.forward()``.

        Args:
            input_values: (B, C, T) raw signal (padded to max T in batch).
            input_channel_index: (B, C) channel identity tokens.
            input_mask: (B, C) which channels are real.
            input_sampling_rate: (B,) per-item sampling rate.
            channel_emb_fn: Maps channel token indices to embedding vectors.
                Used by :class:`PerChannelStrategy` for channel identity.

        Returns:
            (B, num_tokens, embed_dim)
        """
        B, C_in, T = input_values.shape

        transformed = self.channel_strategy(
            input_values,
            input_mask=input_mask,
            **kwargs,
        )

        if self.uses_per_channel:
            sr = input_sampling_rate.repeat_interleave(C_in)
            seq_len = kwargs.get("input_seq_len")
            if seq_len is not None:
                seq_len = seq_len.repeat_interleave(C_in)
        else:
            sr = input_sampling_rate
            seq_len = kwargs.get("input_seq_len")

        if self._do_patching:
            sampling_rate = sr[0].item()
            patches = patch_signal(
                transformed,
                self.patch_duration,
                self.stride,
                sampling_rate,
            )
            tokens = self.temporal_embedding(patches)
        else:
            if self.temporal_embedding.has_fixed_token_count:
                tokens = self.temporal_embedding(
                    transformed,
                    input_sampling_rate=sr,
                    input_seq_len=seq_len,
                )
            else:
                tokens = self.temporal_embedding(transformed.transpose(1, 2))

        tokens = self.post_proj_norm(tokens)

        if self.uses_per_channel:
            tokens = self._reassemble_per_channel(
                tokens,
                B,
                input_mask,
                input_channel_index,
                channel_emb_fn,
            )

        return tokens

    def _reassemble_per_channel(
        self,
        tokens,
        B,
        channel_mask,
        channel_index,
        channel_emb_fn,
    ):
        """Reshape per-channel tokens and fuse channel identity embedding.

        Args:
            tokens: (B * C_pad, N, D_tok) where N = P (patched) or
                T (per-timepoint) and D_tok = ``token_embed_dim``.
            B: Original batch size.
            channel_mask: (B, C_pad) which channels are real.
            channel_index: (B, C_pad) channel token indices.
            channel_emb_fn: Maps channel indices to embedding vectors.

        Returns:
            (B, C_pad * N, embed_dim) with channel identity embeddings
            fused (added or concatenated) and padded channels zeroed out.
        """
        C = channel_mask.shape[1]
        N = tokens.shape[1]

        tokens = tokens.reshape(B, C, N, -1)

        if channel_emb_fn is not None:
            ch_emb = channel_emb_fn(channel_index)
            if self.channel_fusion == "concat":
                ch_emb = ch_emb.unsqueeze(2).expand(-1, -1, N, -1)
                tokens = torch.cat([tokens, ch_emb], dim=-1)
            else:
                tokens = tokens + ch_emb.unsqueeze(2)

        tokens = tokens.reshape(B, C * N, -1)

        token_mask = channel_mask.unsqueeze(2).expand(B, C, N).reshape(B, C * N)
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)

        return tokens


__all__ = ["EEGTokenizer"]
