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
from foundry.models.embeddings.temporal import CWTEmbedding


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
        temporal_embedding: nn.Module,
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

    def pretokenize(
        self,
        signal: np.ndarray,
        channel_tokens: np.ndarray,
        sampling_rate: float,
        sequence_length: float,
    ) -> dict:
        """CPU-side per-sample preparation.

        Called from ``POYOEEGModel.tokenize()`` during data loading.

        Args:
            signal: (T, C_actual) raw signal.
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
            patch_samples = max(1, round(self.patch_duration * sampling_rate))
            stride_samples = max(1, round(self.stride * sampling_rate))
            if num_samples > patch_samples:
                num_patches = (
                    num_samples - patch_samples
                ) // stride_samples + 1
            else:
                num_patches = 1

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
            # CWT produces target_time_tokens; PerTimepoint produces T tokens
            if hasattr(self.temporal_embedding, "target_time_tokens"):
                num_time_tokens = self.temporal_embedding.target_time_tokens
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
            if isinstance(self.temporal_embedding, CWTEmbedding):
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
        tokens = tokens * token_mask.unsqueeze(-1).float()

        return tokens


__all__ = ["EEGTokenizer"]
