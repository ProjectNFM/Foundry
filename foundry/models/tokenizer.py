from __future__ import annotations

from typing import Callable

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
from foundry.models.masking import MaskingStrategy


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

    When ``masking`` is provided the tokenizer also supports masked
    reconstruction pretraining: ``pretokenize`` extracts reconstruction
    targets at masked positions and ``forward`` replaces masked token
    embeddings with a learned ``mask_emb``.

    Args:
        channel_strategy: Module handling channel dimension transform.
        temporal_embedding: Module converting (patched or raw) signal to
            token embeddings.
        embed_dim: Output embedding dimension.
        patch_duration: If set, patches the signal with this duration in
            seconds on GPU.  If ``None``, no patching is applied.
        stride: Stride between patches in seconds.  Defaults to
            ``patch_duration`` (non-overlapping patches).
        masking: Optional masking strategy for pretraining.
    """

    def __init__(
        self,
        channel_strategy: ChannelStrategy,
        temporal_embedding: nn.Module,
        embed_dim: int,
        patch_duration: float | None = None,
        stride: float | None = None,
        masking: MaskingStrategy | None = None,
    ):
        super().__init__()
        self.channel_strategy = channel_strategy
        self.temporal_embedding = temporal_embedding
        self.embed_dim = embed_dim
        self.patch_duration = patch_duration
        self.stride = stride if stride is not None else patch_duration
        self._do_patching = patch_duration is not None
        self.post_proj_norm = nn.LayerNorm(embed_dim)
        self.masking = masking

        if masking is not None:
            self.mask_emb = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.mask_emb, std=0.02)

    @property
    def uses_per_channel(self) -> bool:
        return isinstance(self.channel_strategy, PerChannelStrategy)

    @property
    def supports_variable_sampling_rate(self) -> bool:
        """Whether this tokenizer supports mixed sampling rates in one dataset."""
        return not self._do_patching

    @property
    def recon_output_dim(self) -> int:
        """Expected reconstruction target width for masked reconstruction."""
        if self._do_patching:
            patch_samples = getattr(
                self.temporal_embedding, "patch_samples", None
            )
            if patch_samples is None:
                raise AttributeError(
                    "Patch temporal embedding must define 'patch_samples'."
                )
            if self.uses_per_channel:
                return int(patch_samples)
            num_channels = getattr(self.channel_strategy, "num_channels", None)
            if num_channels is None:
                raise AttributeError(
                    "Patch tokenizer with non-per-channel strategy must define "
                    "'num_channels' on channel_strategy."
                )
            return int(num_channels) * int(patch_samples)

        if self.uses_per_channel:
            return 1

        num_channels = getattr(self.channel_strategy, "num_channels", None)
        if num_channels is None:
            raise AttributeError(
                "Non-per-channel strategy must define 'num_channels' to infer "
                "reconstruction output dimension."
            )
        return int(num_channels)

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
            When masking is active also includes ``masking_mask``,
            ``reconstruction_targets``, and ``masked_timestamps``.
        """
        result = self.channel_strategy.prepare_pretokenize(
            signal,
            channel_tokens,
            sampling_rate,
        )

        padded_signal = result["input_values"]  # (C_pad, T)

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

        if self.masking is not None:
            self._add_masking_fields(
                result,
                padded_signal,
                signal,
                sampling_rate,
                sequence_length,
            )

        return result

    def _add_masking_fields(
        self,
        result: dict,
        padded_signal: torch.Tensor | np.ndarray,
        raw_signal: np.ndarray,
        sampling_rate: float,
        sequence_length: float,
    ) -> None:
        """Generate a mask and extract reconstruction targets for pretraining.

        Adds three keys to *result* **in-place**:

        * ``masking_mask`` -- ``(num_tokens,)`` bool, ``True`` = masked.
        * ``reconstruction_targets`` -- ``(num_tokens, output_dim)`` float
          tensor.  Entries at unmasked positions are zero; masked entries
          contain the raw-signal patch/sample that the model should
          reconstruct.
        * ``masked_timestamps`` -- ``(n_masked,)`` timestamps of masked
          tokens (used as output query timestamps for reconstruction).

        The target extraction logic branches on the channel strategy and
        temporal embedding to handle every valid combination (see the
        *Masking Details* section in the plan).

        Args:
            result: Pretokenize output dict to update in-place.
            padded_signal: ``(C_pad, T)`` channel-padded signal from the
                channel strategy's ``prepare_pretokenize``.
            raw_signal: ``(T, C_actual)`` original unpadded signal.
            sampling_rate: Sampling rate in Hz.
            sequence_length: Duration of the context window in seconds.
        """
        if isinstance(padded_signal, np.ndarray):
            padded_signal = torch.from_numpy(padded_signal).float()
        elif not isinstance(padded_signal, torch.Tensor):
            padded_signal = torch.tensor(padded_signal, dtype=torch.float32)
        else:
            padded_signal = padded_signal.float()

        C_pad = padded_signal.shape[0]

        if self._do_patching:
            num_samples = raw_signal.shape[0]
            patch_samples = max(1, round(self.patch_duration * sampling_rate))
            stride_samples = max(1, round(self.stride * sampling_rate))
            if num_samples > patch_samples:
                num_patches = (
                    num_samples - patch_samples
                ) // stride_samples + 1
            else:
                num_patches = 1

            # CPU-side patching: unfold (C_pad, T) -> (C_pad, num_patches, patch_samples)
            sig_t = padded_signal  # (C_pad, T)
            T = sig_t.shape[1]
            if T >= patch_samples:
                patches_cpu = sig_t.unfold(1, patch_samples, stride_samples)
            else:
                patches_cpu = sig_t.unsqueeze(1)

            if self.uses_per_channel:
                # Flat token order: all patches of ch0, then ch1, ...
                # patches_cpu: (C_pad, num_patches, patch_samples)
                # Each token reconstructs (1 * patch_samples)
                flat_patches = patches_cpu.reshape(
                    C_pad * num_patches, patch_samples
                )
                num_tokens = C_pad * num_patches
                mask = self.masking.generate_mask(num_tokens)
                targets = flat_patches[mask]
            else:
                # Fixed / SpatialProjection: token = patch across all channels
                # patches_cpu: (C_pad, num_patches, patch_samples)
                # -> (num_patches, C_pad, patch_samples)
                patches_transposed = patches_cpu.permute(1, 0, 2)
                num_tokens = num_patches
                mask = self.masking.generate_mask(num_tokens)
                targets = patches_transposed[mask].reshape(
                    -1, C_pad * patch_samples
                )
        else:
            if hasattr(self.temporal_embedding, "target_time_tokens"):
                num_time_tokens = self.temporal_embedding.target_time_tokens
                # Resample signal to target_time_tokens
                # padded_signal: (C_pad, T)
                sig_resampled = torch.nn.functional.interpolate(
                    padded_signal.unsqueeze(0),
                    size=num_time_tokens,
                    mode="linear",
                    align_corners=True,
                ).squeeze(0)  # (C_pad, num_time_tokens)

                if self.uses_per_channel:
                    # (C_pad * num_time_tokens,) tokens, each target is (1,)
                    flat = sig_resampled.reshape(C_pad * num_time_tokens)
                    num_tokens = C_pad * num_time_tokens
                    mask = self.masking.generate_mask(num_tokens)
                    targets = flat[mask].unsqueeze(-1)
                else:
                    # (num_time_tokens,) tokens, each target is (C_pad,)
                    num_tokens = num_time_tokens
                    mask = self.masking.generate_mask(num_tokens)
                    targets = sig_resampled.T[mask]  # (n_masked, C_pad)
            else:
                T = raw_signal.shape[0]
                if self.uses_per_channel:
                    # (C_pad * T,) tokens, each target is (1,)
                    flat = padded_signal.reshape(C_pad * T)
                    num_tokens = C_pad * T
                    mask = self.masking.generate_mask(num_tokens)
                    targets = flat[mask].unsqueeze(-1)
                else:
                    # (T,) tokens, each target is (C_pad,)
                    num_tokens = T
                    mask = self.masking.generate_mask(num_tokens)
                    targets = padded_signal.T[mask]  # (n_masked, C_pad)

        timestamps = result["input_timestamps"]
        if isinstance(timestamps, np.ndarray):
            timestamps = torch.from_numpy(timestamps).float()

        num_tokens = mask.shape[0]
        output_dim = targets.shape[1] if targets.ndim == 2 else 1
        full_targets = torch.zeros(num_tokens, output_dim)
        full_targets[mask] = targets.reshape(-1, output_dim)

        result["masking_mask"] = mask
        result["reconstruction_targets"] = full_targets
        result["masked_timestamps"] = timestamps[mask]

    def forward(
        self,
        input_values: torch.Tensor,
        *,
        input_channel_index: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        input_sampling_rate: torch.Tensor | None = None,
        channel_emb_fn: Callable | None = None,
        masking_mask: torch.Tensor | None = None,
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
            masking_mask: (B, num_tokens) boolean mask from pretokenize.
                ``True`` positions are replaced with ``mask_emb``.

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
            if sr is None:
                raise ValueError(
                    "input_sampling_rate is required when patching is enabled."
                )
            if sr.numel() == 0:
                raise ValueError(
                    "input_sampling_rate must be non-empty when patching is enabled."
                )
            if not torch.allclose(sr, sr[0], rtol=1e-5, atol=1e-5):
                raise ValueError(
                    "Patching tokenizer requires a uniform sampling rate "
                    f"within each batch, got rates: {torch.unique(sr).tolist()}"
                )
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

        if masking_mask is not None and self.masking is not None:
            tokens = self._apply_masking(tokens, masking_mask, B, C_in)

        if self.uses_per_channel:
            tokens = self._reassemble_per_channel(
                tokens,
                B,
                input_mask,
                input_channel_index,
                channel_emb_fn,
            )

        return tokens

    def _apply_masking(
        self,
        tokens: torch.Tensor,
        masking_mask: torch.Tensor,
        B: int,
        C_in: int,
    ) -> torch.Tensor:
        """Replace masked token embeddings with the learned ``mask_emb``.

        Must be called **after** ``post_proj_norm`` and **before**
        ``_reassemble_per_channel`` (if applicable).

        For :class:`PerChannelStrategy` the tokens are still in
        ``(B*C, N, D)`` form and the mask is ``(B, C*N)`` flat, so we
        reshape the mask to ``(B*C, N)`` before applying it.

        Args:
            tokens: Embedded tokens.  Shape is ``(B, N, D)`` for
                fixed / spatial-projection strategies, or
                ``(B*C, N, D)`` for per-channel strategies.
            masking_mask: ``(B, num_tokens)`` boolean mask where
                ``True`` means the position should be replaced.
            B: Original batch size (before per-channel expansion).
            C_in: Number of (padded) channels in the batch.

        Returns:
            Token tensor of the same shape with masked positions
            overwritten by ``mask_emb``.
        """
        if self.uses_per_channel:
            # tokens: (B*C, N, D), mask: (B, C*N)
            N = tokens.shape[1]
            C = C_in
            # Reshape mask to (B*C, N)
            mask_reshaped = masking_mask.reshape(B, C, N).reshape(B * C, N)
            tokens = tokens.clone()
            tokens[mask_reshaped] = self.mask_emb
        else:
            # tokens: (B, N, D), mask: (B, N)
            tokens = tokens.clone()
            tokens[masking_mask] = self.mask_emb
        return tokens

    def _reassemble_per_channel(
        self,
        tokens,
        B,
        channel_mask,
        channel_index,
        channel_emb_fn,
    ):
        """Reshape per-channel tokens and add channel identity embedding.

        Args:
            tokens: (B * C_pad, N, D) where N = P (patched) or T (per-timepoint).
            B: Original batch size.
            channel_mask: (B, C_pad) which channels are real.
            channel_index: (B, C_pad) channel token indices.
            channel_emb_fn: Maps channel indices to embedding vectors.

        Returns:
            (B, C_pad * N, D) with channel identity embeddings added and
            padded channels zeroed out.
        """
        C = channel_mask.shape[1]
        N = tokens.shape[1]
        D = tokens.shape[2]

        tokens = tokens.reshape(B, C, N, D)

        if channel_emb_fn is not None:
            ch_emb = channel_emb_fn(channel_index)
            tokens = tokens + ch_emb.unsqueeze(2)

        tokens = tokens.reshape(B, C * N, D)

        token_mask = channel_mask.unsqueeze(2).expand(B, C, N).reshape(B, C * N)
        tokens = tokens * token_mask.unsqueeze(-1).float()

        return tokens


__all__ = ["EEGTokenizer"]
