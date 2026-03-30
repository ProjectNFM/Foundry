from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class ChannelStrategy(nn.Module, ABC):
    """Transforms raw signal according to a channel-handling policy.

    Channel strategies handle how the channel dimension is transformed
    *before* any patching or temporal embedding.  They are agnostic to
    the downstream patching decision (which lives in ``EEGTokenizer``).
    """

    @abstractmethod
    def prepare_pretokenize(
        self,
        signal: np.ndarray,
        channel_tokens: np.ndarray,
        sampling_rate: float,
    ) -> dict:
        """CPU-side per-sample preparation (padding, metadata).

        Called from ``EEGTokenizer.pretokenize()`` during data loading.

        Args:
            signal: (T, C_actual) raw signal with only valid channels.
            channel_tokens: (C_actual,) integer channel-token indices.
            sampling_rate: Sampling rate in Hz.

        Returns:
            dict with at minimum ``input_values``, ``input_channel_index``,
            and ``input_mask`` tensors.
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """GPU-side channel transform.

        Args:
            input_values: (B, C, T) batched signal.

        Returns:
            Transformed signal tensor.
        """
        ...


class FixedChannelStrategy(ChannelStrategy):
    """Pad or truncate to a fixed number of channels.

    Output shape from ``forward``: ``(B, num_channels, T)`` (unchanged).
    Downstream temporal embedding sees ``C = num_channels``.

    Args:
        num_channels: Fixed channel count to pad/truncate to.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def prepare_pretokenize(self, signal, channel_tokens, sampling_rate):
        T, C_actual = signal.shape
        C = self.num_channels

        if C_actual > C:
            signal = signal[:, :C]
            channel_tokens = channel_tokens[:C]
            C_actual = C

        padded = np.zeros((C, T), dtype=signal.dtype)
        padded[:C_actual, :] = signal.T[:C_actual, :]

        mask = np.zeros(C, dtype=bool)
        mask[:C_actual] = True

        padded_tokens = np.zeros(C, dtype=channel_tokens.dtype)
        padded_tokens[:C_actual] = channel_tokens

        return {
            "input_values": torch.from_numpy(padded).float(),
            "input_channel_index": torch.from_numpy(padded_tokens).long(),
            "input_mask": torch.from_numpy(mask),
            "input_sampling_rate": torch.tensor(
                sampling_rate, dtype=torch.float32
            ),
            "input_seq_len": torch.tensor(T, dtype=torch.long),
        }

    def forward(self, input_values, **kwargs):
        return input_values


class PerChannelStrategy(ChannelStrategy):
    """Process each channel independently.

    ``forward()`` reshapes ``(B, C_pad, T)`` to ``(B*C_pad, 1, T)`` so
    the temporal embedding processes each channel as a separate item.
    The ``EEGTokenizer._reassemble_per_channel`` method later reshapes
    tokens back to ``(B, C*N, D)`` and adds channel identity embeddings.

    Args:
        max_channels: Maximum number of channels to pad to.
    """

    def __init__(self, max_channels: int):
        super().__init__()
        self.max_channels = max_channels

    def prepare_pretokenize(self, signal, channel_tokens, sampling_rate):
        T, C_actual = signal.shape
        C = self.max_channels

        if C_actual > C:
            signal = signal[:, :C]
            channel_tokens = channel_tokens[:C]
            C_actual = C

        padded = np.zeros((C, T), dtype=signal.dtype)
        padded[:C_actual, :] = signal.T[:C_actual, :]

        mask = np.zeros(C, dtype=bool)
        mask[:C_actual] = True

        padded_tokens = np.zeros(C, dtype=channel_tokens.dtype)
        padded_tokens[:C_actual] = channel_tokens

        return {
            "input_values": torch.from_numpy(padded).float(),
            "input_channel_index": torch.from_numpy(padded_tokens).long(),
            "input_mask": torch.from_numpy(mask),
            "input_sampling_rate": torch.tensor(
                sampling_rate, dtype=torch.float32
            ),
            "input_seq_len": torch.tensor(T, dtype=torch.long),
        }

    def forward(self, input_values, **kwargs):
        B, C, T = input_values.shape
        return input_values.reshape(B * C, 1, T)


class SpatialProjectionStrategy(ChannelStrategy):
    """Project variable channels to a fixed number of latent sources.

    ``forward()`` output: ``(B, num_sources, T)``.
    Can be followed by patching (``patch_duration`` set on the tokenizer)
    or used directly with CWT / PerTimepoint temporal embeddings.

    The ``projector`` argument controls how channels are mixed.  Built-in
    options (all in :mod:`foundry.models.embeddings.spatial`):

    * :class:`LinearSpatialProjector` — single shared linear layer.
    * :class:`SessionSpatialProjector` — per-session linear layers.
    * :class:`PerceiverSpatialProjector` — cross-attention bottleneck.

    Any ``nn.Module`` whose ``forward(x, **kwargs)`` maps
    ``(B, num_channels, T) → (B, num_sources, T)`` may be used.

    Args:
        num_channels: Fixed channel count to pad/truncate to before projection.
        num_sources: Number of latent sources after projection.
        projector: Spatial projector module.
    """

    def __init__(
        self,
        num_channels: int,
        num_sources: int,
        projector: nn.Module,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_sources = num_sources
        self.spatial = projector

    def prepare_pretokenize(self, signal, channel_tokens, sampling_rate):
        T, C_actual = signal.shape
        C = self.num_channels

        if C_actual > C:
            signal = signal[:, :C]
            channel_tokens = channel_tokens[:C]
            C_actual = C

        padded = np.zeros((C, T), dtype=signal.dtype)
        padded[:C_actual, :] = signal.T[:C_actual, :]

        mask = np.zeros(C, dtype=bool)
        mask[:C_actual] = True

        padded_tokens = np.zeros(C, dtype=channel_tokens.dtype)
        padded_tokens[:C_actual] = channel_tokens

        return {
            "input_values": torch.from_numpy(padded).float(),
            "input_channel_index": torch.from_numpy(padded_tokens).long(),
            "input_mask": torch.from_numpy(mask),
            "input_sampling_rate": torch.tensor(
                sampling_rate, dtype=torch.float32
            ),
            "input_seq_len": torch.tensor(T, dtype=torch.long),
        }

    def forward(self, input_values, **kwargs):
        return self.spatial(input_values, **kwargs)


__all__ = [
    "ChannelStrategy",
    "FixedChannelStrategy",
    "PerChannelStrategy",
    "SpatialProjectionStrategy",
]
