from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    if activation.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Available: {list(activations.keys())}"
        )
    return activations[activation.lower()]


class EmbeddingBase(nn.Module, ABC):
    """Abstract base class for all input embedding layers.

    Subclasses must implement ``pretokenize`` (data preparation during
    tokenization) and ``forward`` (the embedding computation itself).
    """

    @property
    def requires_patching(self) -> bool:
        """Whether the embedding expects patched input from ``patch_time_series``.

        Patch-based embeddings (default) receive ``(num_patches, C, patch_samples)``
        arrays in ``pretokenize``.  Embeddings that operate on the full time
        series (e.g. CWT) override this to ``False`` and receive the raw signal
        instead.
        """
        return True

    @abstractmethod
    def pretokenize(
        self,
        patches_array: np.ndarray,
        channel_tokens: np.ndarray,
    ) -> dict:
        """Transform patched signal into the format expected by ``forward``.

        Called once per sample during tokenization (before batching) so each
        embedding type can reshape, pad, or otherwise prepare the data.

        Args:
            patches_array: ``(num_patches, num_channels_actual, patch_samples)``
                raw patched signal containing only valid channels.
            channel_tokens: ``(num_channels_actual,)`` integer channel-token
                indices.

        Returns:
            dict with at minimum ``input_values``, ``input_channel_index``,
            and ``input_mask`` tensors.
        """
        ...

    @abstractmethod
    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor: ...


class FixedChannelWindowEmbedding(EmbeddingBase):
    """Base for embeddings that operate on a fixed channel count and patch size.

    Handles channel padding / truncation during ``pretokenize`` so that
    every sample presented to ``forward`` has exactly ``num_channels``
    channels and ``patch_samples`` time steps per patch.
    """

    def __init__(self, embed_dim: int, num_channels: int, patch_samples: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_samples = patch_samples

    def pretokenize(
        self,
        patches_array: np.ndarray,
        channel_tokens: np.ndarray,
    ) -> dict:
        num_patches, num_channels_actual, patch_samples = patches_array.shape
        num_channels = self.num_channels

        if num_channels_actual > num_channels:
            patches_array = patches_array[:, :num_channels, :]
            channel_tokens = channel_tokens[:num_channels]
            num_channels_actual = num_channels

        padded_signal = np.zeros(
            (num_patches, num_channels, patch_samples),
            dtype=patches_array.dtype,
        )
        padded_signal[:, :num_channels_actual, :] = patches_array

        padded_channel_tokens = np.zeros(
            num_channels, dtype=channel_tokens.dtype
        )
        padded_channel_tokens[:num_channels_actual] = channel_tokens

        channel_mask = np.zeros(num_channels, dtype=bool)
        channel_mask[:num_channels_actual] = True

        return {
            "input_values": torch.from_numpy(padded_signal).float(),
            "input_channel_index": torch.from_numpy(
                padded_channel_tokens
            ).long(),
            "input_mask": torch.from_numpy(channel_mask),
        }


__all__ = [
    "EmbeddingBase",
    "FixedChannelWindowEmbedding",
    "get_activation",
]
