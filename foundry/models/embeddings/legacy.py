import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class EmbeddingBase(nn.Module, ABC):
    """Abstract base class for input embedding layers.

    .. deprecated::
        Use ``EEGTokenizer`` with a channel strategy and temporal embedding
        instead.  Temporal embeddings now inherit directly from
        ``nn.Module``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from EmbeddingBase which is deprecated. "
            "Temporal embeddings should inherit from nn.Module directly.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def requires_patching(self) -> bool:
        return True

    @abstractmethod
    def pretokenize(
        self,
        patches_array: np.ndarray,
        channel_tokens: np.ndarray,
    ) -> dict: ...

    @abstractmethod
    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor: ...


class FixedChannelWindowEmbedding(EmbeddingBase):
    """Base for embeddings on a fixed channel count and patch size.

    .. deprecated::
        Use ``FixedChannelStrategy`` for channel handling and inherit
        temporal embeddings from ``nn.Module`` directly.
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
]
