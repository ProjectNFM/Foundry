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


class LinearEmbedding(FixedChannelWindowEmbedding):
    """Converts patched EEG signal to embeddings via linear projection.

    Flattens the (channels, time) dimensions and projects to embed_dim.
    """

    def __init__(self, embed_dim: int, num_channels: int, patch_samples: int):
        super().__init__(embed_dim, num_channels, patch_samples)
        self.projection = nn.Linear(num_channels * patch_samples, embed_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_values: (batch, num_patches, num_channels, patch_samples)

        Returns:
            (batch, num_patches, embed_dim)
        """
        batch, P, C, S = input_values.shape
        return self.projection(input_values.reshape(batch, P, C * S))


class MLPEmbedding(FixedChannelWindowEmbedding):
    """Converts patched EEG signal to embeddings via MLP.

    Flattens the (channels, time) dimensions and passes through hidden layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        patch_samples: int,
        hidden_dims: list[int],
        activation: str = "gelu",
    ):
        super().__init__(embed_dim, num_channels, patch_samples)

        layers = []
        input_dim = num_channels * patch_samples
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            input_values: (batch, num_patches, num_channels, patch_samples)

        Returns:
            (batch, num_patches, embed_dim)
        """
        batch, P, C, S = input_values.shape
        return self.mlp(input_values.reshape(batch, P, C * S))


class CNNEmbedding(FixedChannelWindowEmbedding):
    """Converts patched EEG signal to embeddings via 1D CNN.

    Treats channels as Conv1d input channels and convolves over the time
    dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        patch_samples: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        activation: str = "gelu",
    ):
        super().__init__(embed_dim, num_channels, patch_samples)

        conv_out_time = patch_samples - kernel_size + 1
        flattened_dim = conv_out_time * num_filters

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, num_filters, kernel_size, padding=0),
            get_activation(activation),
            nn.Flatten(),
            nn.Linear(flattened_dim, embed_dim),
        )

        for module in self.cnn:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            input_values: (batch, num_patches, num_channels, patch_samples)

        Returns:
            (batch, num_patches, embed_dim)
        """
        batch, P, C, S = input_values.shape
        x = input_values.reshape(batch * P, C, S)
        return self.cnn(x).reshape(batch, P, self.embed_dim)


__all__ = [
    "EmbeddingBase",
    "FixedChannelWindowEmbedding",
    "LinearEmbedding",
    "MLPEmbedding",
    "CNNEmbedding",
    "get_activation",
]
