import torch
import torch.nn as nn

from foundry.models.embeddings.base import FixedChannelWindowEmbedding


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


__all__ = ["LinearEmbedding"]
