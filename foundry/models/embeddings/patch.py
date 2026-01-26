import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts variable-sized EEG patches to fixed-size embeddings.

    Uses dynamic projection layers created on-the-fly based on input dimensions.
    Each unique combination of (time_steps, channels) gets its own projection layer.
    This is a standalone component - it just does the embedding projection.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of output embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict()

    def get_projection(self, time_steps: int, channels: int) -> nn.Module:
        """
        Get or create projection layer for given dimensions.

        Args:
            time_steps: Number of time steps in each patch
            channels: Number of channels in each patch

        Returns:
            Linear projection layer that maps flattened patches to embed_dim
        """
        key = f"{time_steps}_{channels}"
        if key not in self.projections:
            projection = nn.Linear(time_steps * channels, self.embed_dim)
            nn.init.xavier_uniform_(projection.weight, gain=1.0)
            nn.init.zeros_(projection.bias)
            self.projections[key] = projection
        return self.projections[key]

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to embeddings.

        Args:
            input_values: Patches of shape (batch_size, num_patches, time_steps, channels)

        Returns:
            Embeddings of shape (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, time_steps, channels = input_values.shape
        projection = self.get_projection(time_steps, channels)

        flattened = input_values.view(batch_size * num_patches, -1)
        embeddings = projection(flattened).view(
            batch_size, num_patches, self.embed_dim
        )

        return embeddings
