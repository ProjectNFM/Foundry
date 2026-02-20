import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    """
    Converts variable-sized EEG tokens to fixed-size embeddings using a linear projection.

    Uses dynamic projection layers created on-the-fly based on input dimensions.
    Each unique patch_samples size gets its own projection layer.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of output embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict()
        self.register_buffer("_device_tracker", torch.zeros(1))

    def get_projection(self, patch_samples: int) -> nn.Module:
        """
        Get or create projection layer for given patch size.

        Args:
            patch_samples: Number of samples in each patch

        Returns:
            Linear projection layer that maps patches to embed_dim
        """
        key = str(patch_samples)
        if key not in self.projections:
            projection = nn.Linear(patch_samples, self.embed_dim)
            nn.init.xavier_uniform_(projection.weight, gain=1.0)
            nn.init.zeros_(projection.bias)
            projection = projection.to(self._device_tracker.device)
            self.projections[key] = projection
        return self.projections[key]

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens to embeddings.

        Args:
            input_values: Tokens of shape (batch_size, num_tokens, patch_samples)

        Returns:
            Embeddings of shape (batch_size, num_tokens, embed_dim)
        """
        batch_size, num_tokens, patch_samples = input_values.shape
        projection = self.get_projection(patch_samples)

        flattened = input_values.view(batch_size * num_tokens, patch_samples)
        embeddings = projection(flattened).view(batch_size, num_tokens, self.embed_dim)

        return embeddings
