import torch
import torch.nn as nn

from foundry.models.embeddings.utils import get_activation


class MLPEmbedding(nn.Module):
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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_samples = patch_samples

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
