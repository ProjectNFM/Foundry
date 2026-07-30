import torch
import torch.nn as nn

from foundry.models.embeddings.activations import get_activation
from foundry.models.embeddings.temporal.base import TemporalEmbedding
from foundry.models.signal_preparation import compute_num_patches


class PatchMLPEmbedding(TemporalEmbedding):
    """Convert patched EEG signal to embeddings via MLP.

    Flattens the (channels, time) dimensions and passes through hidden layers.

    Args:
        embed_dim: Output embedding dimension.
        num_input_channels: Number of input channels per patch.
        patch_samples: Number of time samples per patch.
        hidden_dims: List of hidden layer sizes.
        activation: Activation function name.
    """

    def __init__(
        self,
        embed_dim: int,
        num_input_channels: int,
        patch_samples: int,
        hidden_dims: list[int],
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_input_channels = num_input_channels
        self.patch_samples = patch_samples

        layers = []
        input_dim = num_input_channels * patch_samples
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

    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        num_samples = round(sampling_rate * sequence_length)
        return compute_num_patches(
            num_samples, self.patch_samples, self.patch_samples
        )

    @property
    def has_fixed_token_count(self) -> bool:
        return True

    def forward(self, patches: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            patches: (B, P, C, S)

        Returns:
            (B, P, embed_dim)
        """
        B, P, C, S = patches.shape
        return self.mlp(patches.reshape(B, P, C * S))


__all__ = ["PatchMLPEmbedding"]
