import torch
import torch.nn as nn

from foundry.models.embeddings.activations import get_activation


class PatchCNNEmbedding(nn.Module):
    """Convert patched EEG signal to embeddings via 1D CNN.

    Treats channels as Conv1d input channels and convolves over the time
    dimension.

    Args:
        embed_dim: Output embedding dimension.
        num_input_channels: Number of input channels per patch.
        patch_samples: Number of time samples per patch.
        num_filters: Number of convolutional filters.
        kernel_size: Convolutional kernel width.
        activation: Activation function name.
    """

    def __init__(
        self,
        embed_dim: int,
        num_input_channels: int,
        patch_samples: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_input_channels = num_input_channels
        self.patch_samples = patch_samples

        conv_out_time = patch_samples - kernel_size + 1
        flattened_dim = conv_out_time * num_filters

        self.cnn = nn.Sequential(
            nn.Conv1d(num_input_channels, num_filters, kernel_size, padding=0),
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

    def forward(self, patches: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            patches: (B, P, C, S)

        Returns:
            (B, P, embed_dim)
        """
        B, P, C, S = patches.shape
        x = patches.reshape(B * P, C, S)
        return self.cnn(x).reshape(B, P, self.embed_dim)


__all__ = ["PatchCNNEmbedding"]
