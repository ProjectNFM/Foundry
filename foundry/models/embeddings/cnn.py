import torch
import torch.nn as nn

from foundry.models.embeddings.utils import get_activation


class CNNEmbedding(nn.Module):
    """
    Converts variable-sized EEG patches to fixed-size embeddings using 1D CNN.

    Uses dynamic CNN networks created on-the-fly based on input dimensions.
    Each unique combination of (time_steps, channels) gets its own CNN.
    The CNN operates on the time dimension and flattens the output before projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        activation: str = "gelu",
    ):
        """
        Args:
            embed_dim: Dimension of output embeddings
            num_filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            activation: Activation function name (relu, gelu, silu, tanh, etc.)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.projections = nn.ModuleDict()

    def get_projection(self, time_steps: int, channels: int) -> nn.Module:
        """
        Get or create CNN projection for given dimensions.

        Args:
            time_steps: Number of time steps in each patch
            channels: Number of channels in each patch

        Returns:
            CNN module that maps patches to embed_dim
        """
        key = f"{time_steps}_{channels}"
        if key not in self.projections:
            conv_out_time = time_steps - self.kernel_size + 1
            flattened_dim = conv_out_time * self.num_filters

            cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    padding=0,
                ),
                get_activation(self.activation),
                nn.Flatten(),
                nn.Linear(flattened_dim, self.embed_dim),
            )

            for module in cnn:
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)

            self.projections[key] = cnn
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

        reshaped = input_values.view(batch_size * num_patches, time_steps, channels)
        reshaped = reshaped.transpose(1, 2)

        embeddings = projection(reshaped).view(
            batch_size, num_patches, self.embed_dim
        )

        return embeddings
