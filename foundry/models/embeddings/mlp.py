import torch
import torch.nn as nn

from foundry.models.embeddings.utils import get_activation


class MLPEmbedding(nn.Module):
    """
    Converts variable-sized EEG tokens to fixed-size embeddings using MLP.

    Uses dynamic MLP networks created on-the-fly based on input dimensions.
    Each unique patch_samples size gets its own MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        activation: str = "gelu",
    ):
        """
        Args:
            embed_dim: Dimension of output embeddings
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name (relu, gelu, silu, tanh, etc.)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.projections = nn.ModuleDict()

    def get_projection(self, patch_samples: int) -> nn.Module:
        """
        Get or create MLP projection for given patch size.

        Args:
            patch_samples: Number of samples in each patch

        Returns:
            MLP that maps patches to embed_dim
        """
        key = str(patch_samples)
        if key not in self.projections:
            layers = []
            input_dim = patch_samples

            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(get_activation(self.activation))
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, self.embed_dim))

            mlp = nn.Sequential(*layers)

            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)

            self.projections[key] = mlp
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
        embeddings = projection(flattened).view(
            batch_size, num_tokens, self.embed_dim
        )

        return embeddings
