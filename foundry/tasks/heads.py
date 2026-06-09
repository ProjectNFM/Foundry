"""Readout heads for projecting backbone embeddings to task predictions."""

import torch
import torch.nn as nn


class ReadoutHead(nn.Module):
    """Single linear projection from backbone embeddings to task predictions.

    The default readout for classification and regression tasks. Maps each
    embedding vector to a logits or regression vector of length ``output_dim``.

    Args:
        embed_dim: Dimension of backbone output embeddings.
        output_dim: Number of logits (classification) or target dimensions
            (regression).

    Shape:
        - Input: ``(N, embed_dim)``
        - Output: ``(N, output_dim)``
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


class MLPReadoutHead(nn.Module):
    """Multi-layer MLP projection from embeddings to task predictions.

    Use for deeper readouts or SSL projection heads where a single linear layer
    is insufficient. Builds ``num_layers - 1`` hidden blocks
    (linear → activation) followed by a final linear to ``output_dim``.

    Args:
        embed_dim: Dimension of backbone output embeddings.
        output_dim: Number of logits or target dimensions.
        hidden_dim: Width of hidden layers. Defaults to ``embed_dim``.
        num_layers: Total number of linear layers, including the output layer.
            ``num_layers=1`` is equivalent to :class:`ReadoutHead`.
        activation: Nonlinearity between hidden layers. One of ``"gelu"`` or
            ``"relu"``.

    Shape:
        - Input: ``(N, embed_dim)``
        - Output: ``(N, output_dim)``
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]

        layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)
