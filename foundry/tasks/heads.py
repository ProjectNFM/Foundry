import torch
import torch.nn as nn


class ReadoutHead(nn.Module):
    """Pure projection from backbone embeddings to task predictions.

    No loss, no metrics, no data logic. Just a forward pass.
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


class MLPReadoutHead(nn.Module):
    """Multi-layer projection head (for SSL projection, deeper readouts)."""

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
