import torch
import torch.nn as nn


class PerTimepointEmbedding(nn.Module):
    """Project each timepoint independently via a linear layer.

    No patching or resampling is applied.  Variable sequence lengths are
    handled by the caller via padding and masking.

    Args:
        embed_dim: Output embedding dimension.
        input_dim: Size of each timepoint vector (e.g. ``num_sources`` after
            spatial projection, or ``1`` for per-channel mode).
    """

    def __init__(self, embed_dim: int, input_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) padded to max T in the batch.

        Returns:
            (B, T, embed_dim)
        """
        return self.projection(x)


__all__ = ["PerTimepointEmbedding"]
