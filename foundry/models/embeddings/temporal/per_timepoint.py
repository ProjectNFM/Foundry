import torch
import torch.nn as nn


class PerTimepointLinearEmbedding(nn.Module):
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


class PerTimepointIdentityEmbedding(nn.Module):
    """Use timepoint features as tokens without a learned projection.

    This is useful when channel strategy output already matches the tokenizer
    embedding dimension and no additional temporal transform is desired.

    Args:
        embed_dim: Required token feature dimension.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feature_dim) padded to max T in the batch.

        Returns:
            (B, T, embed_dim)
        """
        if x.ndim != 3:
            raise ValueError(
                "IdentityTemporalEmbedding expects input with shape (B, T, D); "
                f"got tensor with {x.ndim} dimensions."
            )

        input_dim = x.shape[-1]
        if input_dim != self.embed_dim:
            raise ValueError(
                "IdentityTemporalEmbedding requires feature_dim to match "
                f"embed_dim; got feature_dim={input_dim}, "
                f"embed_dim={self.embed_dim}."
            )

        return x


__all__ = ["PerTimepointLinearEmbedding", "PerTimepointIdentityEmbedding"]
