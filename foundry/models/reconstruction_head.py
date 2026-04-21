from __future__ import annotations

import torch.nn as nn


class ReconstructionHead(nn.Module):
    """Projects decoder embeddings back to raw signal space for reconstruction.

    Args:
        embed_dim: Input embedding dimension from the decoder.
        output_dim: Target reconstruction dimensionality.  Depends on the
            tokenizer combination (e.g. ``num_channels * patch_samples``
            for fixed-channel patch tokenizers).
        hidden_dim: Hidden layer dimension.  Defaults to ``embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Project decoder embeddings to signal space.

        Args:
            x: ``(N, embed_dim)`` decoder embeddings at masked positions.

        Returns:
            ``(N, output_dim)`` reconstructed signal vectors.
        """
        return self.net(x)


__all__ = ["ReconstructionHead"]
