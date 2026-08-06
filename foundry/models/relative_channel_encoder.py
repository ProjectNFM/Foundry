"""Relative inter-channel attention encoder for dynamic channel embeddings.

Computes per-channel identity embeddings by attending across channels,
replacing the static per-session lookup table with signal-conditioned
representations that generalize across subjects.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeChannelEncoder(nn.Module):
    """Signal-conditioned channel embedding via cross-channel attention.

    Three stages:
    1. Attention-weighted temporal pooling: (B, C, N, D_tok) -> (B, C, D_tok)
    2. Cross-channel multi-head attention: each channel attends to all others
    3. Projection + LayerNorm: (B, C, D_tok) -> (B, C, channel_emb_dim)

    Padded channels (indicated by channel_mask=False) are ignored in
    cross-attention via key_padding_mask and zeroed in the output.

    Args:
        token_dim: Dimension of temporal token embeddings (D_tok).
        channel_emb_dim: Output channel embedding dimension.
        num_heads: Number of attention heads for cross-channel attention.
    """

    def __init__(
        self,
        token_dim: int,
        channel_emb_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.channel_emb_dim = channel_emb_dim

        # Bias is omitted: scores feed into softmax over N, which is shift-invariant.
        self.time_attn_score = nn.Linear(token_dim, 1, bias=False)
        self.cross_channel_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(token_dim, channel_emb_dim)
        self.norm = nn.LayerNorm(channel_emb_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamic channel embeddings from temporal tokens.

        Args:
            tokens: (B, C, N, D_tok) per-channel temporal token embeddings.
            channel_mask: (B, C) boolean mask — True for real channels.

        Returns:
            (B, C, channel_emb_dim) channel embeddings with padded channels
            zeroed out.
        """
        B, C, N, D = tokens.shape

        # Stage 1: attention-weighted temporal pooling
        scores = self.time_attn_score(tokens).squeeze(-1)  # (B, C, N)
        mask_expand = channel_mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(~mask_expand, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, C, N)
        weights = weights.masked_fill(~mask_expand, 0.0)
        pooled = torch.einsum("bcn,bcnd->bcd", weights, tokens)  # (B, C, D)

        # Stage 2: cross-channel attention
        # key_padding_mask: True means "ignore this position" in PyTorch MHA
        key_padding_mask = ~channel_mask  # (B, C)
        # Reshape to (B, C, D) for batch_first MHA
        attended, _ = self.cross_channel_attn(
            pooled,
            pooled,
            pooled,
            key_padding_mask=key_padding_mask,
        )  # (B, C, D)

        # Stage 3: projection + norm
        out = self.proj(attended)  # (B, C, channel_emb_dim)
        out = self.norm(out)

        # Zero out padded channels
        out = out.masked_fill(~channel_mask.unsqueeze(-1), 0.0)

        return out
