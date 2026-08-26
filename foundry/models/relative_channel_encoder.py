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
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dynamic channel embeddings from temporal tokens.

        Args:
            tokens: (B, C, N, D_tok) per-channel temporal token embeddings.
            channel_mask: (B, C) boolean mask — True for real channels.
            token_mask: Optional (B, C, N) boolean mask — True for tokens to
                include in temporal pooling. When None, all tokens are used.
                Used to exclude masked tokens from contributing to channel
                embeddings during masked pretraining.

        Returns:
            (B, C, channel_emb_dim) channel embeddings with padded channels
            zeroed out.
        """
        B, C, N, D = tokens.shape

        # Stage 1: attention-weighted temporal pooling
        scores = self.time_attn_score(tokens).squeeze(-1)  # (B, C, N)

        if token_mask is not None:
            effective_channel_mask = channel_mask & token_mask.any(dim=-1)
            scores = scores.masked_fill(~token_mask, float("-inf"))
        else:
            effective_channel_mask = channel_mask

        mask_expand = effective_channel_mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(~mask_expand, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, C, N)
        weights = weights.masked_fill(~mask_expand, 0.0)
        if token_mask is not None:
            weights = weights.masked_fill(~token_mask, 0.0)
        # Elementwise reduction is algebraically equivalent to the einsum but
        # avoids its expensive BmmBackward kernel for the many small
        # per-channel matrices in EEG batches.
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=2)  # (B, C, D)

        # Stage 2: cross-channel attention
        # key_padding_mask: True means "ignore this position" in PyTorch MHA
        key_padding_mask = ~effective_channel_mask  # (B, C)
        attended, _ = self.cross_channel_attn(
            pooled,
            pooled,
            pooled,
            key_padding_mask=key_padding_mask,
        )  # (B, C, D)

        # Stage 3: projection + norm
        out = self.proj(attended)  # (B, C, channel_emb_dim)
        out = self.norm(out)

        # Zero out padded/fully-masked channels
        out = out.masked_fill(~effective_channel_mask.unsqueeze(-1), 0.0)

        return out
