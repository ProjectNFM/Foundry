from typing import Optional

import torch
import torch.nn as nn
from torch_brain.nn import (
    FeedForward,
    RotaryCrossAttention,
    RotarySelfAttention,
)


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dim_head: int = 64,
        cross_heads: int = 1,
        ffn_dropout: float = 0.2,
        atn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_atn = RotaryCrossAttention(
            dim=embed_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(dim=embed_dim, dropout=ffn_dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        latent_timestamp_emb: torch.Tensor,
        input_timestamp_emb: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode inputs into latents via cross-attention."""
        latents = latents + self.cross_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.ffn(latents)
        return latents


class PerceiverProcessor(nn.Module):
    """
    Processor that refines latents with self-attention layers.

    Standalone component that can be used independently.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int = 2,
        dim_head: int = 64,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=lin_dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=embed_dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(embed_dim),
                            FeedForward(dim=embed_dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

    def forward(
        self, latents: torch.Tensor, latent_timestamp_emb: torch.Tensor
    ) -> torch.Tensor:
        """Process latents with self-attention."""
        for self_attn, self_ff in self.layers:
            latents = latents + self.dropout(
                self_attn(latents, latent_timestamp_emb)
            )
            latents = latents + self.dropout(self_ff(latents))
        return latents


class PerceiverDecoder(nn.Module):
    """
    Decoder that generates outputs from latents via cross-attention.

    Standalone component that can be used independently.
    """

    def __init__(
        self,
        embed_dim: int,
        dim_head: int = 64,
        cross_heads: int = 1,
        ffn_dropout: float = 0.2,
        atn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_atn = RotaryCrossAttention(
            dim=embed_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(dim=embed_dim, dropout=ffn_dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        query_timestamp_emb: torch.Tensor,
        latent_timestamp_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latents into outputs via cross-attention."""
        queries = queries + self.cross_atn(
            queries, latents, query_timestamp_emb, latent_timestamp_emb
        )
        outputs = queries + self.ffn(queries)
        return outputs


class PerceiverIOBackbone(nn.Module):
    """
    Complete Perceiver IO backbone combining encoder, processor, and decoder.

    This is a convenience wrapper - you can also use the individual components
    (PerceiverEncoder, PerceiverProcessor, PerceiverDecoder) separately for
    more flexibility.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = PerceiverEncoder(
            embed_dim=embed_dim,
            dim_head=dim_head,
            cross_heads=cross_heads,
            ffn_dropout=ffn_dropout,
            atn_dropout=atn_dropout,
        )

        self.processor = PerceiverProcessor(
            embed_dim=embed_dim,
            depth=depth,
            dim_head=dim_head,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
        )

        self.decoder = PerceiverDecoder(
            embed_dim=embed_dim,
            dim_head=dim_head,
            cross_heads=cross_heads,
            ffn_dropout=ffn_dropout,
            atn_dropout=atn_dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_timestamp_emb: torch.Tensor,
        latents: torch.Tensor,
        latent_timestamp_emb: torch.Tensor,
        output_queries: torch.Tensor,
        output_timestamp_emb: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full encoder-processor-decoder pass."""
        latents = self.encoder(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = self.processor(latents, latent_timestamp_emb)
        outputs = self.decoder(
            output_queries, latents, output_timestamp_emb, latent_timestamp_emb
        )
        return outputs
