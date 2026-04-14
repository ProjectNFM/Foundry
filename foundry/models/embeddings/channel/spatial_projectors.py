import torch
import torch.nn as nn
from torch_brain.nn import RotaryCrossAttention, RotaryTimeEmbedding


class LinearSpatialProjector(nn.Module):
    """Project channels to latent sources via a single shared linear layer.

    Args:
        num_channels: Number of input channels (after padding).
        num_sources: Number of latent sources to produce.
    """

    def __init__(self, num_channels: int, num_sources: int):
        super().__init__()
        self.num_sources = num_sources
        self.linear = nn.Linear(num_channels, num_sources)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, T)`` padded channel signals.

        Returns:
            ``(B, num_sources, T)``
        """
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class SessionSpatialProjector(nn.Module):
    """Project variable-channel recordings to a fixed number of latent sources.

    Each session (recording setup) has its own layer(s) mapping its channel
    count to ``num_sources``.

    Two modes of operation:

    1. **Direct** (default) — each session has a single linear layer mapping
       ``num_channels → num_sources``.
    2. **Per-session MLP** (``hidden_dim``) — each session gets its own
       two-layer MLP: ``num_channels → hidden_dim → num_sources``.

    Args:
        session_configs: Mapping of ``session_id`` (str) to the number of
            channels for that session.
        num_sources: Number of latent sources to produce.
        hidden_dim: If given, each session gets its own two-layer MLP with
            this hidden size.
    """

    def __init__(
        self,
        session_configs: dict[str, int],
        num_sources: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.hidden_dim = hidden_dim

        self.session_layers = nn.ModuleDict()
        if hidden_dim is not None:
            for session_id, num_channels in session_configs.items():
                self.session_layers[str(session_id)] = nn.Sequential(
                    nn.Linear(num_channels, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_sources),
                )
        else:
            for session_id, num_channels in session_configs.items():
                self.session_layers[str(session_id)] = nn.Linear(
                    num_channels, num_sources
                )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: ``(Batch, Max_Channels, Max_Time)`` padded with zeros.
            **kwargs: Must contain ``input_session_ids``,
                ``input_channel_counts``, and ``input_seq_len``.

        Returns:
            ``(Batch, num_sources, Max_Time)``
        """
        session_ids = kwargs["input_session_ids"]
        channel_counts = kwargs["input_channel_counts"]
        seq_lens = kwargs["input_seq_len"]

        B, Max_C, Max_T = x.shape
        device = x.device

        out = torch.zeros(
            B, self.num_sources, Max_T, device=device, dtype=x.dtype
        )

        for i in range(B):
            sess_id = str(
                session_ids[i].item()
                if torch.is_tensor(session_ids[i])
                else session_ids[i]
            )
            c = channel_counts[i]
            t = seq_lens[i]

            x_i = x[i, :c, :]
            x_i_t = x_i.transpose(0, 1)  # (Max_T, c)

            projected = self.session_layers[sess_id](x_i_t)
            projected = projected.transpose(0, 1)  # (num_sources, Max_T)

            # Re-zero time-padding corrupted by linear bias
            if t < Max_T:
                projected[:, t:] = 0.0

            out[i] = projected

        return out


class PerceiverSpatialProjector(nn.Module):
    """Perceiver-style cross-attention bottleneck for spatial projection.

    Maps an arbitrary number of physical channels (N) into a fixed number of
    latent, universal brain sources (K) via cross-attention.  At each
    timepoint, K learnable, dataset-agnostic query vectors attend to the N
    input channels—serving as keys and values—yielding a fixed ``(K × T)``
    spatial projection regardless of the original sensor count.

    Delegates to :class:`torch_brain.nn.RotaryCrossAttention` for the
    attention computation.  Positional (rotary) embeddings are neutralised
    by feeding identical (zero) timestamps for every position, so the
    rotation reduces to the identity.

    Args:
        num_sources: Number of latent sources (K) to produce.
        d_attn: Dimensionality of the cross-attention space.
        num_heads: Number of attention heads.
        dim_head: Per-head dimension.  Defaults to ``d_attn // num_heads``.
    """

    def __init__(
        self,
        num_sources: int,
        d_attn: int = 64,
        num_heads: int = 4,
        dim_head: int | None = None,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.d_attn = d_attn
        dim_head = dim_head or d_attn // num_heads

        self.queries = nn.Parameter(torch.empty(num_sources, d_attn))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.context_proj = nn.Linear(1, d_attn)

        self.cross_attn = RotaryCrossAttention(
            dim=d_attn,
            context_dim=d_attn,
            heads=num_heads,
            dim_head=dim_head,
            rotate_value=False,
        )

        # Values of t_min / t_max are irrelevant — timestamps are always zero
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=1e-4,
            t_max=1.0,
        )

        self.to_scalar = nn.Linear(d_attn, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, T)`` padded channel signals.
            **kwargs: May contain ``input_mask`` ``(B, C)`` indicating valid
                channels.

        Returns:
            ``(B, num_sources, T)``
        """
        input_mask = kwargs.get("input_mask")
        B, C, T = x.shape
        K = self.num_sources

        # (B, C, T) → (B*T, C, d_attn)
        context = self.context_proj(x.permute(0, 2, 1).unsqueeze(-1))
        context = context.reshape(B * T, C, self.d_attn)

        # (K, d_attn) → (B*T, K, d_attn)
        q = self.queries.unsqueeze(0).expand(B * T, -1, -1)

        # Zero timestamps → identity rotation (cos 0 = 1, sin 0 = 0)
        zero_q = torch.zeros(B * T, K, device=x.device)
        zero_kv = torch.zeros(B * T, C, device=x.device)
        q_pos = self.rotary_emb(zero_q)
        kv_pos = self.rotary_emb(zero_kv)

        context_mask = None
        if input_mask is not None:
            context_mask = (
                input_mask.unsqueeze(1).expand(B, T, C).reshape(B * T, C)
            )

        out = self.cross_attn(q, context, q_pos, kv_pos, context_mask)

        # (B*T, K, d_attn) → scalar → (B, K, T)
        out = self.to_scalar(out).squeeze(-1).reshape(B, T, K).permute(0, 2, 1)

        return out


__all__ = [
    "LinearSpatialProjector",
    "SessionSpatialProjector",
    "PerceiverSpatialProjector",
]
