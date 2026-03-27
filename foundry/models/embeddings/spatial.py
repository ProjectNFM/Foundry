import torch
import torch.nn as nn


class SessionSpatialProjector(nn.Module):
    """Project variable-channel recordings to a fixed number of latent sources.

    Each session (recording setup) has its own linear layer mapping its channel
    count to a shared hidden dimension (or directly to ``num_sources``).  An
    optional shared MLP is applied afterwards so that all sessions are mapped
    into the same representational space.

    Args:
        session_configs: Mapping of ``session_id`` (str) to the number of
            channels for that session.
        num_sources: Number of latent sources to produce.
        shared_hidden_dim: If given, session-specific layers first project to
            this size, then a shared MLP maps to ``num_sources``.
    """

    def __init__(
        self,
        session_configs: dict[str, int],
        num_sources: int,
        shared_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.shared_hidden_dim = shared_hidden_dim

        out_dim = (
            shared_hidden_dim if shared_hidden_dim is not None else num_sources
        )

        self.session_layers = nn.ModuleDict()
        for session_id, num_channels in session_configs.items():
            self.session_layers[str(session_id)] = nn.Linear(
                num_channels, out_dim
            )

        if shared_hidden_dim is not None:
            self.shared_mlp = nn.Sequential(
                nn.GELU(), nn.Linear(shared_hidden_dim, num_sources)
            )
        else:
            self.shared_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        session_ids: list,
        channel_counts: list | torch.Tensor,
        seq_lens: list | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(Batch, Max_Channels, Max_Time)`` padded with zeros.
            session_ids: Session identifier for each batch item.
            channel_counts: True channel count per batch item.
            seq_lens: True time-sample count per batch item.

        Returns:
            ``(Batch, num_sources, Max_Time)``
        """
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

            if self.shared_mlp is not None:
                projected = self.shared_mlp(projected)

            projected = projected.transpose(0, 1)  # (num_sources, Max_T)

            # Re-zero time-padding corrupted by linear bias
            if t < Max_T:
                projected[:, t:] = 0.0

            out[i] = projected

        return out


__all__ = ["SessionSpatialProjector"]
