from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.models.embeddings.activations import get_activation


class ResampleCNNEmbedding(nn.Module):
    """Temporal embedding via time-resampling followed by a 1-D CNN.

    Ablation baseline for :class:`CWTEmbedding`: instead of decomposing the
    signal with learnable wavelets, simply resample to a fixed number of time
    tokens and let a plain CNN learn temporal features directly.

    The resampling uses differentiable ``grid_sample`` (identical to the one
    inside ``CWTEmbedding``) so that variable-length sequences are handled
    correctly — only the valid region is interpolated, with zeros beyond.

    Args:
        embed_dim: Output embedding dimension per time token.
        num_sources: Number of input source channels (after spatial projection).
        target_time_tokens: Number of output time tokens after resampling.
        num_filters: Width of the convolutional layers.
        kernel_size: 1-D convolution kernel width (same-padded).
        num_conv_layers: Number of ``Conv1d → activation`` blocks.
        activation: Activation function name (see :func:`get_activation`).
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        target_time_tokens: int = 128,
        num_filters: int = 64,
        kernel_size: int = 7,
        num_conv_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.target_time_tokens = target_time_tokens

        layers: list[nn.Module] = []
        in_channels = num_sources
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    num_filters,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(get_activation(activation))
            in_channels = num_filters
        self.cnn = nn.Sequential(*layers)

        self.feature_proj = nn.Linear(num_filters, embed_dim)

        for m in self.cnn:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.feature_proj.weight, gain=1.0)
        nn.init.zeros_(self.feature_proj.bias)

    def _resample(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiably resample variable-length signals to a fixed length.

        Uses ``grid_sample`` on a ``(B, C, 1, Max_T)`` pseudo-image so that
        each batch item interpolates only within its valid region ``[0, seq_len-1]``.

        Args:
            x: ``(B, C, Max_T)`` zero-padded signal.
            seq_lens: ``(B,)`` true sample count per item.

        Returns:
            ``(B, C, target_time_tokens)``
        """
        B, C, Max_T = x.shape
        device = x.device
        orig_dtype = x.dtype

        # grid_sample does not support bfloat16
        if orig_dtype == torch.bfloat16:
            x = x.float()

        seq_lens = seq_lens.clamp(min=1, max=Max_T)

        # Map each item's valid range to grid_sample coordinates [-1, end_x]
        end_x = -1.0 + 2.0 * (seq_lens.float() - 1) / max(1, Max_T - 1)
        steps = torch.linspace(
            0.0, 1.0, self.target_time_tokens, device=device
        ).unsqueeze(0)
        x_coords = -1.0 + steps * (end_x.unsqueeze(1) - (-1.0))

        # Treat signal as single-row image: (B, C, 1, Max_T)
        grid_x = x_coords.unsqueeze(1)  # (B, 1, T_out)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, 1, T_out, 2)

        resampled = F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)

        if orig_dtype == torch.bfloat16:
            resampled = resampled.to(orig_dtype)
        return resampled

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_sampling_rate: torch.Tensor,
        input_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, num_sources, max_T)`` spatially-projected signal.
            input_sampling_rate: ``(B,)`` sampling rate per item (accepted for
                interface compatibility with :class:`CWTEmbedding`; unused).
            input_seq_len: ``(B,)`` true sample count per item.

        Returns:
            ``(B, target_time_tokens, embed_dim)``
        """
        resampled = self._resample(x, input_seq_len)
        features = self.cnn(resampled)
        return self.feature_proj(features.transpose(1, 2))


__all__ = ["ResampleCNNEmbedding"]
