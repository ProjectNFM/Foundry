from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.models.embeddings.activations import get_activation
from foundry.models.embeddings.temporal.base import TemporalEmbedding

_ANTIALIAS_ORDER = 6


class ResampleCNNEmbedding(TemporalEmbedding):
    """Temporal embedding via time-resampling followed by a 1-D CNN.

    Ablation baseline for :class:`CWTEmbedding`: instead of decomposing the
    signal with learnable wavelets, simply resample to a fixed number of time
    tokens and let a plain CNN learn temporal features directly.

    Resampling uses differentiable ``grid_sample`` on a ``(B, C, 1, Max_T)``
    pseudo-image so that variable-length sequences are handled correctly —
    only the valid region is interpolated, with zeros beyond.

    Before downsampling, an FFT-based Butterworth low-pass filter removes
    frequencies above the target Nyquist to prevent aliasing.

    The output token count is determined at runtime from ``target_token_rate``
    and the input sequence duration, ensuring convolution kernels always
    cover the same physical time regardless of input length.

    Args:
        embed_dim: Output embedding dimension per time token.
        num_sources: Number of input source channels (after spatial projection).
        target_token_rate: Output token rate in Hz.  The number of time tokens
            is ``round(target_token_rate × duration_seconds)`` per batch.
        num_filters: Width of the convolutional layers.
        kernel_size: 1-D convolution kernel width (same-padded).
        num_conv_layers: Number of ``Conv1d → activation`` blocks.
        activation: Activation function name (see :func:`get_activation`).
        antialias: Apply a low-pass filter before downsampling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        target_token_rate: float = 100.0,
        num_filters: int = 64,
        kernel_size: int = 9,
        num_conv_layers: int = 2,
        activation: str = "gelu",
        antialias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.target_token_rate = target_token_rate
        self.antialias = antialias

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

    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        return max(1, round(self.target_token_rate * sequence_length))

    @property
    def has_fixed_token_count(self) -> bool:
        return True

    def _compute_target_tokens(
        self,
        input_seq_len: torch.Tensor,
        input_sampling_rate: torch.Tensor,
    ) -> int:
        durations = input_seq_len.float() / input_sampling_rate
        max_duration = durations.max().item()
        return max(1, round(self.target_token_rate * max_duration))

    def _antialias_lowpass(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        target_time_tokens: int,
    ) -> torch.Tensor:
        """FFT-based Butterworth low-pass filter for anti-aliased downsampling.

        Per-item cutoff is set to the target Nyquist (``target_time_tokens /
        seq_len`` of the input Nyquist).  When all items are being upsampled
        the filter is a no-op.
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        max_ratio = seq_lens.float().max().item() / target_time_tokens
        if max_ratio <= 1.0:
            return x

        n_fft = 1 << (T - 1).bit_length()
        X = torch.fft.rfft(x, n=n_fft, dim=-1)
        num_bins = X.shape[-1]

        freqs = torch.arange(num_bins, device=device, dtype=dtype) / max(
            1, num_bins - 1
        )
        cutoffs = (float(target_time_tokens) / seq_lens.float()).clamp(max=1.0)

        f_ratio = freqs.unsqueeze(0) / cutoffs.unsqueeze(1).clamp(min=1e-6)
        mask = torch.rsqrt(1.0 + f_ratio.pow(2 * _ANTIALIAS_ORDER))

        X_filtered = X * mask.unsqueeze(1)
        return torch.fft.irfft(X_filtered, n=n_fft, dim=-1)[..., :T]

    def _resample(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        target_time_tokens: int,
    ) -> torch.Tensor:
        """Differentiably resample variable-length signals to a fixed length.

        Uses ``grid_sample`` on a ``(B, C, 1, Max_T)`` pseudo-image so that
        each batch item interpolates only within its valid region ``[0, seq_len-1]``.

        Args:
            x: ``(B, C, Max_T)`` zero-padded signal.
            seq_lens: ``(B,)`` true sample count per item.
            target_time_tokens: Number of output time tokens.

        Returns:
            ``(B, C, target_time_tokens)``
        """
        B, C, Max_T = x.shape
        device = x.device
        orig_dtype = x.dtype

        if orig_dtype == torch.bfloat16:
            x = x.float()

        seq_lens = seq_lens.clamp(min=1, max=Max_T)

        if self.antialias:
            x = self._antialias_lowpass(x, seq_lens, target_time_tokens)

        end_x = -1.0 + 2.0 * (seq_lens.float() - 1) / max(1, Max_T - 1)
        steps = torch.linspace(
            0.0, 1.0, target_time_tokens, device=device
        ).unsqueeze(0)
        x_coords = -1.0 + steps * (end_x.unsqueeze(1) - (-1.0))

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
            input_sampling_rate: ``(B,)`` sampling rate per item.
            input_seq_len: ``(B,)`` true sample count per item.

        Returns:
            ``(B, target_time_tokens, embed_dim)`` where
            ``target_time_tokens = round(target_token_rate × max_duration)``.
        """
        target_time_tokens = self._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        resampled = self._resample(x, input_seq_len, target_time_tokens)
        features = self.cnn(resampled)
        return self.feature_proj(features.transpose(1, 2))


__all__ = ["ResampleCNNEmbedding"]
