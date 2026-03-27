from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.models.embeddings.base import EmbeddingBase
from foundry.models.embeddings.spatial import SessionSpatialProjector


class ContinuousCWTLayer(nn.Module):
    """Learnable Continuous Wavelet Transform with time resampling.

    Applies complex Morlet wavelets at **learnable** center frequencies
    (and cycle counts) to each channel independently, then resamples the
    resulting time-frequency map to a fixed number of time tokens using
    differentiable ``grid_sample``.

    Args:
        init_freqs: Initial center frequencies (Hz) for the wavelets.
        target_time_tokens: Number of output time tokens after resampling.
        n_cycles: Initial wavelet width parameter (trades time vs frequency
            resolution).
    """

    def __init__(
        self,
        init_freqs: list[float],
        target_time_tokens: int,
        n_cycles: float = 7.0,
    ):
        super().__init__()
        self.freqs = nn.Parameter(torch.tensor(init_freqs, dtype=torch.float32))
        self.target_time_tokens = target_time_tokens
        self.n_cycles = nn.Parameter(
            torch.full((len(init_freqs),), float(n_cycles), dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        fs: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, Max_T)`` signal (zero-padded beyond each sample's
               true length).
            fs: ``(B,)`` sampling rate per batch item.
            seq_lens: ``(B,)`` true sample count per batch item.

        Returns:
            ``(B, C, 2, F, target_time_tokens)`` -- magnitude and phase
            across learnable frequency bins and resampled time tokens.
        """
        B, C, Max_T = x.shape
        F_dim = len(self.freqs)
        device = x.device
        dtype = x.dtype

        # -- wavelet generation --
        f = torch.clamp(self.freqs, min=0.1).view(1, F_dim, 1)
        n_c = torch.clamp(self.n_cycles, min=1.0).view(1, F_dim, 1)

        K = Max_T if Max_T % 2 != 0 else Max_T + 1
        t = torch.arange(-(K // 2), K // 2 + 1, device=device, dtype=dtype)
        t = t.view(1, 1, K)
        t_sec = t / fs.view(B, 1, 1)

        sigma = n_c / (2 * math.pi * f)
        envelope = torch.exp(-(t_sec**2) / (2 * sigma**2))

        real_wavelet = torch.cos(2 * math.pi * f * t_sec) * envelope
        imag_wavelet = torch.sin(2 * math.pi * f * t_sec) * envelope

        norm_factor = (
            torch.sum(
                torch.sqrt(real_wavelet**2 + imag_wavelet**2),
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )
        real_wavelet = real_wavelet / norm_factor
        imag_wavelet = imag_wavelet / norm_factor

        # -- grouped convolution --
        x_reshaped = x.reshape(1, B * C, Max_T)
        weight_real = (
            real_wavelet.unsqueeze(1)
            .expand(B, C, F_dim, K)
            .reshape(B * C * F_dim, 1, K)
        )
        weight_imag = (
            imag_wavelet.unsqueeze(1)
            .expand(B, C, F_dim, K)
            .reshape(B * C * F_dim, 1, K)
        )

        out_real = F.conv1d(
            x_reshaped, weight_real, groups=B * C, padding="same"
        )
        out_imag = F.conv1d(
            x_reshaped, weight_imag, groups=B * C, padding="same"
        )

        out_real = out_real.view(B, C, F_dim, Max_T)
        out_imag = out_imag.view(B, C, F_dim, Max_T)

        # -- continuous time projection via grid_sample --
        out_complex = torch.stack([out_real, out_imag], dim=2)
        out_complex_flat = out_complex.view(B, C * 2, F_dim, Max_T)

        y_coords = torch.linspace(-1.0, 1.0, F_dim, device=device)
        end_x = -1.0 + 2.0 * (seq_lens.float() - 1) / (Max_T - 1)
        steps = torch.linspace(
            0.0, 1.0, self.target_time_tokens, device=device
        ).unsqueeze(0)
        x_coords = -1.0 + steps * (end_x.unsqueeze(1) - (-1.0))

        grid_x = x_coords.unsqueeze(1).expand(B, F_dim, self.target_time_tokens)
        grid_y = y_coords.view(1, F_dim, 1).expand(
            B, F_dim, self.target_time_tokens
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)

        continuous_complex = F.grid_sample(
            out_complex_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        continuous_complex = continuous_complex.view(
            B, C, 2, F_dim, self.target_time_tokens
        )

        cont_real = continuous_complex[:, :, 0, :, :]
        cont_imag = continuous_complex[:, :, 1, :, :]

        cont_mag = torch.sqrt(cont_real**2 + cont_imag**2 + 1e-8)
        cont_phase = torch.atan2(cont_imag, cont_real) / math.pi

        return torch.stack([cont_mag, cont_phase], dim=2)


class CWTEmbedding(EmbeddingBase):
    """Embedding via spatial projection followed by learnable CWT.

    Unlike patch-based embeddings, this operates on the **full time series**
    and produces a fixed number of time tokens through differentiable wavelet
    analysis and time resampling.

    When ``session_configs`` is ``None`` (the default), a single linear
    spatial projection is used (channels are padded/truncated to
    ``num_channels``, then projected to ``num_sources``).  When provided,
    a :class:`SessionSpatialProjector` handles per-session variable channel
    counts.

    Args:
        embed_dim: Output embedding dimension per time token.
        num_channels: Fixed channel count (pad/truncate) when session-specific
            projection is not used.
        num_sources: Number of latent spatial sources after projection.
        init_freqs: Initial CWT center frequencies in Hz.
        target_time_tokens: Number of time tokens produced by the CWT layer.
        n_cycles: Initial wavelet width (time-frequency trade-off).
        shared_spatial_hidden_dim: Optional hidden size between the spatial
            projection and the source output.
        session_configs: If provided, mapping of ``session_id`` (str) to its
            channel count; enables per-session spatial projection.
    """

    @property
    def requires_patching(self) -> bool:
        return False

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        num_sources: int,
        init_freqs: list[float],
        target_time_tokens: int = 128,
        n_cycles: float = 7.0,
        shared_spatial_hidden_dim: int | None = None,
        session_configs: dict[str, int] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.num_sources = num_sources
        self.target_time_tokens = target_time_tokens
        self._num_freqs = len(init_freqs)

        # -- spatial projection --
        if session_configs is not None:
            self.spatial = SessionSpatialProjector(
                session_configs=session_configs,
                num_sources=num_sources,
                shared_hidden_dim=shared_spatial_hidden_dim,
            )
            self._use_session_spatial = True
        else:
            if shared_spatial_hidden_dim is not None:
                self.spatial = nn.Sequential(
                    nn.Linear(num_channels, shared_spatial_hidden_dim),
                    nn.GELU(),
                    nn.Linear(shared_spatial_hidden_dim, num_sources),
                )
            else:
                self.spatial = nn.Linear(num_channels, num_sources)
            self._use_session_spatial = False

        # -- CWT --
        self.cwt = ContinuousCWTLayer(
            init_freqs=init_freqs,
            target_time_tokens=target_time_tokens,
            n_cycles=n_cycles,
        )

        # -- feature projection: (sources * 2 * freqs) -> embed_dim --
        feat_dim = num_sources * 2 * len(init_freqs)
        self.feature_proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_uniform_(self.feature_proj.weight, gain=1.0)
        nn.init.zeros_(self.feature_proj.bias)

    # --------------------------------------------------------------------- #
    # pretokenize (called per-sample before batching)
    # --------------------------------------------------------------------- #
    def pretokenize(
        self,
        signal: np.ndarray,
        channel_tokens: np.ndarray,
        *,
        sampling_rate: float,
        sequence_length: float,
    ) -> dict:
        """Prepare a single sample for the CWT embedding.

        Unlike patch-based embeddings, this receives the full (unpatched)
        signal as ``(num_samples, num_channels_actual)`` and returns it
        in channel-first layout with channel padding/truncation.

        Args:
            signal: ``(num_samples, num_channels_actual)``
            channel_tokens: ``(num_channels_actual,)``
            sampling_rate: Sampling rate in Hz.
            sequence_length: Duration of the context window in seconds.

        Returns:
            dict with ``input_values``, ``input_channel_index``,
            ``input_mask``, ``input_sampling_rate``, ``input_seq_len``,
            and ``input_timestamps``.
        """
        num_samples, num_channels_actual = signal.shape
        num_channels = self.num_channels

        if num_channels_actual > num_channels:
            signal = signal[:, :num_channels]
            channel_tokens = channel_tokens[:num_channels]
            num_channels_actual = num_channels

        padded = np.zeros((num_channels, num_samples), dtype=signal.dtype)
        padded[:num_channels_actual, :] = signal.T[:num_channels_actual, :]

        padded_channel_tokens = np.zeros(
            num_channels, dtype=channel_tokens.dtype
        )
        padded_channel_tokens[:num_channels_actual] = channel_tokens

        channel_mask = np.zeros(num_channels, dtype=bool)
        channel_mask[:num_channels_actual] = True

        timestamps = np.linspace(
            0.0, sequence_length, self.target_time_tokens, dtype=np.float32
        )

        return {
            "input_values": torch.from_numpy(padded).float(),
            "input_channel_index": torch.from_numpy(
                padded_channel_tokens
            ).long(),
            "input_mask": torch.from_numpy(channel_mask),
            "input_sampling_rate": torch.tensor(
                sampling_rate, dtype=torch.float32
            ),
            "input_seq_len": torch.tensor(num_samples, dtype=torch.long),
            "input_timestamps": torch.from_numpy(timestamps).float(),
        }

    # --------------------------------------------------------------------- #
    # forward (called on batched data)
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_values: torch.Tensor,
        *,
        input_sampling_rate: torch.Tensor | None = None,
        input_seq_len: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_values: ``(batch, num_channels, max_T)``
            input_sampling_rate: ``(batch,)`` sampling rate per item.
            input_seq_len: ``(batch,)`` true sample count per item.

        Returns:
            ``(batch, target_time_tokens, embed_dim)``
        """
        B, C, Max_T = input_values.shape

        if input_sampling_rate is None:
            raise ValueError(
                "CWTEmbedding requires input_sampling_rate "
                "(pass sampling_rate through the tokenizer)."
            )
        if input_seq_len is None:
            raise ValueError(
                "CWTEmbedding requires input_seq_len "
                "(pass seq_len through the tokenizer)."
            )

        # -- spatial projection --
        if self._use_session_spatial:
            session_ids = kwargs.get("input_session_ids")
            channel_counts = kwargs.get("input_channel_counts")
            sources = self.spatial(
                input_values, session_ids, channel_counts, input_seq_len
            )
        else:
            # (B, C, T) -> (B, T, C) -> linear -> (B, T, S) -> (B, S, T)
            sources = self.spatial(input_values.transpose(1, 2)).transpose(1, 2)

        # -- CWT --
        # (B, num_sources, 2, F, target_time_tokens)
        cwt_out = self.cwt(sources, input_sampling_rate, input_seq_len)

        # -- flatten features per time token --
        # permute to (B, target_time_tokens, num_sources, 2, F)
        B2, S, two, F_dim, T = cwt_out.shape
        cwt_flat = cwt_out.permute(0, 4, 1, 2, 3).reshape(
            B2, T, S * two * F_dim
        )

        return self.feature_proj(cwt_flat)


__all__ = ["ContinuousCWTLayer", "CWTEmbedding"]
