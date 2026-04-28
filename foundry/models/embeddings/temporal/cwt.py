from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    threshold = 20.0
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


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
        phase_stability_eps: float = 1e-4,
        min_freq_hz: float = 0.1,
        min_cycles: float = 1.0,
    ):
        super().__init__()
        init_freqs_tensor = torch.tensor(init_freqs, dtype=torch.float32)
        init_cycles_tensor = torch.full(
            (len(init_freqs),), float(n_cycles), dtype=torch.float32
        )
        if torch.any(init_freqs_tensor <= min_freq_hz):
            raise ValueError(
                f"All init_freqs must be > min_freq_hz ({min_freq_hz})."
            )
        if torch.any(init_cycles_tensor <= min_cycles):
            raise ValueError(f"n_cycles must be > min_cycles ({min_cycles}).")

        self.min_freq_hz = min_freq_hz
        self.min_cycles = min_cycles
        self.target_time_tokens = target_time_tokens
        self.phase_stability_eps = phase_stability_eps
        self.freqs_unconstrained = nn.Parameter(
            _inverse_softplus(init_freqs_tensor - min_freq_hz)
        )
        self.n_cycles_unconstrained = nn.Parameter(
            _inverse_softplus(init_cycles_tensor - min_cycles)
        )

    @property
    def freqs(self) -> torch.Tensor:
        return self.min_freq_hz + F.softplus(self.freqs_unconstrained)

    @property
    def n_cycles(self) -> torch.Tensor:
        return self.min_cycles + F.softplus(self.n_cycles_unconstrained)

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
        F_dim = self.freqs_unconstrained.numel()
        device = x.device
        orig_dtype = x.dtype

        if torch.any(~torch.isfinite(fs)) or torch.any(fs <= 0):
            raise ValueError(
                f"input_sampling_rate must be finite and > 0. Got {fs}."
            )

        # FFT and grid_sample do not support bfloat16; run in float32 and
        # restore dtype at the end.
        if orig_dtype == torch.bfloat16:
            x = x.float()
        dtype = x.dtype

        f = self.freqs.to(device=device, dtype=dtype).view(1, F_dim, 1)
        n_c = self.n_cycles.to(device=device, dtype=dtype).view(1, F_dim, 1)

        K = Max_T if Max_T % 2 != 0 else Max_T + 1
        t = torch.arange(-(K // 2), K // 2 + 1, device=device, dtype=dtype)
        t = t.view(1, 1, K)
        t_sec = t / fs.view(B, 1, 1)

        sigma = n_c / (2 * math.pi * f)
        envelope_exponent = -(t_sec**2) / (2 * sigma**2)
        envelope = torch.exp(envelope_exponent.clamp(min=-60.0))

        real_wavelet = torch.cos(2 * math.pi * f * t_sec) * envelope
        imag_wavelet = torch.sin(2 * math.pi * f * t_sec) * envelope

        # For Morlet components, sqrt(real^2 + imag^2) equals the envelope.
        # Using envelope directly avoids undefined sqrt gradients at zero.
        norm_factor = torch.sum(envelope, dim=-1, keepdim=True) + 1e-8
        real_wavelet = real_wavelet / norm_factor
        imag_wavelet = imag_wavelet / norm_factor

        # FFT-based correlation keeps per-sample wavelets, so mixed fs in a
        # batch is naturally supported without approximating to one global fs.
        kernel_real = real_wavelet.flip(-1)
        kernel_imag = imag_wavelet.flip(-1)
        full_len = Max_T + K - 1
        n_fft = 1 << (full_len - 1).bit_length()

        x_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        real_fft = torch.fft.rfft(kernel_real, n=n_fft, dim=-1)
        imag_fft = torch.fft.rfft(kernel_imag, n=n_fft, dim=-1)

        out_real_full = torch.fft.irfft(
            x_fft.unsqueeze(2) * real_fft.unsqueeze(1),
            n=n_fft,
            dim=-1,
        )[..., :full_len]
        out_imag_full = torch.fft.irfft(
            x_fft.unsqueeze(2) * imag_fft.unsqueeze(1),
            n=n_fft,
            dim=-1,
        )[..., :full_len]

        start = K // 2
        end = start + Max_T
        out_real = out_real_full[..., start:end]
        out_imag = out_imag_full[..., start:end]

        out_complex = torch.stack([out_real, out_imag], dim=2)
        out_complex_flat = out_complex.view(B, C * 2, F_dim, Max_T)

        y_coords = torch.linspace(-1.0, 1.0, F_dim, device=device)
        seq_lens = seq_lens.clamp(min=1, max=Max_T)
        end_x = -1.0 + 2.0 * (seq_lens.float() - 1) / max(1, (Max_T - 1))
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

        mag_sq = cont_real.square() + cont_imag.square()
        cont_mag = torch.sqrt(mag_sq + 1e-8)
        raw_phase = torch.atan2(cont_imag, cont_real) / math.pi

        # Phase is undefined for near-zero magnitude and can explode gradients.
        # Smoothly suppressing phase there keeps learnable CWT params stable.
        phase_weight = mag_sq / (mag_sq + self.phase_stability_eps)
        cont_phase = raw_phase * phase_weight

        result = torch.stack([cont_mag, cont_phase], dim=2)
        return result.to(orig_dtype) if orig_dtype != dtype else result


class CWTEmbedding(nn.Module):
    """Temporal embedding via learnable CWT.

    Operates on spatially-projected signal (``num_sources`` channels) and
    produces a fixed number of time tokens through differentiable wavelet
    analysis and time resampling.

    The spatial projection is handled by the upstream channel strategy;
    this module is purely temporal.

    Args:
        embed_dim: Output embedding dimension per time token.
        num_sources: Number of input source channels (after spatial projection).
        init_freqs: Initial CWT center frequencies in Hz.
        target_time_tokens: Number of time tokens produced by the CWT layer.
        n_cycles: Initial wavelet width (time-frequency trade-off).
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        init_freqs: list[float],
        target_time_tokens: int = 128,
        n_cycles: float = 7.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.target_time_tokens = target_time_tokens
        self._num_freqs = len(init_freqs)

        self.cwt = ContinuousCWTLayer(
            init_freqs=init_freqs,
            target_time_tokens=target_time_tokens,
            n_cycles=n_cycles,
        )

        feat_dim = num_sources * 2 * len(init_freqs)
        self.feature_proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_uniform_(self.feature_proj.weight, gain=1.0)
        nn.init.zeros_(self.feature_proj.bias)

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
            ``(B, target_time_tokens, embed_dim)``
        """
        # (B, num_sources, 2, F, target_time_tokens)
        cwt_out = self.cwt(x, input_sampling_rate, input_seq_len)

        B, S, two, F_dim, T = cwt_out.shape
        cwt_flat = cwt_out.permute(0, 4, 1, 2, 3).reshape(B, T, S * two * F_dim)

        return self.feature_proj(cwt_flat)


__all__ = ["ContinuousCWTLayer", "CWTEmbedding"]
