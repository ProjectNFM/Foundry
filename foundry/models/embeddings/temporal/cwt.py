from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_MAG_EPS = 1e-6

FreqSpacing = Literal["linear", "log", "mel", "inverse"]

_VALID_SPACINGS: set[str] = {"linear", "log", "mel", "inverse"}

_MEL_BREAK_HZ = 700.0


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / _MEL_BREAK_HZ)


def _mel_to_hz(mel: float) -> float:
    return _MEL_BREAK_HZ * (10.0 ** (mel / 2595.0) - 1.0)


def generate_freqs(
    num_freqs: int,
    min_freq: float,
    max_freq: float,
    spacing: FreqSpacing = "log",
) -> list[float]:
    """Generate a list of center frequencies between *min_freq* and *max_freq*.

    Args:
        num_freqs: How many frequencies to produce.
        min_freq: Lowest frequency (Hz), inclusive.
        max_freq: Highest frequency (Hz), inclusive.
        spacing: Distribution of frequencies between the endpoints.

            * ``"linear"`` -- uniform spacing in Hz.
            * ``"log"`` -- uniform spacing in log-Hz (more resolution at low
              frequencies, natural for 1/f-like neural signals).
            * ``"mel"`` -- uniform spacing on the mel scale (perceptually
              motivated, less aggressive than log at the low end).
            * ``"inverse"`` -- uniform spacing in 1/Hz (the most aggressive
              low-frequency concentration).

    Returns:
        Sorted list of *num_freqs* floats in ``[min_freq, max_freq]``.
    """
    if num_freqs < 1:
        raise ValueError(f"num_freqs must be >= 1, got {num_freqs}")
    if min_freq <= 0:
        raise ValueError(f"min_freq must be > 0, got {min_freq}")
    if max_freq < min_freq:
        raise ValueError(
            f"max_freq ({max_freq}) must be >= min_freq ({min_freq})"
        )
    if spacing not in _VALID_SPACINGS:
        raise ValueError(
            f"Unknown spacing {spacing!r}. Choose from {sorted(_VALID_SPACINGS)}."
        )

    if min_freq == max_freq:
        return [min_freq] * num_freqs

    if num_freqs == 1:
        return [math.sqrt(min_freq * max_freq)]

    if spacing == "linear":
        step = (max_freq - min_freq) / (num_freqs - 1)
        return [min_freq + i * step for i in range(num_freqs)]

    if spacing == "log":
        log_min, log_max = math.log(min_freq), math.log(max_freq)
        step = (log_max - log_min) / (num_freqs - 1)
        return [math.exp(log_min + i * step) for i in range(num_freqs)]

    if spacing == "mel":
        mel_min, mel_max = _hz_to_mel(min_freq), _hz_to_mel(max_freq)
        step = (mel_max - mel_min) / (num_freqs - 1)
        return [_mel_to_hz(mel_min + i * step) for i in range(num_freqs)]

    # spacing == "inverse": uniform in 1/f, then reverse so result is ascending
    inv_min, inv_max = 1.0 / max_freq, 1.0 / min_freq
    step = (inv_max - inv_min) / (num_freqs - 1)
    return [1.0 / (inv_max - i * step) for i in range(num_freqs)]


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    threshold = 20.0
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def _resolve_init_freqs(
    init_freqs: list[float] | None,
    num_freqs: int | None,
    min_freq: float | None,
    max_freq: float | None,
    freq_spacing: FreqSpacing | None,
) -> list[float]:
    """Resolve the explicit-vs-generated frequency interface.

    Exactly one path must be specified: either *init_freqs* directly, or the
    ``(num_freqs, min_freq, max_freq)`` triple (with optional *freq_spacing*).
    """
    have_explicit = init_freqs is not None
    have_generated = num_freqs is not None

    if have_explicit and have_generated:
        raise ValueError(
            "Specify either 'init_freqs' or the (num_freqs, min_freq, "
            "max_freq) parameters, not both."
        )
    if not have_explicit and not have_generated:
        raise ValueError(
            "Must specify either 'init_freqs' or 'num_freqs' (with "
            "'min_freq' and 'max_freq')."
        )

    if have_explicit:
        return init_freqs  # type: ignore[return-value]

    if min_freq is None or max_freq is None:
        raise ValueError(
            "'min_freq' and 'max_freq' are required when using 'num_freqs'."
        )
    return generate_freqs(
        num_freqs=num_freqs,  # type: ignore[arg-type]
        min_freq=min_freq,
        max_freq=max_freq,
        spacing=freq_spacing or "log",
    )


class ContinuousCWTLayer(nn.Module):
    """Learnable Continuous Wavelet Transform with time resampling.

    Applies complex Morlet wavelets at **learnable** center frequencies
    (and cycle counts) to each channel independently, then resamples the
    resulting time-frequency map to a given number of time tokens using
    differentiable ``grid_sample``.

    The output token count is supplied at forward time so that it can be
    derived dynamically from ``target_token_rate × duration``.

    Frequencies can be supplied in two ways:

    1. **Explicit** -- pass ``init_freqs`` as a list of Hz values.
    2. **Generated** -- pass ``num_freqs``, ``min_freq``, ``max_freq``
       (and optionally ``freq_spacing``) to have them computed
       automatically via :func:`generate_freqs`.
    """

    def __init__(
        self,
        init_freqs: list[float] | None = None,
        *,
        num_freqs: int | None = None,
        min_freq: float | None = None,
        max_freq: float | None = None,
        freq_spacing: FreqSpacing | None = None,
        n_cycles: float = 2.5,
        phase_stability_eps: float = 1e-4,
        min_freq_hz: float = 0.1,
        min_cycles: float = 1.0,
    ):
        super().__init__()
        resolved_freqs = _resolve_init_freqs(
            init_freqs,
            num_freqs,
            min_freq,
            max_freq,
            freq_spacing,
        )
        init_freqs_tensor = torch.tensor(resolved_freqs, dtype=torch.float32)
        init_cycles_tensor = torch.full(
            (len(resolved_freqs),), float(n_cycles), dtype=torch.float32
        )
        if torch.any(init_freqs_tensor <= min_freq_hz):
            raise ValueError(
                f"All init_freqs must be > min_freq_hz ({min_freq_hz})."
            )
        if torch.any(init_cycles_tensor <= min_cycles):
            raise ValueError(f"n_cycles must be > min_cycles ({min_cycles}).")

        self.min_freq_hz = min_freq_hz
        self.min_cycles = min_cycles
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

    def get_watched_params(self) -> dict[str, torch.Tensor]:
        """Expose human-interpretable CWT parameters for logging."""
        return {
            "freqs_hz": self.freqs.detach(),
            "n_cycles": self.n_cycles.detach(),
        }

    def forward(
        self,
        x: torch.Tensor,
        fs: torch.Tensor,
        seq_lens: torch.Tensor,
        target_time_tokens: int,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, Max_T)`` signal (zero-padded beyond each sample's
               true length).
            fs: ``(B,)`` sampling rate per batch item.
            seq_lens: ``(B,)`` true sample count per batch item.
            target_time_tokens: Number of output time tokens along the
                resampled time axis.

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
            0.0, 1.0, target_time_tokens, device=device
        ).unsqueeze(0)
        x_coords = -1.0 + steps * (end_x.unsqueeze(1) - (-1.0))

        grid_x = x_coords.unsqueeze(1).expand(B, F_dim, target_time_tokens)
        grid_y = y_coords.view(1, F_dim, 1).expand(B, F_dim, target_time_tokens)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        continuous_complex = F.grid_sample(
            out_complex_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        continuous_complex = continuous_complex.view(
            B, C, 2, F_dim, target_time_tokens
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


def _apply_highpass(
    x: torch.Tensor,
    sampling_rate: torch.Tensor,
    window_sec: float,
) -> torch.Tensor:
    """Subtract a causal running mean to remove DC drift before the CWT."""
    max_fs = sampling_rate.max().item()
    window = max(3, round(window_sec * max_fs))
    if window % 2 == 0:
        window += 1
    B, C, T = x.shape
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    padded = F.pad(x, (window - 1, 0), mode="reflect")
    running_mean = F.conv1d(padded.reshape(B * C, 1, -1), kernel).reshape(
        B, C, T
    )
    return x - running_mean


def _condition_scalogram(
    mag: torch.Tensor,
    phase: torch.Tensor,
    *,
    log_mag: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply optional magnitude conditioning to a CWT scalogram.

    Args:
        mag: ``(B, S, F, T)`` magnitude.
        phase: ``(B, S, F, T)`` phase.
        log_mag: Compress magnitudes with ``log(mag + eps)``.

    Returns:
        Conditioned (mag, phase) with the same shapes.
    """
    if log_mag:
        mag = torch.log(mag + _LOG_MAG_EPS)
    return mag, phase


class CWTEmbedding(nn.Module):
    """Temporal embedding via learnable CWT.

    Operates on spatially-projected signal (``num_sources`` channels) and
    produces time tokens through differentiable wavelet analysis and time
    resampling.

    The output token count is determined at runtime from ``target_token_rate``
    and the input sequence duration, ensuring a consistent temporal resolution
    regardless of input length.

    The spatial projection is handled by the upstream channel strategy;
    this module is purely temporal.

    Frequencies can be supplied in two ways (same interface as
    :class:`ContinuousCWTLayer`):

    1. **Explicit** -- pass ``init_freqs`` as a list of Hz values.
    2. **Generated** -- pass ``num_freqs``, ``min_freq``, ``max_freq``
       (and optionally ``freq_spacing``) to have them computed via
       :func:`generate_freqs`.

    Optional scalogram conditioning (all default to ``False``):

    * ``highpass`` -- subtract a causal running mean from the input signal
      before the CWT, removing DC drift and slow offsets.
    * ``log_mag`` -- apply ``log(mag + eps)`` to the CWT magnitude,
      compressing dynamic range for robustness to additive corruptions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        target_token_rate: float = 100.0,
        init_freqs: list[float] | None = None,
        *,
        num_freqs: int | None = None,
        min_freq: float | None = None,
        max_freq: float | None = None,
        freq_spacing: FreqSpacing | None = None,
        n_cycles: float = 2.5,
        highpass: bool = False,
        highpass_window_sec: float = 0.05,
        log_mag: bool = False,
    ):
        super().__init__()
        resolved_freqs = _resolve_init_freqs(
            init_freqs,
            num_freqs,
            min_freq,
            max_freq,
            freq_spacing,
        )

        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.target_token_rate = target_token_rate
        self._num_freqs = len(resolved_freqs)
        self._highpass = highpass
        self._highpass_window_sec = highpass_window_sec
        self._log_mag = log_mag

        self.cwt = ContinuousCWTLayer(
            init_freqs=resolved_freqs,
            n_cycles=n_cycles,
        )

        feat_dim = num_sources * 2 * len(resolved_freqs)
        self.feature_proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_uniform_(self.feature_proj.weight, gain=1.0)
        nn.init.zeros_(self.feature_proj.bias)

    def _compute_target_tokens(
        self,
        input_seq_len: torch.Tensor,
        input_sampling_rate: torch.Tensor,
    ) -> int:
        durations = input_seq_len.float() / input_sampling_rate
        max_duration = durations.max().item()
        return max(1, round(self.target_token_rate * max_duration))

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
        if self._highpass:
            x = _apply_highpass(
                x, input_sampling_rate, self._highpass_window_sec
            )

        target_time_tokens = self._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt(
            x, input_sampling_rate, input_seq_len, target_time_tokens
        )

        B, S, _two, F_dim, T = cwt_out.shape
        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]

        if self._log_mag:
            mag, phase = _condition_scalogram(mag, phase, log_mag=self._log_mag)

        cwt_out = torch.stack([mag, phase], dim=2)
        cwt_flat = cwt_out.permute(0, 4, 1, 2, 3).reshape(B, T, S * 2 * F_dim)

        return self.feature_proj(cwt_flat)


class CWTCNNEmbedding(nn.Module):
    """Temporal embedding via learnable CWT followed by a 1-D CNN.

    Combines CWT's sampling-rate-invariant frequency decomposition with a
    convolutional stack that learns temporal patterns across frequency bins.
    The CWT provides an interpretable time-frequency representation; the CNN
    refines it by learning cross-frequency and local-temporal interactions.

    The pipeline is:
        CWT → (B, S*2*F, T) → Conv1d stack → (B, num_filters, T) → Linear → (B, T, embed_dim)

    Frequencies can be supplied in two ways (same interface as
    :class:`ContinuousCWTLayer`):

    1. **Explicit** -- pass ``init_freqs`` as a list of Hz values.
    2. **Generated** -- pass ``num_freqs``, ``min_freq``, ``max_freq``
       (and optionally ``freq_spacing``) to have them computed via
       :func:`generate_freqs`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        target_token_rate: float = 100.0,
        init_freqs: list[float] | None = None,
        *,
        num_freqs: int | None = None,
        min_freq: float | None = None,
        max_freq: float | None = None,
        freq_spacing: FreqSpacing | None = None,
        n_cycles: float = 2.5,
        num_filters: int = 64,
        kernel_size: int = 9,
        num_conv_layers: int = 2,
        activation: str = "gelu",
        highpass: bool = False,
        highpass_window_sec: float = 0.05,
        log_mag: bool = False,
    ):
        super().__init__()
        from foundry.models.embeddings.activations import get_activation

        resolved_freqs = _resolve_init_freqs(
            init_freqs,
            num_freqs,
            min_freq,
            max_freq,
            freq_spacing,
        )

        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.target_token_rate = target_token_rate
        self._num_freqs = len(resolved_freqs)
        self._highpass = highpass
        self._highpass_window_sec = highpass_window_sec
        self._log_mag = log_mag

        self.cwt = ContinuousCWTLayer(
            init_freqs=resolved_freqs,
            n_cycles=n_cycles,
        )

        cwt_channels = num_sources * 2 * len(resolved_freqs)
        layers: list[nn.Module] = []
        in_channels = cwt_channels
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

    def _compute_target_tokens(
        self,
        input_seq_len: torch.Tensor,
        input_sampling_rate: torch.Tensor,
    ) -> int:
        durations = input_seq_len.float() / input_sampling_rate
        max_duration = durations.max().item()
        return max(1, round(self.target_token_rate * max_duration))

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
        if self._highpass:
            x = _apply_highpass(
                x, input_sampling_rate, self._highpass_window_sec
            )

        target_time_tokens = self._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt(
            x, input_sampling_rate, input_seq_len, target_time_tokens
        )

        B, S, _two, F_dim, T = cwt_out.shape

        if self._log_mag:
            mag = cwt_out[:, :, 0, :, :]
            phase = cwt_out[:, :, 1, :, :]
            mag, phase = _condition_scalogram(mag, phase, log_mag=self._log_mag)
            cwt_out = torch.stack([mag, phase], dim=2)

        cwt_flat = cwt_out.reshape(B, S * 2 * F_dim, T)

        features = self.cnn(cwt_flat)
        return self.feature_proj(features.transpose(1, 2))


__all__ = [
    "ContinuousCWTLayer",
    "CWTEmbedding",
    "CWTCNNEmbedding",
    "generate_freqs",
]
