"""Representation stability under signal corruptions: CWT vs ResampleCNN.

Tests whether the CWT layer's frequency decomposition provides inherent
robustness to common signal corruptions compared to a plain CNN operating
on the raw waveform. All models are randomly initialized — we are testing
architectural inductive bias, not learned features.

For each (architecture, corruption, severity) triple the script measures
cosine similarity and relative L2 distance between clean and corrupted
representations, averaged over many random signals and random weight
initializations.

Outputs (in docs/figures/cwt_stability/):
    cosine_similarity.pdf  — 6-panel figure, one per corruption
    relative_l2.pdf        — matching figure for relative L2 distance

Usage:
    uv run scripts/analysis/cwt_stability.py
    uv run scripts/analysis/cwt_stability.py --n-signals 100 --n-seeds 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from foundry.models.embeddings.temporal.cwt import CWTEmbedding
from foundry.models.embeddings.temporal.resample_cnn import ResampleCNNEmbedding

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

STYLE = {
    "CWT": {"color": "#0072B2", "marker": "o", "ls": "-"},
    "CNN": {"color": "#D55E00", "marker": "s", "ls": "-"},
    "log-mag": {"color": "#56B4E9", "marker": "^", "ls": "--"},
    "unit-mag": {"color": "#CC79A7", "marker": "v", "ls": ":"},
    "log-mag+zscore": {"color": "#E69F00", "marker": "P", "ls": "--"},
    "highpass": {"color": "#009E73", "marker": "D", "ls": "-."},
    "min2Hz": {"color": "#F0E442", "marker": "X", "ls": "-."},
    "freq-norm": {"color": "#882255", "marker": "h", "ls": ":"},
    "inst-freq": {"color": "#44AA99", "marker": "<", "ls": "--"},
    "sep-proj": {"color": "#AA4499", "marker": ">", "ls": ":"},
    "taper": {"color": "#332288", "marker": "d", "ls": "-."},
    "scaled-nc": {"color": "#999933", "marker": "p", "ls": "--"},
    "hp+fnorm": {"color": "#000000", "marker": "*", "ls": "-"},
    "hp+log+fnorm": {"color": "#E31A1C", "marker": "H", "ls": "-"},
    "hp+log": {"color": "#009E73", "marker": "D", "ls": "-"},
}

LOG_SCALE_CORRUPTIONS = {"amplitude_scale"}

SAMPLING_RATE = 500.0
DURATION = 1.0
NUM_SAMPLES = int(SAMPLING_RATE * DURATION)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_RATE)
    freqs[0] = 1.0
    spectrum /= np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples)
    return pink / (np.std(pink) + 1e-12)


def generate_synthetic_eeg(
    n_signals: int, rng: np.random.Generator
) -> np.ndarray:
    """Sum of canonical EEG oscillations + 1/f background.

    Returns (n_signals, NUM_SAMPLES) float64 array, z-scored per signal.
    """
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    bands = [
        (2.0, 1.0),  # delta
        (6.0, 0.7),  # theta
        (10.0, 0.5),  # alpha
        (20.0, 0.3),  # beta
    ]
    signals = np.zeros((n_signals, NUM_SAMPLES))
    for i in range(n_signals):
        sig = _pink_noise(NUM_SAMPLES, rng) * 0.3
        for freq, amp in bands:
            phase = rng.uniform(0, 2 * np.pi)
            sig += amp * np.sin(2 * np.pi * freq * t + phase)
        std = np.std(sig)
        signals[i] = (sig - np.mean(sig)) / (std + 1e-12)
    return signals


# ---------------------------------------------------------------------------
# Corruption functions — each returns a corrupted copy
# ---------------------------------------------------------------------------


def corrupt_amplitude_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    return x * factor


def corrupt_white_noise(
    x: torch.Tensor, snr_db: float, rng: np.random.Generator
) -> torch.Tensor:
    noise = torch.from_numpy(
        rng.standard_normal(x.shape).astype(np.float32)
    ).to(x.device)
    signal_power = x.square().mean()
    noise_power = noise.square().mean()
    scale = torch.sqrt(
        signal_power / (noise_power * 10 ** (snr_db / 10) + 1e-12)
    )
    return x + noise * scale


def corrupt_spikes(
    x: torch.Tensor, n_spikes: int, rng: np.random.Generator
) -> torch.Tensor:
    out = x.clone()
    B, C, T = out.shape
    for b in range(B):
        positions = rng.integers(0, T, size=n_spikes)
        signs = rng.choice([-1.0, 1.0], size=n_spikes).astype(np.float32)
        std = out[b].std().item()
        for pos, sign in zip(positions, signs):
            out[b, :, pos] += sign * 10.0 * std
    return out


def corrupt_zero_dropout(
    x: torch.Tensor, fraction: float, rng: np.random.Generator
) -> torch.Tensor:
    out = x.clone()
    B, C, T = out.shape
    n_zero = max(1, int(T * fraction))
    for b in range(B):
        start = rng.integers(0, max(1, T - n_zero))
        out[b, :, start : start + n_zero] = 0.0
    return out


def corrupt_dc_offset(x: torch.Tensor, offset_std_mult: float) -> torch.Tensor:
    std = x.std(dim=-1, keepdim=True)
    return x + offset_std_mult * std


def corrupt_line_noise(
    x: torch.Tensor, amplitude_std_mult: float
) -> torch.Tensor:
    B, C, T = x.shape
    t = torch.linspace(0, DURATION, T, device=x.device)
    sin60 = torch.sin(2 * torch.pi * 60.0 * t).unsqueeze(0).unsqueeze(0)
    std = x.std(dim=-1, keepdim=True)
    return x + amplitude_std_mult * std * sin60


# ---------------------------------------------------------------------------
# Corruption registry
# ---------------------------------------------------------------------------


@dataclass
class Corruption:
    name: str
    display_name: str
    severity_values: list[float]
    severity_label: str
    apply_fn: object  # callable


CORRUPTIONS = [
    Corruption(
        name="amplitude_scale",
        display_name="Amplitude Scaling",
        severity_values=[0.01, 0.1, 0.5, 2.0, 10.0, 100.0],
        severity_label="Scale factor",
        apply_fn=lambda x, s, rng: corrupt_amplitude_scale(x, s),
    ),
    Corruption(
        name="white_noise",
        display_name="Additive White Noise",
        severity_values=[20, 10, 5, 0, -5, -10],
        severity_label="SNR (dB)",
        apply_fn=lambda x, s, rng: corrupt_white_noise(x, s, rng),
    ),
    Corruption(
        name="spikes",
        display_name="Random Spikes",
        severity_values=[1, 2, 5, 10, 20],
        severity_label="Number of spikes",
        apply_fn=lambda x, s, rng: corrupt_spikes(x, int(s), rng),
    ),
    Corruption(
        name="zero_dropout",
        display_name="Zero Dropout",
        severity_values=[0.05, 0.10, 0.20, 0.40, 0.60],
        severity_label="Fraction zeroed",
        apply_fn=lambda x, s, rng: corrupt_zero_dropout(x, s, rng),
    ),
    Corruption(
        name="dc_offset",
        display_name="DC Offset",
        severity_values=[0.1, 0.5, 1.0, 5.0, 10.0],
        severity_label="Offset (× signal std)",
        apply_fn=lambda x, s, rng: corrupt_dc_offset(x, s),
    ),
    Corruption(
        name="line_noise",
        display_name="60 Hz Line Noise",
        severity_values=[0.1, 0.5, 1.0, 2.0, 5.0],
        severity_label="Amplitude (× signal std)",
        apply_fn=lambda x, s, rng: corrupt_line_noise(x, s),
    ),
]


# ---------------------------------------------------------------------------
# Model factories — baselines
# ---------------------------------------------------------------------------

CWT_KWARGS = dict(
    embed_dim=256,
    num_sources=1,
    target_token_rate=100.0,
    num_freqs=9,
    min_freq=0.5,
    max_freq=30.0,
    freq_spacing="log",
    n_cycles=2.5,
)

CNN_KWARGS = dict(
    embed_dim=256,
    num_sources=1,
    target_token_rate=100.0,
    num_filters=12,
    kernel_size=9,
    num_conv_layers=2,
    activation="gelu",
    antialias=True,
)


def make_cwt(seed: int) -> CWTEmbedding:
    torch.manual_seed(seed)
    return CWTEmbedding(**CWT_KWARGS)


def make_cnn(seed: int) -> ResampleCNNEmbedding:
    torch.manual_seed(seed)
    return ResampleCNNEmbedding(**CNN_KWARGS)


# ---------------------------------------------------------------------------
# Helper: extract CWT scalogram then apply custom transform before projection
# ---------------------------------------------------------------------------


class _CWTVariant(torch.nn.Module):
    """Base for CWT variants that modify the scalogram before projection."""

    def __init__(self, cwt_emb: CWTEmbedding):
        super().__init__()
        self.cwt_emb = cwt_emb

    def _get_cwt_output(self, x, input_sampling_rate, input_seq_len):
        target_time_tokens = self.cwt_emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        return self.cwt_emb.cwt(
            x, input_sampling_rate, input_seq_len, target_time_tokens
        )

    def _project(self, mag, phase):
        B, S, F_dim, T = mag.shape
        cwt_normed = torch.stack([mag, phase], dim=2)
        flat = cwt_normed.permute(0, 4, 1, 2, 3).reshape(B, T, S * 2 * F_dim)
        return self.cwt_emb.feature_proj(flat)

    def transform(self, mag, phase):
        raise NotImplementedError

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        cwt_out = self._get_cwt_output(x, input_sampling_rate, input_seq_len)
        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]
        mag, phase = self.transform(mag, phase)
        return self._project(mag, phase)


# ---------------------------------------------------------------------------
# A. Magnitude normalization variants
# ---------------------------------------------------------------------------


class CWTLogMag(_CWTVariant):
    """log(mag) compresses dynamic range; scaling becomes additive."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def transform(self, mag, phase):
        return torch.log(mag + 1e-6), phase


class CWTUnitMag(_CWTVariant):
    """L2-normalize magnitude per token — keeps spectral shape, strips amplitude."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def transform(self, mag, phase):
        B, S, F, T = mag.shape
        flat = mag.reshape(B, -1, T)
        normed = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        return normed.reshape(B, S, F, T), phase


class CWTLogMagZscore(_CWTVariant):
    """log(mag) followed by per-token z-scoring to remove the additive constant
    that breaks log-mag under large scaling."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def transform(self, mag, phase):
        log_mag = torch.log(mag + 1e-6)
        B, S, F, T = log_mag.shape
        flat = log_mag.reshape(B, -1, T)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        normed = (flat - mean) / std
        return normed.reshape(B, S, F, T), phase


# ---------------------------------------------------------------------------
# B. Higher min_freq (2 Hz) — avoids truncated low-freq wavelets
# ---------------------------------------------------------------------------


class CWTMin2Hz(_CWTVariant):
    def __init__(self, seed: int):
        torch.manual_seed(seed)
        emb = CWTEmbedding(
            embed_dim=256,
            num_sources=1,
            target_token_rate=100.0,
            num_freqs=9,
            min_freq=2.0,
            max_freq=30.0,
            freq_spacing="log",
            n_cycles=2.5,
        )
        super().__init__(emb)

    def transform(self, mag, phase):
        return mag, phase


# ---------------------------------------------------------------------------
# C. Per-frequency instance norm on magnitude
# ---------------------------------------------------------------------------


class CWTFreqNorm(_CWTVariant):
    """Normalize magnitude independently per frequency bin across time.
    Each bin becomes relative (z-scored within its own time series)."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def transform(self, mag, phase):
        mean = mag.mean(dim=-1, keepdim=True)
        std = mag.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (mag - mean) / std, phase


# ---------------------------------------------------------------------------
# D. Input high-pass (subtract per-sample running mean)
# ---------------------------------------------------------------------------


class CWTHighpass(_CWTVariant):
    """Subtract a causal running mean (window=50 samples = 100ms at 500Hz)
    from the input signal before the CWT, removing DC and slow drift."""

    def __init__(self, seed: int, window: int = 50):
        super().__init__(make_cwt(seed))
        self.window = window

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        kernel = torch.ones(1, 1, self.window, device=x.device) / self.window
        B, C, T = x.shape
        padded = F.pad(x, (self.window - 1, 0), mode="reflect")
        running_mean = F.conv1d(padded.reshape(B * C, 1, -1), kernel).reshape(
            B, C, T
        )
        x_hp = x - running_mean

        cwt_out = self._get_cwt_output(x_hp, input_sampling_rate, input_seq_len)
        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]
        return self._project(mag, phase)

    def transform(self, mag, phase):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# E. Wavelet envelope tapering (Tukey window on kernel)
# ---------------------------------------------------------------------------


class CWTTaper(_CWTVariant):
    """Apply a Tukey window to wavelet kernels to reduce spectral leakage
    from kernel truncation at the edges."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        import math as _math

        emb = self.cwt_emb
        cwt = emb.cwt
        target_tt = emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )

        B, C, Max_T = x.shape
        F_dim = cwt.freqs_unconstrained.numel()
        device = x.device
        orig_dtype = x.dtype
        if orig_dtype == torch.bfloat16:
            x = x.float()
        dtype = x.dtype

        f = cwt.freqs.to(device=device, dtype=dtype).view(1, F_dim, 1)
        n_c = cwt.n_cycles.to(device=device, dtype=dtype).view(1, F_dim, 1)

        K = Max_T if Max_T % 2 != 0 else Max_T + 1
        t = torch.arange(
            -(K // 2), K // 2 + 1, device=device, dtype=dtype
        ).view(1, 1, K)
        t_sec = t / input_sampling_rate.view(B, 1, 1)

        sigma = n_c / (2 * _math.pi * f)
        envelope = torch.exp((-(t_sec**2) / (2 * sigma**2)).clamp(min=-60.0))

        # Tukey taper: cosine taper on outer 10% of kernel
        alpha = 0.2
        taper = torch.ones(K, device=device, dtype=dtype)
        n_taper = int(alpha * K / 2)
        if n_taper > 0:
            ramp = 0.5 * (
                1
                - torch.cos(
                    torch.linspace(
                        0, _math.pi, n_taper, device=device, dtype=dtype
                    )
                )
            )
            taper[:n_taper] = ramp
            taper[-n_taper:] = ramp.flip(0)
        envelope = envelope * taper.view(1, 1, K)

        real_wavelet = torch.cos(2 * _math.pi * f * t_sec) * envelope
        imag_wavelet = torch.sin(2 * _math.pi * f * t_sec) * envelope
        norm_factor = torch.sum(envelope, dim=-1, keepdim=True) + 1e-8
        real_wavelet = real_wavelet / norm_factor
        imag_wavelet = imag_wavelet / norm_factor

        kernel_real = real_wavelet.flip(-1)
        kernel_imag = imag_wavelet.flip(-1)
        full_len = Max_T + K - 1
        n_fft = 1 << (full_len - 1).bit_length()

        x_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        out_real = torch.fft.irfft(
            x_fft.unsqueeze(2)
            * torch.fft.rfft(kernel_real, n=n_fft, dim=-1).unsqueeze(1),
            n=n_fft,
            dim=-1,
        )[..., K // 2 : K // 2 + Max_T]
        out_imag = torch.fft.irfft(
            x_fft.unsqueeze(2)
            * torch.fft.rfft(kernel_imag, n=n_fft, dim=-1).unsqueeze(1),
            n=n_fft,
            dim=-1,
        )[..., K // 2 : K // 2 + Max_T]

        out_complex = torch.stack([out_real, out_imag], dim=2).view(
            B, C * 2, F_dim, Max_T
        )

        seq_lens = input_seq_len.clamp(min=1, max=Max_T)
        y_coords = torch.linspace(-1.0, 1.0, F_dim, device=device)
        end_x = -1.0 + 2.0 * (seq_lens.float() - 1) / max(1, Max_T - 1)
        steps = torch.linspace(0, 1, target_tt, device=device).unsqueeze(0)
        x_coords = -1.0 + steps * (end_x.unsqueeze(1) - (-1.0))
        grid = torch.stack(
            [
                x_coords.unsqueeze(1).expand(B, F_dim, target_tt),
                y_coords.view(1, F_dim, 1).expand(B, F_dim, target_tt),
            ],
            dim=-1,
        )

        cc = F.grid_sample(
            out_complex,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        cc = cc.view(B, C, 2, F_dim, target_tt)
        cont_real, cont_imag = cc[:, :, 0], cc[:, :, 1]

        mag_sq = cont_real.square() + cont_imag.square()
        mag = torch.sqrt(mag_sq + 1e-8)
        raw_phase = torch.atan2(cont_imag, cont_real) / _math.pi
        phase_w = mag_sq / (mag_sq + cwt.phase_stability_eps)
        phase = raw_phase * phase_w

        return self._project(mag, phase)

    def transform(self, mag, phase):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# F. Instantaneous frequency instead of phase
# ---------------------------------------------------------------------------


class CWTInstFreq(_CWTVariant):
    """Replace phase with instantaneous frequency (time derivative of phase).
    Smoother and non-circular, avoids phase wrapping issues."""

    def __init__(self, seed: int):
        super().__init__(make_cwt(seed))

    def transform(self, mag, phase):
        inst_freq = torch.diff(phase, dim=-1)
        inst_freq = F.pad(inst_freq, (1, 0), value=0.0)
        return mag, inst_freq


# ---------------------------------------------------------------------------
# G. Separate magnitude and phase projections
# ---------------------------------------------------------------------------


class CWTSepProj(torch.nn.Module):
    """Project magnitude and phase through separate linear layers, then sum.
    Prevents magnitude/phase ratio from affecting output direction."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.cwt_emb = make_cwt(seed)
        num_freqs = self.cwt_emb._num_freqs
        embed_dim = self.cwt_emb.embed_dim

        torch.manual_seed(seed + 10000)
        self.proj_mag = torch.nn.Linear(num_freqs, embed_dim)
        self.proj_phase = torch.nn.Linear(num_freqs, embed_dim)
        torch.nn.init.xavier_uniform_(self.proj_mag.weight)
        torch.nn.init.xavier_uniform_(self.proj_phase.weight)
        torch.nn.init.zeros_(self.proj_mag.bias)
        torch.nn.init.zeros_(self.proj_phase.bias)

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        target_tt = self.cwt_emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt_emb.cwt(
            x, input_sampling_rate, input_seq_len, target_tt
        )

        mag = cwt_out[:, :, 0, :, :]  # (B, S, F, T)
        phase = cwt_out[:, :, 1, :, :]

        B, S, F_dim, T = mag.shape
        mag_flat = mag.permute(0, 3, 1, 2).reshape(B, T, S * F_dim)
        phase_flat = phase.permute(0, 3, 1, 2).reshape(B, T, S * F_dim)

        return self.proj_mag(mag_flat) + self.proj_phase(phase_flat)


# ---------------------------------------------------------------------------
# H. Frequency-scaled n_cycles
# ---------------------------------------------------------------------------


class CWTScaledNCycles(_CWTVariant):
    """n_cycles proportional to frequency: n_cycles = freq * 0.23
    giving ~2.5 at 10 Hz, ~7 at 30 Hz, ~0.5 at 2 Hz.
    Matches wavelet duration to window length at low frequencies."""

    def __init__(self, seed: int):
        torch.manual_seed(seed)
        from foundry.models.embeddings.temporal.cwt import generate_freqs

        freqs = generate_freqs(9, 0.5, 30.0, "log")
        n_cycles_per_freq = [max(1.5, f * 0.23) for f in freqs]
        avg_nc = sum(n_cycles_per_freq) / len(n_cycles_per_freq)
        emb = CWTEmbedding(
            embed_dim=256,
            num_sources=1,
            target_token_rate=100.0,
            init_freqs=freqs,
            n_cycles=avg_nc,
        )
        super().__init__(emb)

        with torch.no_grad():
            from foundry.models.embeddings.temporal.cwt import _inverse_softplus

            target_cycles = torch.tensor(n_cycles_per_freq, dtype=torch.float32)
            emb.cwt.n_cycles_unconstrained.copy_(
                _inverse_softplus(target_cycles - emb.cwt.min_cycles)
            )

    def transform(self, mag, phase):
        return mag, phase


# ---------------------------------------------------------------------------
# Combined: highpass + freq-norm
# ---------------------------------------------------------------------------


class CWTCombined(torch.nn.Module):
    """Highpass input conditioning + per-frequency instance norm on magnitude.

    Combines the three strongest individual improvements:
    - Input highpass removes DC drift before the CWT
    - Per-frequency instance norm strips absolute amplitude per bin
    - Together they should handle amplitude scaling, DC offset, and line noise
    """

    def __init__(self, seed: int, window: int = 50):
        super().__init__()
        self.cwt_emb = make_cwt(seed)
        self.window = window

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        kernel = torch.ones(1, 1, self.window, device=x.device) / self.window
        B, C, T = x.shape
        padded = F.pad(x, (self.window - 1, 0), mode="reflect")
        running_mean = F.conv1d(padded.reshape(B * C, 1, -1), kernel).reshape(
            B, C, T
        )
        x_hp = x - running_mean

        target_tt = self.cwt_emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt_emb.cwt(
            x_hp, input_sampling_rate, input_seq_len, target_tt
        )

        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]

        mean = mag.mean(dim=-1, keepdim=True)
        std = mag.std(dim=-1, keepdim=True).clamp(min=1e-8)
        mag = (mag - mean) / std

        B2, S, F_dim, T2 = mag.shape
        combined = torch.stack([mag, phase], dim=2)
        flat = combined.permute(0, 4, 1, 2, 3).reshape(B2, T2, S * 2 * F_dim)
        return self.cwt_emb.feature_proj(flat)


# ---------------------------------------------------------------------------
# Combined: highpass + log(mag) + freq-norm
# ---------------------------------------------------------------------------


class CWTTriple(torch.nn.Module):
    """Highpass → CWT → log(mag) → per-frequency instance norm.

    Stacks all three strongest individual improvements:
    - Highpass removes DC drift before the CWT
    - Log compresses dynamic range, helping additive noise/spikes
    - Per-frequency instance norm strips residual amplitude scaling
    """

    def __init__(self, seed: int, window: int = 50):
        super().__init__()
        self.cwt_emb = make_cwt(seed)
        self.window = window

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        kernel = torch.ones(1, 1, self.window, device=x.device) / self.window
        B, C, T = x.shape
        padded = F.pad(x, (self.window - 1, 0), mode="reflect")
        running_mean = F.conv1d(padded.reshape(B * C, 1, -1), kernel).reshape(
            B, C, T
        )
        x_hp = x - running_mean

        target_tt = self.cwt_emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt_emb.cwt(
            x_hp, input_sampling_rate, input_seq_len, target_tt
        )

        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]

        mag = torch.log(mag + 1e-6)

        mean = mag.mean(dim=-1, keepdim=True)
        std = mag.std(dim=-1, keepdim=True).clamp(min=1e-8)
        mag = (mag - mean) / std

        B2, S, F_dim, T2 = mag.shape
        combined = torch.stack([mag, phase], dim=2)
        flat = combined.permute(0, 4, 1, 2, 3).reshape(B2, T2, S * 2 * F_dim)
        return self.cwt_emb.feature_proj(flat)


# ---------------------------------------------------------------------------
# Combined: highpass + log(mag) (no freq-norm)
# ---------------------------------------------------------------------------


class CWTHpLog(torch.nn.Module):
    """Highpass → CWT → log(mag).

    Highpass removes DC drift; log compresses dynamic range for additive
    robustness.  No freq-norm so the log-compressed values are preserved,
    keeping the strong additive-corruption resistance of log-mag.
    """

    def __init__(self, seed: int, window: int = 50):
        super().__init__()
        self.cwt_emb = make_cwt(seed)
        self.window = window

    def forward(self, x, *, input_sampling_rate, input_seq_len):
        kernel = torch.ones(1, 1, self.window, device=x.device) / self.window
        B, C, T = x.shape
        padded = F.pad(x, (self.window - 1, 0), mode="reflect")
        running_mean = F.conv1d(padded.reshape(B * C, 1, -1), kernel).reshape(
            B, C, T
        )
        x_hp = x - running_mean

        target_tt = self.cwt_emb._compute_target_tokens(
            input_seq_len, input_sampling_rate
        )
        cwt_out = self.cwt_emb.cwt(
            x_hp, input_sampling_rate, input_seq_len, target_tt
        )

        mag = cwt_out[:, :, 0, :, :]
        phase = cwt_out[:, :, 1, :, :]

        mag = torch.log(mag + 1e-6)

        B2, S, F_dim, T2 = mag.shape
        combined = torch.stack([mag, phase], dim=2)
        flat = combined.permute(0, 4, 1, 2, 3).reshape(B2, T2, S * 2 * F_dim)
        return self.cwt_emb.feature_proj(flat)


# ---------------------------------------------------------------------------
# Factory registry
# ---------------------------------------------------------------------------


VARIANT_FACTORIES = {
    "CWT": make_cwt,
    "CNN": make_cnn,
    "hp+fnorm": lambda s: CWTCombined(s),
    "hp+log+fnorm": lambda s: CWTTriple(s),
    "hp+log": lambda s: CWTHpLog(s),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cosine_sim(clean: torch.Tensor, corrupted: torch.Tensor) -> float:
    """Mean per-token cosine similarity, averaged over batch."""
    sim = F.cosine_similarity(clean, corrupted, dim=-1)
    return sim.mean().item()


def relative_l2(clean: torch.Tensor, corrupted: torch.Tensor) -> float:
    """‖clean - corrupted‖ / ‖clean‖, per token, averaged over batch."""
    diff_norm = torch.norm(clean - corrupted, dim=-1)
    clean_norm = torch.norm(clean, dim=-1).clamp(min=1e-8)
    return (diff_norm / clean_norm).mean().item()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    n_signals: int,
    n_seeds: int,
    device: torch.device,
    architectures: dict | None = None,
) -> dict:
    """Returns results[arch_name][corruption_name] = {severity: {cosine: [...], l2: [...]}}"""
    if architectures is None:
        architectures = VARIANT_FACTORIES

    rng_signals = np.random.default_rng(42)
    signals_np = generate_synthetic_eeg(n_signals, rng_signals)
    signals = torch.from_numpy(signals_np.astype(np.float32)).to(device)
    signals = signals.unsqueeze(1)

    fs = torch.full((n_signals,), SAMPLING_RATE, device=device)
    seq_len = torch.full(
        (n_signals,), NUM_SAMPLES, device=device, dtype=torch.long
    )

    results = {}

    for arch_name, factory in architectures.items():
        print(f"\n  Architecture: {arch_name}")
        results[arch_name] = {}

        for corruption in CORRUPTIONS:
            print(
                f"    Corruption: {corruption.display_name}", end="", flush=True
            )
            results[arch_name][corruption.name] = {}

            for severity in corruption.severity_values:
                cos_vals = []
                l2_vals = []

                for seed in range(n_seeds):
                    model = factory(seed).to(device).eval()

                    with torch.no_grad():
                        clean_repr = model(
                            signals,
                            input_sampling_rate=fs,
                            input_seq_len=seq_len,
                        )

                        rng_corrupt = np.random.default_rng(seed + 1000)
                        corrupted_signal = corruption.apply_fn(
                            signals, severity, rng_corrupt
                        )
                        corrupted_repr = model(
                            corrupted_signal,
                            input_sampling_rate=fs,
                            input_seq_len=seq_len,
                        )

                    cos_vals.append(cosine_sim(clean_repr, corrupted_repr))
                    l2_vals.append(relative_l2(clean_repr, corrupted_repr))

                results[arch_name][corruption.name][severity] = {
                    "cosine": cos_vals,
                    "l2": l2_vals,
                }
                print(".", end="", flush=True)
            print()

    return results


def print_summary(results: dict, metric_key: str = "cosine") -> None:
    """Print a summary table of mean metric values per architecture and corruption."""
    arch_names = list(results.keys())
    header = f"{'Corruption':<25} {'Severity':<12} " + " ".join(
        f"{a:>14}" for a in arch_names
    )
    print(f"\n{'=' * len(header)}")
    print(f"Summary: {metric_key}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for corruption in CORRUPTIONS:
        for severity in corruption.severity_values:
            row = f"{corruption.display_name:<25} {str(severity):<12} "
            for arch_name in arch_names:
                val = np.mean(
                    results[arch_name][corruption.name][severity][metric_key]
                )
                row += f"{val:>14.4f}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metric(
    results: dict,
    metric_key: str,
    metric_label: str,
    output_path: Path,
    ideal_value: float,
    n_signals: int,
    n_seeds: int,
    title_suffix: str = "",
) -> None:
    n_variants = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    arch_names = list(results.keys())

    for idx, corruption in enumerate(CORRUPTIONS):
        ax = axes[idx]
        use_log = corruption.name in LOG_SCALE_CORRUPTIONS

        for arch_name in arch_names:
            sev_data = results[arch_name][corruption.name]
            severities = list(sev_data.keys())
            means = [np.mean(sev_data[s][metric_key]) for s in severities]
            stds = [np.std(sev_data[s][metric_key]) for s in severities]

            if use_log:
                x_vals = [float(s) for s in severities]
            else:
                x_vals = list(range(len(severities)))

            st = STYLE.get(arch_name, {})
            ax.errorbar(
                x_vals,
                means,
                yerr=stds,
                label=arch_name,
                color=st.get("color", "#888888"),
                marker=st.get("marker", "o"),
                linestyle=st.get("ls", "-"),
                markersize=4,
                capsize=2,
                linewidth=1.3,
                alpha=0.85,
            )

        if use_log:
            ax.set_xscale("log")
        else:
            ax.set_xticks(range(len(severities)))
            ax.set_xticklabels([str(s) for s in severities], fontsize=8)

        ax.set_xlabel(corruption.severity_label, fontsize=9)
        ax.set_title(corruption.display_name, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in [0, 3]:
        axes[i].set_ylabel(metric_label, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(4, (n_variants + 1) // 2)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"{metric_label}: CWT Variants vs. CNN Baseline{title_suffix}\n"
        f"({n_signals} signals × {n_seeds} random inits, ideal = {ideal_value})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CWT vs CNN representation stability under signal corruptions"
    )
    parser.add_argument(
        "--n-signals",
        type=int,
        default=50,
        help="Number of synthetic EEG signals (default: 50)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of random weight initializations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/cwt_stability"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Signals: {args.n_signals}, Seeds: {args.n_seeds}")
    print(f"Output: {args.output_dir}")

    print("\n" + "=" * 60)
    print("Running: All CWT variants")
    print("=" * 60)
    results = run_evaluation(
        args.n_signals, args.n_seeds, device, VARIANT_FACTORIES
    )
    print_summary(results, "cosine")

    plot_metric(
        results,
        metric_key="cosine",
        metric_label="Cosine Similarity",
        output_path=args.output_dir / "cosine_similarity.pdf",
        ideal_value=1.0,
        n_signals=args.n_signals,
        n_seeds=args.n_seeds,
    )
    plot_metric(
        results,
        metric_key="l2",
        metric_label="Relative L2 Distance",
        output_path=args.output_dir / "relative_l2.pdf",
        ideal_value=0.0,
        n_signals=args.n_signals,
        n_seeds=args.n_seeds,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
