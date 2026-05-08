"""Generate appendix figures illustrating the learnable CWT layer.

Standalone script (no W&B needed) that produces publication-quality figures
showing how the ContinuousCWTLayer works, using synthetic signals.

Outputs (in docs/figures/):
    appendix_cwt_wavelets.pdf       — Morlet wavelets at multiple frequencies
    appendix_cwt_scalogram.pdf      — Scalogram of a chirp + burst signal
    appendix_cwt_resampling.pdf     — Variable-length → fixed token count
    appendix_cwt_phase_stability.pdf — Phase weighting mechanism
    appendix_cwt_fs_invariance.pdf  — Same signal at 128/256/512 Hz
    appendix_cwt_freq_response.pdf  — Bandpass filter curves per wavelet
    appendix_cwt_ncycles.pdf        — Effect of n_cycles on bandwidth

Usage:
    uv run scripts/analysis/appendix_cwt_figures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from foundry.models.embeddings.temporal.cwt import ContinuousCWTLayer

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
        "figure.figsize": (7, 4),
    }
)

OUTPUT_DIR = Path("docs/figures")
INIT_FREQS = [2.0, 5.0, 10.0, 15.0, 20.0]
FS = 256.0
DURATION = 3.0

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#D55E00"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, "#56B4E9"]


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def _make_chirp(
    fs: float = FS, duration: float = DURATION
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(fs * duration)
    t = torch.linspace(0, duration, n)
    f0, f1 = 2.0, 20.0
    chirp_phase = 2 * math.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    chirp = torch.sin(chirp_phase)
    burst_mask = ((t >= 1.5) & (t <= 2.0)).float()
    burst = torch.sin(2 * math.pi * 10.0 * t) * burst_mask
    signal = chirp + 0.7 * burst
    return t, signal


def plot_wavelets() -> None:
    """Morlet wavelets at each init frequency."""
    layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=7.0)
    F_dim = len(INIT_FREQS)

    with torch.no_grad():
        fs_val = torch.tensor([FS])
        N = int(FS * DURATION)
        K = N if N % 2 != 0 else N + 1
        t = torch.arange(-(K // 2), K // 2 + 1).float()
        t_sec = t / fs_val

        f = layer.freqs.view(1, F_dim, 1)
        n_c = layer.n_cycles.view(1, F_dim, 1)
        sigma = n_c / (2 * math.pi * f)
        t_sec_exp = t_sec.view(1, 1, K)
        envelope = torch.exp(-(t_sec_exp**2) / (2 * sigma**2))
        real_part = torch.cos(2 * math.pi * f * t_sec_exp) * envelope

    t_plot = t.numpy() / FS
    fig, axes = plt.subplots(F_dim, 1, figsize=(7, 1.8 * F_dim), sharex=True)

    for i in range(F_dim):
        ax = axes[i]
        env_np = envelope[0, i].numpy()
        real_np = real_part[0, i].numpy()
        ax.fill_between(t_plot, -env_np, env_np, alpha=0.15, color=COLORS[i])
        ax.plot(
            t_plot, real_np, color=COLORS[i], linewidth=0.8, label="Real part"
        )
        ax.plot(
            t_plot,
            env_np,
            color=COLORS[i],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            label="Envelope",
        )
        ax.plot(
            t_plot,
            -env_np,
            color=COLORS[i],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )
        ax.set_ylabel(f"{INIT_FREQS[i]} Hz")
        ax.set_xlim(-0.6, 0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Complex Morlet Wavelets at Learnable Center Frequencies", fontsize=12
    )
    fig.tight_layout()
    _save(fig, "appendix_cwt_wavelets.pdf")


def plot_scalogram() -> None:
    """Scalogram of chirp + burst signal."""
    layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=7.0)
    t, signal = _make_chirp()
    N = len(signal)

    with torch.no_grad():
        x = signal.unsqueeze(0).unsqueeze(0)
        fs = torch.tensor([FS])
        seq_lens = torch.tensor([N])
        out = layer(x, fs, seq_lens, target_time_tokens=N)
        mag = out[0, 0, 0].numpy()

    t_np = t.numpy()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 4.5),
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )

    axes[0].plot(t_np, signal.numpy(), color="#333", linewidth=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Input Signal (chirp + 10 Hz burst)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    im = axes[1].imshow(
        mag,
        aspect="auto",
        origin="lower",
        extent=[0, DURATION, 0, len(INIT_FREQS)],
        cmap="inferno",
        interpolation="bilinear",
    )
    axes[1].set_yticks(np.arange(len(INIT_FREQS)) + 0.5)
    axes[1].set_yticklabels([f"{f:.0f} Hz" for f in INIT_FREQS])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Wavelet frequency")
    axes[1].set_title("CWT Magnitude (Scalogram)")
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)

    fig.tight_layout()
    _save(fig, "appendix_cwt_scalogram.pdf")


def plot_resampling() -> None:
    """Same signal at different lengths → same output token count."""
    layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=7.0)
    target_tokens = 64

    _, signal_full = _make_chirp(fs=FS, duration=DURATION)
    N_full = len(signal_full)
    _, signal_half = _make_chirp(fs=FS, duration=DURATION / 2)
    N_half = len(signal_half)

    max_len = N_full
    padded_half = torch.zeros(max_len)
    padded_half[:N_half] = signal_half

    with torch.no_grad():
        x = torch.stack([signal_full, padded_half]).unsqueeze(1)
        fs = torch.tensor([FS, FS])
        seq_lens = torch.tensor([N_full, N_half])
        out = layer(x, fs, seq_lens, target_time_tokens=target_tokens)
        mag_full = out[0, 0, 0].numpy()
        mag_half = out[1, 0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, mag, label, dur in [
        (axes[0], mag_full, f"Full signal ({DURATION:.1f} s)", DURATION),
        (
            axes[1],
            mag_half,
            f"Half signal ({DURATION / 2:.1f} s)",
            DURATION / 2,
        ),
    ]:
        ax.imshow(
            mag,
            aspect="auto",
            origin="lower",
            extent=[0, dur, 0, len(INIT_FREQS)],
            cmap="inferno",
            interpolation="bilinear",
        )
        ax.set_yticks(np.arange(len(INIT_FREQS)) + 0.5)
        ax.set_yticklabels([f"{f:.0f}" for f in INIT_FREQS])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"{label}\n→ {target_tokens} output tokens")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Time Resampling: Variable Input → Fixed Token Count",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "appendix_cwt_resampling.pdf")


def plot_phase_stability() -> None:
    """Phase weighting curve: suppresses phase where magnitude is near zero."""
    mag_sq = np.linspace(0, 0.05, 500)
    eps_values = [1e-4, 1e-3, 1e-2]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for eps, color in zip(eps_values, [BLUE, ORANGE, GREEN]):
        weight = mag_sq / (mag_sq + eps)
        ax.plot(
            mag_sq,
            weight,
            color=color,
            linewidth=1.5,
            label=rf"$\epsilon = {eps}$",
        )

    ax.set_xlabel(r"$|z|^2$ (magnitude squared)")
    ax.set_ylabel("Phase weight")
    ax.set_title("Phase Stability Weighting")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "appendix_cwt_phase_stability.pdf")


def plot_fs_invariance() -> None:
    """Same physical signal at 128, 256, 512 Hz → near-identical output."""
    layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=7.0)
    target_tokens = 64
    sample_rates = [128.0, 256.0, 512.0]

    mags = []
    for fs_hz in sample_rates:
        _, sig = _make_chirp(fs=fs_hz, duration=DURATION)
        n = len(sig)
        with torch.no_grad():
            x = sig.unsqueeze(0).unsqueeze(0)
            fs_t = torch.tensor([fs_hz])
            seq_lens = torch.tensor([n])
            out = layer(x, fs_t, seq_lens, target_time_tokens=target_tokens)
            mags.append(out[0, 0, 0].numpy())

    vmin = min(m.min() for m in mags)
    vmax = max(m.max() for m in mags)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, mag, fs_hz in zip(axes, mags, sample_rates):
        ax.imshow(
            mag,
            aspect="auto",
            origin="lower",
            extent=[0, DURATION, 0, len(INIT_FREQS)],
            cmap="inferno",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Time (s)")
        ax.set_title(f"fs = {int(fs_hz)} Hz\n({int(fs_hz * DURATION)} samples)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(np.arange(len(INIT_FREQS)) + 0.5)
    axes[0].set_yticklabels([f"{f:.0f}" for f in INIT_FREQS])
    axes[0].set_ylabel("Freq (Hz)")

    fig.suptitle(
        "Sampling-Rate Invariance: Same Signal at Different fs",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "appendix_cwt_fs_invariance.pdf")


def plot_freq_response() -> None:
    """Frequency selectivity: sweep pure sines and plot per-wavelet response."""
    layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=7.0)
    sweep_freqs = np.linspace(0.5, 30.0, 120)
    target_tokens = 64
    N = int(FS * DURATION)

    responses = np.zeros((len(sweep_freqs), len(INIT_FREQS)))
    t = torch.linspace(0, DURATION, N)

    for i, freq in enumerate(sweep_freqs):
        sig = torch.sin(2 * math.pi * freq * t)
        with torch.no_grad():
            x = sig.unsqueeze(0).unsqueeze(0)
            fs_t = torch.tensor([FS])
            seq_lens = torch.tensor([N])
            out = layer(x, fs_t, seq_lens, target_time_tokens=target_tokens)
            mag = out[0, 0, 0].numpy()
            responses[i] = mag.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    for j in range(len(INIT_FREQS)):
        resp = responses[:, j] / responses[:, j].max()
        ax.plot(
            sweep_freqs,
            resp,
            color=COLORS[j],
            linewidth=1.5,
            label=f"{INIT_FREQS[j]} Hz wavelet",
        )

    ax.set_xlabel("Input frequency (Hz)")
    ax.set_ylabel("Normalized response")
    ax.set_title("Wavelet Frequency Selectivity (Bandpass Behavior)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "appendix_cwt_freq_response.pdf")


def plot_ncycles_effect() -> None:
    """Effect of n_cycles on bandwidth."""
    sweep_freqs = np.linspace(0.5, 30.0, 120)
    target_tokens = 64
    N = int(FS * DURATION)
    t = torch.linspace(0, DURATION, N)
    ref_freq_idx = 2  # 10 Hz wavelet

    fig, ax = plt.subplots(figsize=(7, 4))
    for nc, color, ls in [
        (3.0, BLUE, "-"),
        (7.0, ORANGE, "-"),
        (15.0, GREEN, "-"),
    ]:
        layer_nc = ContinuousCWTLayer(init_freqs=INIT_FREQS, n_cycles=nc)
        responses = np.zeros(len(sweep_freqs))
        for i, freq in enumerate(sweep_freqs):
            sig = torch.sin(2 * math.pi * freq * t)
            with torch.no_grad():
                x = sig.unsqueeze(0).unsqueeze(0)
                out = layer_nc(
                    x,
                    torch.tensor([FS]),
                    torch.tensor([N]),
                    target_time_tokens=target_tokens,
                )
                responses[i] = out[0, 0, 0, ref_freq_idx].mean().item()

        responses /= responses.max()
        ax.plot(
            sweep_freqs,
            responses,
            color=color,
            linewidth=1.5,
            linestyle=ls,
            label=f"n_cycles = {nc:.0f}",
        )

    ax.axvline(
        INIT_FREQS[ref_freq_idx], color="#999", linestyle=":", linewidth=0.8
    )
    ax.set_xlabel("Input frequency (Hz)")
    ax.set_ylabel("Normalized response (10 Hz wavelet)")
    ax.set_title("Effect of n_cycles on Frequency Selectivity")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "appendix_cwt_ncycles.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating CWT appendix figures...")
    plot_wavelets()
    plot_scalogram()
    plot_resampling()
    plot_phase_stability()
    plot_fs_invariance()
    plot_freq_response()
    plot_ncycles_effect()
    print("Done!")


if __name__ == "__main__":
    main()
