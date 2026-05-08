"""Generate appendix figures illustrating the ResampleCNN embedding.

Standalone script (no W&B needed) that produces publication-quality figures
showing the anti-aliasing filter and resampling pipeline.

Outputs (in docs/figures/):
    appendix_cnn_butterworth.pdf    — Butterworth low-pass filter response
    appendix_cnn_resampling.pdf     — Before/after resampling of a chirp signal
    appendix_cnn_pipeline.pdf       — Block diagram of the full pipeline

Usage:
    uv run scripts/analysis/appendix_cnn_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
BLUE = "#2196F3"
ORANGE = "#FF9800"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
RED = "#E53935"

ANTIALIAS_ORDER = 6


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_butterworth() -> None:
    """Butterworth low-pass filter magnitude response at different downsample ratios."""
    freqs = np.linspace(0, 1, 500)  # normalized to Nyquist
    ratios = [
        (2, "2x downsample", BLUE),
        (4, "4x downsample", ORANGE),
        (8, "8x downsample", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for ratio, label, color in ratios:
        cutoff = 1.0 / ratio
        f_ratio = freqs / max(cutoff, 1e-6)
        magnitude = 1.0 / np.sqrt(1.0 + f_ratio ** (2 * ANTIALIAS_ORDER))
        ax.plot(freqs, magnitude, color=color, linewidth=1.5, label=label)
        ax.axvline(cutoff, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Normalized frequency (fraction of input Nyquist)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"FFT-Based Butterworth Low-Pass (order {ANTIALIAS_ORDER})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "appendix_cnn_butterworth.pdf")


def plot_resampling() -> None:
    """Before/after resampling of a chirp signal."""
    fs = 256.0
    duration = 1.0
    n = int(fs * duration)
    t = np.linspace(0, duration, n)

    f0, f1 = 2.0, 60.0
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))

    target_tokens = [128, 64, 32]

    fig, axes = plt.subplots(
        len(target_tokens) + 1,
        1,
        figsize=(7, 2.0 * (len(target_tokens) + 1)),
        sharex=True,
    )

    axes[0].plot(t, chirp, color="#333", linewidth=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Original signal ({n} samples at {int(fs)} Hz)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    colors = [BLUE, ORANGE, GREEN]
    for i, (tokens, color) in enumerate(zip(target_tokens, colors)):
        t_resampled = np.linspace(0, duration, tokens)
        freq_cutoff = tokens / n
        freqs_norm = np.linspace(0, 1, n // 2 + 1)
        f_ratio = freqs_norm / max(freq_cutoff, 1e-6)
        bw_mask = 1.0 / np.sqrt(1.0 + f_ratio ** (2 * ANTIALIAS_ORDER))

        chirp_fft = np.fft.rfft(chirp)
        filtered = np.fft.irfft(chirp_fft * bw_mask, n=n)
        filtered_resampled = np.interp(t_resampled, t, filtered)

        ax = axes[i + 1]
        ax.plot(t, chirp, color="#ccc", linewidth=0.5, alpha=0.5)
        ax.plot(
            t_resampled,
            filtered_resampled,
            "o-",
            color=color,
            markersize=2,
            linewidth=0.8,
            label=f"Anti-aliased + resampled ({tokens} tokens)",
        )
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Resampled to {tokens} tokens (effective {tokens / duration:.0f} Hz)"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Anti-Aliased Resampling Pipeline", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "appendix_cnn_resampling.pdf")


def plot_pipeline() -> None:
    """Block diagram of the ResampleCNN pipeline."""
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    blocks = [
        (0.5, "Input\n(B, C, T)", "#E3F2FD"),
        (2.3, "Butterworth\nLow-Pass", "#FFF3E0"),
        (4.1, "grid_sample\nResample", "#E8F5E9"),
        (5.9, "Conv1d\nStack", "#F3E5F5"),
        (7.7, "Linear\nProjection", "#FFEBEE"),
    ]
    bw = 1.4
    bh = 1.2

    for x, label, color in blocks:
        rect = plt.Rectangle(
            (x, 0.4),
            bw,
            bh,
            facecolor=color,
            edgecolor="#333",
            linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + bw / 2,
            1.0,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    for i in range(len(blocks) - 1):
        x_start = blocks[i][0] + bw
        x_end = blocks[i + 1][0]
        ax.annotate(
            "",
            xy=(x_end, 1.0),
            xytext=(x_start, 1.0),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
        )

    ax.text(
        9.5,
        1.0,
        "(B, T', D)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )
    ax.annotate(
        "",
        xy=(9.2, 1.0),
        xytext=(blocks[-1][0] + bw, 1.0),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
    )

    fig.tight_layout()
    _save(fig, "appendix_cnn_pipeline.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating ResampleCNN appendix figures...")
    plot_butterworth()
    plot_resampling()
    plot_pipeline()
    print("Done!")


if __name__ == "__main__":
    main()
