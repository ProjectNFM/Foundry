"""Generate figures showing learned CWT parameters (frequencies and n_cycles).

Fetches training history from W&B for CWT runs in CWT_VS_CNN_BEHAVIOR and
CWT_VS_CNN_POSE groups, then produces figures comparing initial vs final
learned frequencies and n_cycles.

Outputs (in docs/figures/):
    cwt_learned_freqs.pdf          — Init vs final learned frequencies
    cwt_learned_freqs_training.pdf — Frequency evolution during training

Usage:
    uv run scripts/analysis/cwt_learned_params.py
    uv run scripts/analysis/cwt_learned_params.py --plot-only
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PALETTE_TEMPORAL  # noqa: E402

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
CACHE_DIR = Path("docs/figures/data")

COL_CWT = PALETTE_TEMPORAL["cwt"]
COL_CNN = PALETTE_TEMPORAL["cnn"]

WANDB_ENTITY = "poyo-eeg"
WANDB_PROJECT = "foundry"
NUM_FREQS = 9
INIT_FREQS = np.exp(np.linspace(np.log(0.5), np.log(30.0), NUM_FREQS))
INIT_CYCLES = 2.5

FREQ_KEYS = [
    f"params/tokenizer.temporal_embedding.cwt.freqs_hz/{i}"
    for i in range(NUM_FREQS)
]
CYCLE_KEYS = [
    f"params/tokenizer.temporal_embedding.cwt.n_cycles/{i}"
    for i in range(NUM_FREQS)
]


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def fetch_cwt_params(plot_only: bool = False) -> dict:
    """Fetch CWT parameter data from W&B or cache."""
    cache_path = CACHE_DIR / "cwt_learned_params.json"

    if plot_only and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import wandb

    api = wandb.Api(timeout=30)
    data = {}

    for group in ["CWT_VS_CNN_BEHAVIOR", "CWT_VS_CNN_POSE"]:
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": group}
        )
        for run in runs:
            if "cwt" not in run.display_name:
                continue
            if "dim512" in run.display_name:
                continue

            summary = run.summary._json_dict
            final_freqs = [summary.get(k) for k in FREQ_KEYS]
            final_cycles = [summary.get(k) for k in CYCLE_KEYS]

            if final_freqs[0] is None:
                continue

            hist = run.history(
                keys=FREQ_KEYS + CYCLE_KEYS + ["epoch"], samples=500
            )
            hist_data = None
            if not hist.empty:
                hist_data = {
                    "epochs": hist["epoch"].dropna().tolist(),
                    "freqs": {
                        str(i): hist[FREQ_KEYS[i]].dropna().tolist()
                        for i in range(NUM_FREQS)
                    },
                    "cycles": {
                        str(i): hist[CYCLE_KEYS[i]].dropna().tolist()
                        for i in range(NUM_FREQS)
                    },
                }

            task = "behavior" if "BEHAVIOR" in group else "pose"
            key = f"{task}_{run.display_name}"
            data[key] = {
                "group": group,
                "task": task,
                "name": run.display_name,
                "final_freqs": final_freqs,
                "final_cycles": final_cycles,
                "history": hist_data,
            }

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached CWT params to {cache_path}")

    return data


def plot_learned_freqs(data: dict) -> None:
    """Init vs final learned frequencies, grouped by task."""
    behavior_runs = {k: v for k, v in data.items() if v["task"] == "behavior"}
    pose_runs = {k: v for k, v in data.items() if v["task"] == "pose"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, runs, title, color in [
        (axes[0], behavior_runs, "Behavior Classification", COL_CWT),
        (axes[1], pose_runs, "Pose Estimation", COL_CNN),
    ]:
        freq_indices = np.arange(NUM_FREQS)

        ax.plot(
            freq_indices,
            INIT_FREQS,
            "D-",
            color="#999",
            markersize=7,
            linewidth=1.5,
            label="Initialization",
            zorder=3,
        )

        for i, (name, run_data) in enumerate(runs.items()):
            final = run_data["final_freqs"]
            fold = "fold 0" if "fold0" in name else "fold 1"
            ax.plot(
                freq_indices,
                final,
                "o-",
                color=color,
                alpha=0.6 + 0.2 * i,
                markersize=5,
                linewidth=1.2,
                label=f"Learned ({fold})",
                zorder=2,
            )

        ax.set_xticks(freq_indices)
        ax.set_xticklabels([f"{i}" for i in range(NUM_FREQS)], fontsize=8)
        ax.set_xlabel("Frequency bin index")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    axes[0].set_ylabel("Frequency (Hz)")
    fig.suptitle(
        "Learned CWT Center Frequencies vs Initialization", fontsize=12, y=1.02
    )
    fig.tight_layout()
    _save(fig, "cwt_learned_freqs.pdf")


def plot_freq_training_curves(data: dict) -> None:
    """Show how frequencies evolve over training epochs."""
    behavior_runs = {k: v for k, v in data.items() if v["task"] == "behavior"}

    run_key = next(iter(behavior_runs), None)
    if run_key is None or behavior_runs[run_key]["history"] is None:
        print("  Skipping training curves: no history data")
        return

    hist = behavior_runs[run_key]["history"]
    epochs = hist["epochs"]

    freq_colors = plt.cm.viridis(np.linspace(0.1, 0.9, NUM_FREQS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Frequency evolution
    ax = axes[0]
    for i in range(NUM_FREQS):
        freqs = hist["freqs"][str(i)]
        n = min(len(epochs), len(freqs))
        ax.plot(
            epochs[:n],
            freqs[:n],
            color=freq_colors[i],
            linewidth=1.2,
            label=f"f{i} (init={INIT_FREQS[i]:.1f} Hz)",
        )
        ax.axhline(
            INIT_FREQS[i], color=freq_colors[i], linestyle=":", alpha=0.3
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Center Frequencies During Training")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: n_cycles evolution
    ax = axes[1]
    for i in range(NUM_FREQS):
        cycles = hist["cycles"][str(i)]
        n = min(len(epochs), len(cycles))
        ax.plot(
            epochs[:n],
            cycles[:n],
            color=freq_colors[i],
            linewidth=1.2,
            label=f"f{i}",
        )

    ax.axhline(
        INIT_CYCLES, color="#999", linestyle=":", alpha=0.5, label="Init (2.5)"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("n_cycles")
    ax.set_title("Cycle Counts During Training")
    ax.legend(fontsize=6, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "CWT Parameter Evolution During Training (Behavior, fold 0)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "cwt_learned_freqs_training.pdf")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating CWT learned parameter figures...")
    data = fetch_cwt_params(plot_only=args.plot_only)
    plot_learned_freqs(data)
    plot_freq_training_curves(data)
    print("Done!")


if __name__ == "__main__":
    main()
