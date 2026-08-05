"""Cross-dataset summary of downstream from-scratch baselines.

Produces a grouped bar chart comparing EEGNet, POYO CWT-CNN, and POYO
ResampleCNN across three downstream EEG datasets (Kemp Sleep, PhysioNet MI,
Brain Invaders P300). All values are 3-fold intersubject mean F1 with
error bars showing ±1 std.

Sources:
  - Kemp Sleep: exp 023 (30s epochs, full dataset)
  - PhysioNet MI: final baselines (exps 20260804-*-final-baselines)
  - Brain Invaders P300: 3-fold reprocessed (exp 20260804-*-reprocessed-3fold)

Usage:
    uv run python analysis/034_downstream_baselines_summary.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Best channel embedding mode per model per dataset (selected for highest mean F1).
# Format: (mean_f1, std_f1)
RESULTS = {
    "Kemp Sleep\n(5-class staging)": {
        "EEGNet": (0.692, 0.024),
        "POYO CWT-CNN": (0.730, 0.004),  # dynamic
        "POYO ResampleCNN": (0.699, 0.013),  # dynamic
    },
    "PhysioNet MI\n(L/R binary)": {
        "EEGNet": (0.887, 0.027),
        "POYO CWT-CNN": (0.884, 0.033),  # disabled
        "POYO ResampleCNN": (0.880, 0.037),  # disabled
    },
    "Brain Invaders P300\n(Target/NonTarget)": {
        "EEGNet": (0.386, 0.045),
        "POYO CWT-CNN": (0.364, 0.040),  # dynamic
        "POYO ResampleCNN": (0.328, 0.022),  # dynamic
    },
}

MODEL_NAMES = ["EEGNet", "POYO CWT-CNN", "POYO ResampleCNN"]
DATASET_NAMES = list(RESULTS.keys())
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(DATASET_NAMES))
width = 0.22
offsets = [-width, 0, width]

for i, model in enumerate(MODEL_NAMES):
    means = [RESULTS[ds][model][0] for ds in DATASET_NAMES]
    stds = [RESULTS[ds][model][1] for ds in DATASET_NAMES]
    bars = ax.bar(
        x + offsets[i],
        means,
        width,
        yerr=stds,
        capsize=4,
        label=model,
        color=COLORS[i],
        edgecolor="white",
        linewidth=0.5,
        error_kw=dict(lw=1.2),
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

ax.set_ylabel("Macro F1 (3-fold intersubject mean ± std)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(DATASET_NAMES, fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title(
    "Downstream From-Scratch Baselines: EEGNet vs POYO Variants",
    fontsize=12,
    pad=12,
)
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

plt.tight_layout()
out = FIGURES_DIR / "034_downstream_baselines_summary.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
