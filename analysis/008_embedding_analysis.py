"""Experiment 008: Embedding analysis comparison across conditions.

Loads saved embeddings from scripts/extract_embeddings.py runs and produces
side-by-side t-SNE/PCA comparison figures for all 4 conditions:
  - Pretrained CWT-CNN vs Random CWT-CNN
  - Pretrained ResampleCNN vs Random ResampleCNN

Also reports silhouette scores and linear probe results from WandB.

Usage:
    uv run python analysis/008_embedding_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis._wandb_utils import figures_dir

EMBEDDING_ROOT = Path("outputs/embeddings")

CONDITIONS = {
    "Pretrained CWT-CNN": "008_pretrained_cwt_cnn",
    "Random CWT-CNN": "008_random_cwt_cnn",
    "Pretrained ResampleCNN": "008_pretrained_resample_cnn",
    "Random ResampleCNN": "008_random_resample_cnn",
}

CLASS_NAMES_DEFAULT = ["W", "N1", "N2", "N3", "REM"]


def load_condition(name: str) -> dict | None:
    """Load embeddings and metadata for a condition."""
    path = EMBEDDING_ROOT / name
    if not path.exists():
        print(f"  [SKIP] {name} — directory not found")
        return None

    data = {
        "tsne_2d": np.load(path / "tsne_2d.npy"),
        "pca_2d": np.load(path / "pca_2d.npy"),
        "labels": np.load(path / "labels.npy"),
    }
    meta_path = path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)
    return data


def plot_tsne_comparison():
    """Side-by-side t-SNE plots: pretrained vs random for each tokenizer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    pairs = [
        ("Pretrained CWT-CNN", "Random CWT-CNN"),
        ("Pretrained ResampleCNN", "Random ResampleCNN"),
    ]
    colors = plt.cm.Set1(np.linspace(0, 1, 5))

    for row, (pretrained_name, random_name) in enumerate(pairs):
        for col, cond_name in enumerate([pretrained_name, random_name]):
            ax = axes[row, col]
            data = load_condition(CONDITIONS[cond_name])
            if data is None:
                ax.set_title(f"{cond_name}\n(not available)")
                ax.axis("off")
                continue

            tsne = data["tsne_2d"]
            labels = data["labels"]
            class_names = data.get("metadata", {}).get(
                "class_names", CLASS_NAMES_DEFAULT
            )
            sil = data.get("metadata", {}).get("silhouette_score", None)

            for i, name in enumerate(class_names):
                mask = labels == i
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    tsne[mask, 0],
                    tsne[mask, 1],
                    c=[colors[i]],
                    label=name,
                    alpha=0.4,
                    s=6,
                )

            title = cond_name
            if sil is not None:
                title += f"\n(silhouette={sil:.3f})"
            ax.set_title(title, fontsize=11)
            ax.legend(markerscale=3, fontsize=8, loc="upper right")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

    fig.suptitle(
        "Experiment 008: t-SNE of Backbone Embeddings by Sleep Stage",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = figures_dir(__file__) / "008_tsne_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_pca_comparison():
    """Side-by-side PCA plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    pairs = [
        ("Pretrained CWT-CNN", "Random CWT-CNN"),
        ("Pretrained ResampleCNN", "Random ResampleCNN"),
    ]
    colors = plt.cm.Set1(np.linspace(0, 1, 5))

    for row, (pretrained_name, random_name) in enumerate(pairs):
        for col, cond_name in enumerate([pretrained_name, random_name]):
            ax = axes[row, col]
            data = load_condition(CONDITIONS[cond_name])
            if data is None:
                ax.set_title(f"{cond_name}\n(not available)")
                ax.axis("off")
                continue

            pca = data["pca_2d"]
            labels = data["labels"]
            class_names = data.get("metadata", {}).get(
                "class_names", CLASS_NAMES_DEFAULT
            )

            for i, name in enumerate(class_names):
                mask = labels == i
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    pca[mask, 0],
                    pca[mask, 1],
                    c=[colors[i]],
                    label=name,
                    alpha=0.4,
                    s=6,
                )

            ax.set_title(cond_name, fontsize=11)
            ax.legend(markerscale=3, fontsize=8, loc="upper right")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

    fig.suptitle(
        "Experiment 008: PCA of Backbone Embeddings by Sleep Stage",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = figures_dir(__file__) / "008_pca_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_variance_comparison():
    """Cumulative PCA variance curves for all conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    linestyles = ["-", "--", "-.", ":"]

    for i, (cond_name, dir_name) in enumerate(CONDITIONS.items()):
        path = EMBEDDING_ROOT / dir_name / "metadata.json"
        if not path.exists():
            continue
        with open(path) as f:
            meta = json.load(f)
        var_ratio = meta.get("pca_explained_variance_ratio", [])
        if not var_ratio:
            continue
        cumvar = np.cumsum(var_ratio)
        ax.plot(
            range(1, len(cumvar) + 1),
            cumvar,
            label=cond_name,
            linestyle=linestyles[i % 4],
            linewidth=2,
        )

    ax.axhline(0.95, ls="--", color="gray", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Embedding dimensionality: pretrained vs random")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = figures_dir(__file__) / "008_pca_variance_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def print_summary_table():
    """Print a summary table of silhouette scores."""
    print("\n" + "=" * 60)
    print("Experiment 008: Embedding Quality Summary")
    print("=" * 60)
    print(f"{'Condition':<28} {'Silhouette':>10} {'N samples':>10}")
    print("-" * 60)

    for cond_name, dir_name in CONDITIONS.items():
        path = EMBEDDING_ROOT / dir_name / "metadata.json"
        if not path.exists():
            print(f"{cond_name:<28} {'N/A':>10} {'N/A':>10}")
            continue
        with open(path) as f:
            meta = json.load(f)
        sil = meta.get("silhouette_score", float("nan"))
        n = meta.get("n_samples", 0)
        print(f"{cond_name:<28} {sil:>10.4f} {n:>10}")

    print("=" * 60)


def main():
    print("Experiment 008: Embedding Analysis")
    print("Loading conditions from:", EMBEDDING_ROOT.resolve())
    print()

    print_summary_table()
    plot_tsne_comparison()
    plot_pca_comparison()
    plot_variance_comparison()

    print("\nDone! Check figures/ for output plots.")


if __name__ == "__main__":
    main()
