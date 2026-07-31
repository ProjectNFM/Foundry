"""Experiment 019: Dynamic channel embedding visualization.

Loads saved embeddings from scripts/extract_embeddings.py and produces:
  1. Backbone embedding t-SNE/PCA comparison: disabled vs dynamic (by sleep stage)
  2. Dynamic channel embedding t-SNE/PCA colored by channel name
  3. Dynamic channel embedding t-SNE/PCA colored by session ID
  4. Dynamic channel embedding t-SNE/PCA colored by sleep stage

Usage:
    uv run python analysis/019_dynamic_channel_emb_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from analysis._wandb_utils import figures_dir

EMBEDDING_ROOT = Path("outputs/embeddings")

CONDITIONS = {
    "ch-disabled": "019_disabled",
    "ch-dynamic": "019_dynamic",
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_backbone(name: str) -> dict | None:
    """Load backbone embeddings, labels, and session IDs for a condition."""
    path = EMBEDDING_ROOT / name
    if not path.exists():
        print(f"  [SKIP] {name} — directory not found at {path}")
        return None

    data = {}
    data["embeddings"] = np.load(path / "embeddings.npy")
    data["labels"] = np.load(path / "labels.npy")

    meta_path = path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    sid_path = path / "session_ids.json"
    if sid_path.exists():
        with open(sid_path) as f:
            data["session_ids"] = json.load(f)

    return data


def load_channel_embs(name: str) -> dict | None:
    """Load dynamic channel embeddings and metadata."""
    path = EMBEDDING_ROOT / name
    ch_emb_path = path / "channel_embs.npy"
    if not ch_emb_path.exists():
        print(f"  [SKIP] {name} — no channel_embs.npy found")
        return None

    data = {}
    data["channel_embs"] = np.load(ch_emb_path)
    data["channel_labels"] = np.load(path / "channel_labels.npy")

    with open(path / "channel_names.json") as f:
        data["channel_names"] = json.load(f)
    with open(path / "channel_session_ids.json") as f:
        data["channel_session_ids"] = json.load(f)

    return data


def _short_channel_name(full_name: str) -> str:
    """Extract short channel name from session-scoped ID."""
    if "/" in full_name:
        return full_name.split("/", 1)[1]
    return full_name


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------


MAX_SAMPLES_TSNE = 10000


def _subsample(arrays: list[np.ndarray], labels_or_lists: list, n: int, rng):
    """Subsample arrays and parallel lists to at most n items."""
    total = len(arrays[0])
    if total <= n:
        return arrays, labels_or_lists, np.arange(total)
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()
    out_arrays = [a[idx] for a in arrays]
    out_labels = []
    for ll in labels_or_lists:
        if isinstance(ll, np.ndarray):
            out_labels.append(ll[idx])
        elif isinstance(ll, list):
            out_labels.append([ll[i] for i in idx])
        else:
            out_labels.append(ll)
    return out_arrays, out_labels, idx


def compute_dr(
    embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray | None]:
    """Compute PCA and t-SNE projections.

    If max_samples is set and embeddings exceed that count, subsample before
    t-SNE (PCA is always computed on full data, t-SNE uses subsampled PCA).

    Returns (tsne_2d, pca_2d, pca_model, subsample_idx_or_None).
    """
    n_components = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
    pca_high = PCA(n_components=n_components)
    emb_pca = pca_high.fit_transform(embeddings)

    pca_2 = PCA(n_components=2)
    emb_pca2 = pca_2.fit_transform(embeddings)

    if max_samples is None:
        max_samples = MAX_SAMPLES_TSNE

    sub_idx = None
    tsne_input = emb_pca
    if len(emb_pca) > max_samples:
        rng = np.random.RandomState(random_state)
        sub_idx = rng.choice(len(emb_pca), size=max_samples, replace=False)
        sub_idx.sort()
        tsne_input = emb_pca[sub_idx]

    effective_perplexity = min(perplexity, len(tsne_input) // 4)
    effective_perplexity = max(effective_perplexity, 5)

    print(
        f"  Computing t-SNE on {len(tsne_input)} samples "
        f"(perplexity={effective_perplexity})..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=1000,
        random_state=random_state,
        init="pca",
    )
    emb_tsne = tsne.fit_transform(tsne_input)

    return emb_tsne, emb_pca2, pca_high, sub_idx


# ---------------------------------------------------------------------------
# Backbone DR cache: compute once, reuse for stage + session plots
# ---------------------------------------------------------------------------


def compute_backbone_dr() -> dict[str, dict]:
    """Load data and compute DR for each backbone condition. Returns cache."""
    cache = {}
    for cond_label, dir_name in CONDITIONS.items():
        data = load_backbone(dir_name)
        if data is None:
            continue
        embeddings = data["embeddings"]
        labels = data["labels"]
        session_ids = data.get("session_ids", [])
        print(f"  Computing DR for {cond_label} ({len(embeddings)} samples)...")
        tsne_2d, pca_2d, pca_model, sub_idx = compute_dr(embeddings)
        sil = silhouette_score(
            embeddings[:5000] if len(embeddings) > 5000 else embeddings,
            labels[:5000] if len(labels) > 5000 else labels,
        )
        cache[cond_label] = {
            "embeddings": embeddings,
            "labels": labels,
            "session_ids": session_ids,
            "tsne_2d": tsne_2d,
            "pca_2d": pca_2d,
            "sub_idx": sub_idx,
            "silhouette": sil,
        }
    return cache


# ---------------------------------------------------------------------------
# Plot 1a: Backbone embedding comparison by sleep stage
# ---------------------------------------------------------------------------


def plot_backbone_comparison(backbone_cache: dict, out_dir: Path):
    """Side-by-side t-SNE and PCA of backbone embeddings colored by sleep stage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASS_NAMES)))

    for col, (cond_label, dir_name) in enumerate(CONDITIONS.items()):
        entry = backbone_cache.get(cond_label)
        if entry is None:
            for row in range(2):
                axes[row, col].set_title(f"{cond_label}\n(not available)")
                axes[row, col].axis("off")
            continue

        labels = entry["labels"]
        sub_idx = entry["sub_idx"]
        sil = entry["silhouette"]

        for row, (proj, proj_name) in enumerate(
            [(entry["tsne_2d"], "t-SNE"), (entry["pca_2d"], "PCA")]
        ):
            ax = axes[row, col]
            plot_labels = (
                labels[sub_idx]
                if (proj_name == "t-SNE" and sub_idx is not None)
                else labels
            )
            for i, name in enumerate(CLASS_NAMES):
                mask = plot_labels == i
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    c=[colors[i]],
                    label=name,
                    alpha=0.4,
                    s=8,
                )
            subtitle = f"{cond_label} — {proj_name}"
            if proj_name == "t-SNE":
                subtitle += f" (silhouette={sil:.3f})"
            ax.set_title(subtitle, fontsize=11)
            ax.legend(markerscale=3, fontsize=8, loc="upper right")
            ax.set_xlabel(f"{proj_name} 1")
            ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Backbone Embeddings — Disabled vs Dynamic (by Sleep Stage)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "019_backbone_comparison.png", dpi=150)
    print(f"Saved: {out_dir / '019_backbone_comparison.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1b: Backbone embedding comparison by session
# ---------------------------------------------------------------------------


def plot_backbone_by_session(backbone_cache: dict, out_dir: Path):
    """Side-by-side t-SNE and PCA of backbone embeddings colored by session."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for col, (cond_label, dir_name) in enumerate(CONDITIONS.items()):
        entry = backbone_cache.get(cond_label)
        if entry is None:
            for row in range(2):
                axes[row, col].set_title(f"{cond_label}\n(not available)")
                axes[row, col].axis("off")
            continue

        session_ids = entry["session_ids"]
        sub_idx = entry["sub_idx"]

        if not session_ids:
            for row in range(2):
                axes[row, col].set_title(f"{cond_label}\n(no session IDs)")
                axes[row, col].axis("off")
            continue

        unique_sessions = sorted(set(session_ids))
        n_sess = len(unique_sessions)
        cmap = plt.cm.get_cmap("tab20", min(n_sess, 20))

        for row, (proj, proj_name) in enumerate(
            [(entry["tsne_2d"], "t-SNE"), (entry["pca_2d"], "PCA")]
        ):
            ax = axes[row, col]
            if proj_name == "t-SNE" and sub_idx is not None:
                plot_sids = [session_ids[i] for i in sub_idx]
            else:
                plot_sids = session_ids

            for i, sid in enumerate(unique_sessions):
                mask = np.array([s == sid for s in plot_sids])
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    c=[cmap(i % 20)],
                    label=sid if n_sess <= 20 else None,
                    alpha=0.4,
                    s=8,
                )
            subtitle = f"{cond_label} — {proj_name} (n_sess={n_sess})"
            ax.set_title(subtitle, fontsize=11)
            if n_sess <= 20:
                ax.legend(markerscale=2, fontsize=5, loc="upper right", ncol=2)
            ax.set_xlabel(f"{proj_name} 1")
            ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Backbone Embeddings — Disabled vs Dynamic (by Session)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "019_backbone_by_session.png", dpi=150)
    print(f"Saved: {out_dir / '019_backbone_by_session.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Channel embeddings colored by channel name
# ---------------------------------------------------------------------------


def plot_channel_emb_by_channel(
    ch_data: dict,
    out_dir: Path,
    tsne_2d: np.ndarray,
    pca_2d: np.ndarray,
    sub_idx: np.ndarray | None,
):
    """t-SNE and PCA of dynamic channel embeddings colored by channel type."""
    raw_names = ch_data["channel_names"]
    short_names = [_short_channel_name(n) for n in raw_names]
    unique_channels = sorted(set(short_names))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_channels), 2)))

    for ax, (proj, proj_name) in zip(
        axes, [(tsne_2d, "t-SNE"), (pca_2d, "PCA")]
    ):
        if proj_name == "t-SNE" and sub_idx is not None:
            plot_names = [short_names[i] for i in sub_idx]
        else:
            plot_names = short_names
        for i, ch_name in enumerate(unique_channels):
            mask = np.array([s == ch_name for s in plot_names])
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[colors[i]],
                label=ch_name,
                alpha=0.5,
                s=12,
            )
        ax.legend(markerscale=2, fontsize=8, loc="upper right")
        ax.set_title(f"{proj_name} — colored by channel", fontsize=11)
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Dynamic Channel Embeddings — by Channel Type",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "019_channel_emb_by_channel.png", dpi=150)
    print(f"Saved: {out_dir / '019_channel_emb_by_channel.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Channel embeddings colored by session
# ---------------------------------------------------------------------------


def plot_channel_emb_by_session(
    ch_data: dict,
    out_dir: Path,
    tsne_2d: np.ndarray,
    pca_2d: np.ndarray,
    sub_idx: np.ndarray | None,
):
    """t-SNE and PCA of dynamic channel embeddings colored by session."""
    session_ids = ch_data["channel_session_ids"]
    unique_sessions = sorted(set(session_ids))

    n_sess = len(unique_sessions)
    cmap = plt.cm.get_cmap("tab20", min(n_sess, 20))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (proj, proj_name) in zip(
        axes, [(tsne_2d, "t-SNE"), (pca_2d, "PCA")]
    ):
        if proj_name == "t-SNE" and sub_idx is not None:
            plot_sids = [session_ids[i] for i in sub_idx]
        else:
            plot_sids = session_ids
        for i, sid in enumerate(unique_sessions):
            mask = np.array([s == sid for s in plot_sids])
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[cmap(i % 20)],
                label=sid if n_sess <= 20 else None,
                alpha=0.5,
                s=12,
            )
        if n_sess <= 20:
            ax.legend(markerscale=2, fontsize=6, loc="upper right", ncol=2)
        ax.set_title(
            f"{proj_name} — colored by session (n={n_sess})", fontsize=11
        )
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Dynamic Channel Embeddings — by Session",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "019_channel_emb_by_session.png", dpi=150)
    print(f"Saved: {out_dir / '019_channel_emb_by_session.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Channel embeddings colored by sleep stage
# ---------------------------------------------------------------------------


def plot_channel_emb_by_stage(
    ch_data: dict,
    out_dir: Path,
    tsne_2d: np.ndarray,
    pca_2d: np.ndarray,
    sub_idx: np.ndarray | None,
):
    """t-SNE and PCA of dynamic channel embeddings colored by sleep stage."""
    labels = ch_data["channel_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, len(CLASS_NAMES)))

    for ax, (proj, proj_name) in zip(
        axes, [(tsne_2d, "t-SNE"), (pca_2d, "PCA")]
    ):
        plot_labels = (
            labels[sub_idx]
            if (proj_name == "t-SNE" and sub_idx is not None)
            else labels
        )
        for i, name in enumerate(CLASS_NAMES):
            mask = plot_labels == i
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[colors[i]],
                label=name,
                alpha=0.4,
                s=10,
            )
        ax.legend(markerscale=3, fontsize=8, loc="upper right")
        ax.set_title(f"{proj_name} — colored by sleep stage", fontsize=11)
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Dynamic Channel Embeddings — by Sleep Stage",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "019_channel_emb_by_stage.png", dpi=150)
    print(f"Saved: {out_dir / '019_channel_emb_by_stage.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined channel embedding figure (channel + session + stage in one)
# ---------------------------------------------------------------------------


def plot_channel_emb_combined(
    ch_data: dict,
    out_dir: Path,
    tsne_2d: np.ndarray,
    pca_2d: np.ndarray,
    sub_idx: np.ndarray | None,
):
    """Combined 2x3 figure with t-SNE and PCA rows, colored by channel/session/stage."""
    raw_names = ch_data["channel_names"]
    short_names = [_short_channel_name(n) for n in raw_names]
    unique_channels = sorted(set(short_names))
    session_ids = ch_data["channel_session_ids"]
    unique_sessions = sorted(set(session_ids))
    labels = ch_data["channel_labels"]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    ch_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_channels), 2)))
    n_sess = len(unique_sessions)
    sess_cmap = plt.cm.get_cmap("tab20", min(n_sess, 20))
    stage_colors = plt.cm.Set1(np.linspace(0, 1, len(CLASS_NAMES)))

    for row, (proj, proj_name) in enumerate(
        [(tsne_2d, "t-SNE"), (pca_2d, "PCA")]
    ):
        if proj_name == "t-SNE" and sub_idx is not None:
            p_short_names = [short_names[i] for i in sub_idx]
            p_session_ids = [session_ids[i] for i in sub_idx]
            p_labels = labels[sub_idx]
        else:
            p_short_names = short_names
            p_session_ids = session_ids
            p_labels = labels

        # Column 0: by channel
        ax = axes[row, 0]
        for i, ch_name in enumerate(unique_channels):
            mask = np.array([s == ch_name for s in p_short_names])
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[ch_colors[i]],
                label=ch_name,
                alpha=0.5,
                s=10,
            )
        ax.legend(markerscale=2, fontsize=7, loc="upper right")
        ax.set_title(f"{proj_name} — by channel", fontsize=10)
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

        # Column 1: by session
        ax = axes[row, 1]
        for i, sid in enumerate(unique_sessions):
            mask = np.array([s == sid for s in p_session_ids])
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[sess_cmap(i % 20)],
                label=sid if n_sess <= 15 else None,
                alpha=0.5,
                s=10,
            )
        if n_sess <= 15:
            ax.legend(markerscale=2, fontsize=5, loc="upper right", ncol=2)
        ax.set_title(f"{proj_name} — by session (n={n_sess})", fontsize=10)
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

        # Column 2: by stage
        ax = axes[row, 2]
        for i, name in enumerate(CLASS_NAMES):
            mask = p_labels == i
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[stage_colors[i]],
                label=name,
                alpha=0.4,
                s=10,
            )
        ax.legend(markerscale=3, fontsize=8, loc="upper right")
        ax.set_title(f"{proj_name} — by sleep stage", fontsize=10)
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")

    fig.suptitle(
        "Exp 019: Dynamic Channel Embeddings — Combined View",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "019_channel_emb_combined.png", dpi=150)
    print(f"Saved: {out_dir / '019_channel_emb_combined.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(out_dir: Path):
    """Print silhouette scores and sample counts."""
    print("\n" + "=" * 65)
    print("Experiment 019: Dynamic Channel Embedding Analysis Summary")
    print("=" * 65)

    for cond_label, dir_name in CONDITIONS.items():
        data = load_backbone(dir_name)
        if data is None:
            print(f"  {cond_label}: not available")
            continue
        embs = data["embeddings"]
        labels = data["labels"]
        sil = silhouette_score(
            embs[:5000] if len(embs) > 5000 else embs,
            labels[:5000] if len(labels) > 5000 else labels,
        )
        print(
            f"  {cond_label}: n={len(embs)}, embed_dim={embs.shape[1]}, "
            f"silhouette={sil:.4f}"
        )

    # Channel embedding stats
    ch_data = load_channel_embs(CONDITIONS["ch-dynamic"])
    if ch_data is not None:
        ch_embs = ch_data["channel_embs"]
        short_names = [_short_channel_name(n) for n in ch_data["channel_names"]]
        print(
            f"\n  Channel embs: n={len(ch_embs)}, dim={ch_embs.shape[1]}, "
            f"channels={sorted(set(short_names))}"
        )

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    out_dir = figures_dir(__file__)
    print("Experiment 019: Dynamic Channel Embedding Analysis")
    print(f"Loading from: {EMBEDDING_ROOT.resolve()}")
    print(f"Saving to: {out_dir.resolve()}")
    print()

    print_summary(out_dir)

    print("\n--- Backbone embeddings ---")
    backbone_cache = compute_backbone_dr()
    plot_backbone_comparison(backbone_cache, out_dir)
    plot_backbone_by_session(backbone_cache, out_dir)

    print("\n--- Channel embeddings ---")
    ch_data = load_channel_embs(CONDITIONS["ch-dynamic"])
    if ch_data is not None:
        embs = ch_data["channel_embs"]
        print(f"  Channel emb DR ({len(embs)} embeddings)...")
        tsne_2d, pca_2d, _, sub_idx = compute_dr(embs)

        plot_channel_emb_by_channel(ch_data, out_dir, tsne_2d, pca_2d, sub_idx)
        plot_channel_emb_by_session(ch_data, out_dir, tsne_2d, pca_2d, sub_idx)
        plot_channel_emb_by_stage(ch_data, out_dir, tsne_2d, pca_2d, sub_idx)
        plot_channel_emb_combined(ch_data, out_dir, tsne_2d, pca_2d, sub_idx)
    else:
        print("\nNo channel embeddings found — skipping channel plots.")

    print("\nDone! Check figures/ for output plots.")


if __name__ == "__main__":
    main()
