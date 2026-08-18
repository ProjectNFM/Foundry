"""Embedding visualization callback for monitoring representation structure.

Orchestrates deterministic observation selection (Phase 3), cosine-based
metrics (Phase 4), and figure generation for both channel and backbone
representation families.  Every scheduled, non-sanity validation event emits
the complete applicable output set under stable W&B keys.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning import Trainer
from sklearn.decomposition import PCA

from foundry.training.callbacks.embedding_metrics import (
    compute_backbone_silhouettes,
    compute_channel_metrics,
    compute_norm_statistics,
    cosine_distance_matrix,
    format_backbone_silhouettes_for_logging,
    format_channel_metrics_for_logging,
    get_electrode_positions_3d,
    normalize_electrode_name,
    normalize_representations,
)
from foundry.training.callbacks.observation_selector import (
    ObservationIdentity,
    RankObservations,
    SelectionConfig,
    SelectedObservations,
    build_identities_from_metadata,
    gather_and_deduplicate,
    hierarchical_select_windows,
    select_channel_observations,
)
from foundry.training.step_output import extract_step_output

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic PCA utility
# ---------------------------------------------------------------------------


def fit_deterministic_pca(
    vectors: np.ndarray, n_components: int = 2, seed: int = 0
) -> tuple[np.ndarray, PCA]:
    """Fit PCA on L2-normalized vectors with a deterministic solver.

    Returns the transformed 2D coordinates and the fitted PCA object
    (for explained-variance reporting).
    """
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    if n_components < 1:
        return np.zeros((vectors.shape[0], 2)), PCA(n_components=2)
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(vectors)
    if coords.shape[1] < 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    return coords, pca


# ---------------------------------------------------------------------------
# Stable color utilities
# ---------------------------------------------------------------------------

_QUALITATIVE_HEX = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
]


def stable_color_map(names: list[str]) -> dict[str, str]:
    """Map group names to stable hex colors derived from their names.

    The hash-based assignment means a group's color does not change when
    another group is absent from an event or a run.
    """
    return {
        name: _QUALITATIVE_HEX[
            int.from_bytes(
                hashlib.sha256(name.encode("utf-8")).digest()[:8], "little"
            )
            % len(_QUALITATIVE_HEX)
        ]
        for name in set(names)
    }


def labels_to_colors(
    labels: np.ndarray, color_map: dict[str, str]
) -> list[str]:
    """Convert an array of string labels to a list of hex colors."""
    return [color_map.get(str(lb), "#cccccc") for lb in labels]


# ---------------------------------------------------------------------------
# Scalp colorwheel utilities (adapted from old callback)
# ---------------------------------------------------------------------------


def _scalp_hsv_color(
    x: np.ndarray, y: np.ndarray, max_dist: float
) -> np.ndarray:
    """HSV scalp-position color for arrays of x, y coordinates.

    Returns an (N, 3) RGB array.
    """
    from matplotlib.colors import hsv_to_rgb

    angles = np.arctan2(y, x)
    hues = (angles + np.pi) / (2 * np.pi)
    sats = np.clip(np.sqrt(x**2 + y**2) / max(max_dist, 1e-8), 0.15, 1.0)
    hsv = np.stack([hues, sats, np.full_like(hues, 0.85)], axis=-1)
    return hsv_to_rgb(hsv.reshape(-1, 1, 3)).reshape(-1, 3)


def _draw_scalp_colorwheel(ax, electrode_pos_2d: dict, max_dist: float):
    """Circular colorwheel legend matching the HSV scalp encoding."""
    from matplotlib.colors import hsv_to_rgb

    n = 256
    lin = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    H = (T + np.pi) / (2 * np.pi)
    S = np.clip(R, 0.15, 1.0)
    V = np.full_like(H, 0.85)
    rgb_img = hsv_to_rgb(np.stack([H, S, V], axis=-1))

    alpha = np.where(R <= 1.0, 1.0, 0.0)
    rgba = np.concatenate([rgb_img, alpha[..., np.newaxis]], axis=-1)
    ax.imshow(rgba, extent=[-1.3, 1.3, -1.3, 1.3], origin="lower")

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.4)
    ax.plot([-0.1, 0, 0.1], [1.0, 1.12, 1.0], "k-", linewidth=0.8, alpha=0.4)
    ax.text(0, 1.25, "Front", ha="center", va="bottom", fontsize=8)
    ax.text(0, -1.22, "Back", ha="center", va="top", fontsize=8)
    ax.text(-1.25, 0, "L", ha="right", va="center", fontsize=9, weight="bold")
    ax.text(1.25, 0, "R", ha="left", va="center", fontsize=9, weight="bold")

    for _, (x, y) in electrode_pos_2d.items():
        xn = x / max_dist if max_dist > 0 else 0
        yn = y / max_dist if max_dist > 0 else 0
        if xn**2 + yn**2 <= 1.05:
            ax.plot(xn, yn, "k.", markersize=1.5, alpha=0.3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Scalp Position\n(color key)", fontsize=9)


# ---------------------------------------------------------------------------
# Channel figures (Section 9)
# ---------------------------------------------------------------------------


def make_channel_recording_figure(
    coords_2d: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    pca: PCA,
    channel_mode: str,
    max_panels: int,
    event_label: str,
    seed: int,
):
    """Small-multiple PCA by recording, colored by channel identity (Section 9.1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foundry.training.callbacks.observation_selector import stable_key_hash

    unique_recs = sorted(
        set(recording_ids),
        key=lambda r: stable_key_hash(str(r), seed),
    )
    selected_recs = unique_recs[:max_panels]
    n_panels = len(selected_recs)
    if n_panels == 0:
        return None

    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
    )

    for panel_idx, rec_id in enumerate(selected_recs):
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        rec_mask = recording_ids == rec_id
        rec_coords = coords_2d[rec_mask]
        rec_channels = channel_ids[rec_mask]

        unique_ch = sorted(set(rec_channels))
        cmap = stable_color_map(unique_ch)

        for ch in unique_ch:
            ch_mask = rec_channels == ch
            color = cmap[ch]
            parts = str(ch).split("/")
            short_name = parts[-1] if len(parts) > 1 else ch
            ax.scatter(
                rec_coords[ch_mask, 0],
                rec_coords[ch_mask, 1],
                c=color,
                label=short_name,
                alpha=0.5,
                s=12,
            )

        short_rec = str(rec_id)
        if len(short_rec) > 40:
            short_rec = "..." + short_rec[-37:]
        ax.set_title(short_rec, fontsize=8)
        if len(unique_ch) <= 12:
            ax.legend(fontsize=5, markerscale=2, loc="best")

    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols, panel_idx % ncols].axis("off")

    ev = pca.explained_variance_ratio_
    pc1 = ev[0] if len(ev) > 0 else 0.0
    pc2 = ev[1] if len(ev) > 1 else 0.0
    mode_label = "static (1 pt/ch)" if channel_mode == "static" else "dynamic"
    fig.suptitle(
        f"Channel PCA by Recording — {mode_label} [{event_label}]\n"
        f"PC1 {pc1:.1%}, PC2 {pc2:.1%}  |  "
        f"{len(coords_2d)} obs, {n_panels} recordings",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_channel_canonical_figure(
    coords_2d: np.ndarray,
    channel_ids: np.ndarray,
    pca: PCA,
    event_label: str,
):
    """Global channel PCA colored by canonical electrode name (Section 9.2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = np.array(
        [normalize_electrode_name(str(ch)) for ch in channel_ids]
    )
    unique_canonical = sorted(set(canonical))
    cmap = stable_color_map(unique_canonical)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for name in unique_canonical:
        mask = canonical == name
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=cmap[name],
            label=name if len(unique_canonical) <= 30 else None,
            alpha=0.4,
            s=10,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] if len(ev) > 0 else 0.0:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1] if len(ev) > 1 else 0.0:.1%})")
    ax.set_title(
        f"Channel PCA — Canonical Electrode [{event_label}]\n"
        f"{len(coords_2d)} obs, {len(unique_canonical)} electrodes"
    )
    if len(unique_canonical) <= 30:
        ax.legend(fontsize=6, markerscale=2, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def make_channel_anatomy_figure(
    coords_2d: np.ndarray,
    channel_ids: np.ndarray,
    positions_3d: dict[str, np.ndarray],
    pca: PCA,
    event_label: str,
):
    """PCA colored by scalp position for channels with resolved anatomy (Section 9.3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = np.array(
        [normalize_electrode_name(str(ch)) for ch in channel_ids]
    )

    electrode_pos_2d = _get_electrode_positions_2d(positions_3d)

    resolved_mask = np.array([c in electrode_pos_2d for c in canonical])
    n_resolved = int(np.sum(resolved_mask))
    if n_resolved == 0:
        return None

    fig = plt.figure(figsize=(11, 8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    xs = np.zeros(len(canonical))
    ys = np.zeros(len(canonical))
    for i, c in enumerate(canonical):
        if c in electrode_pos_2d:
            xs[i], ys[i] = electrode_pos_2d[c]

    max_d = (
        np.sqrt(xs[resolved_mask] ** 2 + ys[resolved_mask] ** 2).max() or 1.0
    )
    rgb = _scalp_hsv_color(xs, ys, max_d)

    ax.scatter(
        coords_2d[resolved_mask, 0],
        coords_2d[resolved_mask, 1],
        c=rgb[resolved_mask],
        s=30,
        edgecolors="k",
        linewidths=0.3,
        alpha=0.85,
        zorder=3,
    )
    if (~resolved_mask).any():
        ax.scatter(
            coords_2d[~resolved_mask, 0],
            coords_2d[~resolved_mask, 1],
            c="lightgray",
            s=15,
            alpha=0.4,
            label="no position",
            zorder=2,
        )
        ax.legend(markerscale=1.5, fontsize=8)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] if len(ev) > 0 else 0.0:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1] if len(ev) > 1 else 0.0:.1%})")
    ax.set_title(
        f"Channel PCA — Anatomical Position [{event_label}]\n"
        f"{n_resolved} resolved, {int(np.sum(~resolved_mask))} unresolved"
    )

    _draw_scalp_colorwheel(ax_leg, electrode_pos_2d, max_d)
    return fig


def _get_electrode_positions_2d(
    positions_3d: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Project 3D electrode positions to 2D (x, y) by dropping z."""
    return {
        name: (float(pos[0]), float(pos[1]))
        for name, pos in positions_3d.items()
    }


def make_norm_distribution_figure(
    norms: np.ndarray, family: str, event_label: str
):
    """Histogram of raw L2 norms for a representation family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = norms[np.isfinite(norms) & (norms > 0)]
    if len(valid) == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(
        valid, bins=min(50, max(10, len(valid) // 20)), edgecolor="k", alpha=0.7
    )
    stats = compute_norm_statistics(norms)
    ax.axvline(
        stats.get("median", 0),
        color="red",
        linestyle="--",
        label=f"median={stats.get('median', 0):.3f}",
    )
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{family} Norm Distribution [{event_label}]\n"
        f"N={len(valid)}, mean={stats.get('mean', 0):.3f}, std={stats.get('std', 0):.3f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def normalization_counts_for_logging(family: str, result) -> dict[str, int]:
    """Return raw-vector validity counters under the family-specific key."""
    prefix = f"val/embedding_viz/{family}/normalization"
    return {
        f"{prefix}/n_total": result.n_total,
        f"{prefix}/n_valid": result.n_valid,
        f"{prefix}/n_zero": result.n_zero,
        f"{prefix}/n_nonfinite": result.n_nonfinite,
    }


# ---------------------------------------------------------------------------
# Backbone figures (Section 10)
# ---------------------------------------------------------------------------


def make_backbone_pca_figure(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    pca: PCA,
    event_label: str,
    class_names: list[str] | None = None,
):
    """PCA scatter for backbone representations colored by a grouping."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    str_labels = np.array([str(lb) for lb in labels])
    unique_labels = sorted(set(str_labels))
    cmap = stable_color_map(unique_labels)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for lb in unique_labels:
        mask = str_labels == lb
        display_name = lb
        if class_names is not None:
            try:
                idx = int(lb)
                if 0 <= idx < len(class_names):
                    display_name = class_names[idx]
            except (ValueError, TypeError):
                pass
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=cmap[lb],
            label=display_name if len(unique_labels) <= 30 else None,
            alpha=0.4,
            s=10,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] if len(ev) > 0 else 0.0:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1] if len(ev) > 1 else 0.0:.1%})")
    ax.set_title(
        f"Backbone PCA — {label_name} [{event_label}]\n"
        f"{len(coords_2d)} windows, {len(unique_labels)} groups"
    )
    if len(unique_labels) <= 30:
        ax.legend(fontsize=7, markerscale=2, ncol=2, loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class EmbeddingVisualizationCallback(L.Callback):
    """Visualize channel and backbone representation structure during validation.

    Integrates the deterministic observation selector (Phase 3) and cosine-based
    metrics (Phase 4) to produce a complete, reproducible set of figures and
    scalar metrics at every scheduled validation event.

    Args:
        every_n_validation_runs: Emit outputs every N complete, non-sanity
            validation passes.  The first complete pass is event 1.
        sample_seed: Seed for deterministic observation selection.
        window_fraction: Fraction of validation windows to select.
        min_windows: Minimum window budget.
        max_windows: Maximum window budget.
        max_channel_observations: Cap on total (window, channel) vectors.
        max_sessions_per_dataset: Maximum sessions selected per dataset.
        min_windows_per_session: Minimum windows per selected session.
        max_recording_panels: Maximum recording panels in the small-multiple
            channel figure.
        min_positioned_channels: Minimum resolved channel positions for
            anatomical output.
    """

    def __init__(
        self,
        every_n_validation_runs: int = 5,
        sample_seed: int = 42,
        window_fraction: float = 0.10,
        min_windows: int = 256,
        max_windows: int = 2048,
        max_channel_observations: int = 16384,
        max_sessions_per_dataset: int = 8,
        min_windows_per_session: int = 16,
        max_recording_panels: int = 8,
        min_positioned_channels: int = 9,
        # Deprecated parameters (Phase 6 will remove these from configs)
        every_n_epochs: int | None = None,
        max_samples: int | None = None,
        compute_tsne: bool | None = None,
        class_names: list[str] | None = None,
    ):
        super().__init__()

        if every_n_epochs is not None:
            log.warning(
                "EmbeddingVisualizationCallback: 'every_n_epochs' is deprecated, "
                "use 'every_n_validation_runs' instead. Mapping every_n_epochs=%d "
                "to every_n_validation_runs=%d.",
                every_n_epochs,
                every_n_epochs,
            )
            every_n_validation_runs = every_n_epochs
        if max_samples is not None:
            log.warning(
                "EmbeddingVisualizationCallback: 'max_samples' is deprecated, "
                "use 'max_windows' instead."
            )
            max_windows = max_samples
        if compute_tsne is not None:
            log.warning(
                "EmbeddingVisualizationCallback: 'compute_tsne' is deprecated "
                "and ignored. t-SNE has been removed."
            )
        if class_names is not None:
            log.warning(
                "EmbeddingVisualizationCallback: 'class_names' is deprecated. "
                "Class names are now discovered from task configs."
            )

        self.every_n_validation_runs = every_n_validation_runs
        self.min_positioned_channels = min_positioned_channels

        self._selection_config = SelectionConfig(
            seed=sample_seed,
            window_fraction=window_fraction,
            min_windows=min_windows,
            max_windows=max_windows,
            max_channel_observations=max_channel_observations,
            max_sessions_per_dataset=max_sessions_per_dataset,
            min_windows_per_session=min_windows_per_session,
            max_recording_panels=max_recording_panels,
        )

        self._validation_run_count = 0
        self._capture_scheduled = False
        self._idx_to_channel: dict[int, str] = {}
        self._task_class_names: dict[str, list[str]] = {}

        self._rank_obs: RankObservations | None = None
        self._local_identities: list[ObservationIdentity] = []
        self._local_channel_counts: list[int] = []
        self._sample_metadata_lists: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if not getattr(model, "supports_representation_capture", False):
            log.warning(
                "EmbeddingVisualizationCallback: model does not expose the "
                "representation contract."
            )
        else:
            channel_emb = getattr(model, "channel_emb", None)
            vocab = getattr(channel_emb, "vocab", None)
            if vocab is not None:
                self._idx_to_channel = {
                    idx: name for name, idx in vocab.items()
                }

        self._discover_task_class_names(model)

    def _discover_task_class_names(self, model) -> None:
        """Extract class names per task from TaskConfig.class_mapping."""
        if not hasattr(model, "task_configs"):
            return
        for task_name, cfg in model.task_configs.items():
            if hasattr(cfg, "get_class_names"):
                names = cfg.get_class_names()
                if names is not None:
                    self._task_class_names[task_name] = names

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        is_sanity = getattr(trainer, "sanity_checking", False)

        if is_sanity:
            self._capture_scheduled = False
        else:
            self._validation_run_count += 1
            self._capture_scheduled = (
                self.every_n_validation_runs > 0
                and self._validation_run_count % self.every_n_validation_runs
                == 0
            )

        setter = getattr(
            pl_module, "set_validation_representation_capture", None
        )
        if setter is not None:
            setter(self._capture_scheduled)

        if self._capture_scheduled:
            self._clear_buffers()

    # ------------------------------------------------------------------
    # Per-batch accumulation
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._capture_scheduled:
            return

        step_output = extract_step_output(outputs)
        if step_output is None:
            return

        meta = step_output.sample_metadata
        if meta is None or meta.dataset_id is None:
            return

        identities = build_identities_from_metadata(
            meta.dataset_id,
            meta.subject_id or ["__unknown__"] * len(meta.dataset_id),
            meta.session_id or ["__unknown__"] * len(meta.dataset_id),
            meta.absolute_start
            if meta.absolute_start is not None
            else np.zeros(len(meta.dataset_id)),
            meta.window_duration
            if meta.window_duration is not None
            else np.zeros(len(meta.dataset_id)),
        )

        B = len(identities)
        reps = step_output.representations
        backbone = reps.backbone_representations if reps is not None else None
        ch_reps = reps.channel_representations if reps is not None else None
        ch_mask = reps.channel_mask if reps is not None else None

        channel_counts = []
        if (
            ch_reps is not None
            and ch_mask is not None
            and ch_mask.shape == ch_reps.shape[:2]
        ):
            ch_mask_np = ch_mask.cpu().numpy().astype(bool)
            for b in range(B):
                channel_counts.append(int(ch_mask_np[b].sum()))
        elif ch_reps is not None:
            channel_counts = [ch_reps.shape[1]] * B
        else:
            channel_counts = [0] * B

        self._local_identities.extend(identities)
        self._local_channel_counts.extend(channel_counts)

        if self._rank_obs is None:
            self._rank_obs = RankObservations(
                identities=[],
                backbone_representations=None,
                channel_representations=None,
                channel_counts=[],
                target_values={},
            )

        self._rank_obs.identities.extend(identities)
        self._rank_obs.channel_counts.extend(channel_counts)

        for task_name, targets in (step_output.target_values or {}).items():
            t_cpu = targets.detach().cpu()
            if task_name not in self._rank_obs.target_values:
                self._rank_obs.target_values[task_name] = t_cpu
            else:
                self._rank_obs.target_values[task_name] = torch.cat(
                    [self._rank_obs.target_values[task_name], t_cpu], dim=0
                )

        if backbone is not None:
            bb_cpu = backbone.detach().cpu()
            if self._rank_obs.backbone_representations is None:
                self._rank_obs.backbone_representations = bb_cpu
            else:
                self._rank_obs.backbone_representations = torch.cat(
                    [self._rank_obs.backbone_representations, bb_cpu], dim=0
                )

        if ch_reps is not None:
            ch_cpu = ch_reps.detach().cpu()
            if ch_mask is None:
                ch_mask_cpu = torch.ones(B, ch_cpu.shape[1], dtype=torch.bool)
            else:
                ch_mask_cpu = ch_mask.detach().cpu().bool()
                if ch_mask_cpu.shape != ch_cpu.shape[:2]:
                    log.warning(
                        "EmbeddingVisualizationCallback: channel mask shape %s "
                        "does not match channel representations %s; using an "
                        "all-valid mask.",
                        tuple(ch_mask_cpu.shape),
                        tuple(ch_cpu.shape[:2]),
                    )
                    ch_mask_cpu = torch.ones(
                        B, ch_cpu.shape[1], dtype=torch.bool
                    )
            channel_indices = meta.channel_index
            if channel_indices is None:
                ch_idx_cpu = torch.arange(ch_cpu.shape[1]).expand(B, -1)
            else:
                ch_idx_cpu = channel_indices.detach().cpu()
                if ch_idx_cpu.ndim == 1:
                    ch_idx_cpu = ch_idx_cpu.unsqueeze(0).expand(B, -1)
                if ch_idx_cpu.shape != ch_cpu.shape[:2]:
                    log.warning(
                        "EmbeddingVisualizationCallback: channel index shape %s "
                        "does not match channel representations %s; using "
                        "positional channel identifiers.",
                        tuple(ch_idx_cpu.shape),
                        tuple(ch_cpu.shape[:2]),
                    )
                    ch_idx_cpu = torch.arange(ch_cpu.shape[1]).expand(B, -1)
            if self._rank_obs.channel_representations is None:
                self._rank_obs.channel_representations = ch_cpu
                self._rank_obs.channel_indices = ch_idx_cpu
                self._rank_obs.channel_masks = ch_mask_cpu
            else:
                max_c = max(
                    self._rank_obs.channel_representations.shape[1],
                    ch_cpu.shape[1],
                )
                if self._rank_obs.channel_representations.shape[1] < max_c:
                    old = self._rank_obs.channel_representations
                    pad = old.new_zeros(
                        old.shape[0], max_c - old.shape[1], old.shape[2]
                    )
                    self._rank_obs.channel_representations = torch.cat(
                        [old, pad], dim=1
                    )
                if ch_cpu.shape[1] < max_c:
                    pad = ch_cpu.new_zeros(
                        ch_cpu.shape[0],
                        max_c - ch_cpu.shape[1],
                        ch_cpu.shape[2],
                    )
                    ch_cpu = torch.cat([ch_cpu, pad], dim=1)
                old_indices = self._rank_obs.channel_indices
                if old_indices is None:
                    old_indices = torch.zeros(
                        self._rank_obs.channel_representations.shape[0],
                        self._rank_obs.channel_representations.shape[1],
                        dtype=ch_idx_cpu.dtype,
                    )
                if old_indices.shape[1] < max_c:
                    old_indices = torch.nn.functional.pad(
                        old_indices, (0, max_c - old_indices.shape[1])
                    )
                if ch_idx_cpu.shape[1] < max_c:
                    ch_idx_cpu = torch.nn.functional.pad(
                        ch_idx_cpu, (0, max_c - ch_idx_cpu.shape[1])
                    )
                old_masks = self._rank_obs.channel_masks
                if old_masks is None:
                    old_masks = torch.ones(
                        self._rank_obs.channel_representations.shape[0],
                        self._rank_obs.channel_representations.shape[1],
                        dtype=torch.bool,
                    )
                if old_masks.shape[1] < max_c:
                    old_masks = torch.nn.functional.pad(
                        old_masks, (0, max_c - old_masks.shape[1])
                    )
                if ch_mask_cpu.shape[1] < max_c:
                    ch_mask_cpu = torch.nn.functional.pad(
                        ch_mask_cpu, (0, max_c - ch_mask_cpu.shape[1])
                    )
                self._rank_obs.channel_representations = torch.cat(
                    [self._rank_obs.channel_representations, ch_cpu], dim=0
                )
                self._rank_obs.channel_indices = torch.cat(
                    [old_indices, ch_idx_cpu], dim=0
                )
                self._rank_obs.channel_masks = torch.cat(
                    [old_masks, ch_mask_cpu], dim=0
                )

        self._sample_metadata_lists.setdefault("channel_mode", []).append(
            reps.channel_mode if reps is not None else None
        )

    # ------------------------------------------------------------------
    # End-of-validation orchestration
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        setter = getattr(
            pl_module, "set_validation_representation_capture", None
        )
        if setter is not None:
            setter(False)

        scheduled = self._capture_scheduled
        self._capture_scheduled = False

        if (
            not scheduled
            or self._rank_obs is None
            or not self._rank_obs.identities
        ):
            self._clear_buffers()
            return

        world_size = trainer.world_size if hasattr(trainer, "world_size") else 1
        global_rank = (
            trainer.global_rank if hasattr(trainer, "global_rank") else 0
        )

        merged = gather_and_deduplicate(
            self._rank_obs, world_size=world_size, rank=global_rank
        )

        if merged is None:
            self._clear_buffers()
            return

        from foundry.training.callbacks import get_wandb_experiment

        wandb_experiment = get_wandb_experiment(trainer)

        try:
            self._process_and_log(merged, trainer, wandb_experiment)
        except Exception:
            log.exception(
                "EmbeddingVisualizationCallback: error during processing"
            )
        finally:
            self._clear_buffers()

    def _process_and_log(
        self,
        merged: RankObservations,
        trainer: Trainer,
        wandb_experiment,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            import wandb
        except ImportError:
            wandb = None

        config = self._selection_config
        step_label = f"step {trainer.global_step}"

        selection = hierarchical_select_windows(merged.identities, config)

        channel_mode = self._resolve_channel_mode()

        ch_window_indices = []
        if (
            merged.channel_representations is not None
            and channel_mode != "disabled"
        ):
            ch_window_indices = select_channel_observations(
                selection.window_indices,
                merged.identities,
                merged.channel_counts,
                config,
            )
            selection.channel_window_indices = ch_window_indices
            selection.channel_observation_count = sum(
                merged.channel_counts[i] for i in ch_window_indices
            )

        log_dict: dict[str, Any] = {}
        prefix = "val/embedding_viz"

        log_dict[f"{prefix}/sample/window_count"] = selection.window_count
        log_dict[f"{prefix}/sample/channel_observation_count"] = (
            selection.channel_observation_count
        )
        log_dict[f"{prefix}/sample/fingerprint"] = selection.fingerprint

        # ----- Backbone analysis -----
        backbone_available = (
            merged.backbone_representations is not None
            and selection.window_count > 0
        )

        if backbone_available:
            bb_selected = merged.backbone_representations[
                selection.window_indices
            ]
            bb_np = bb_selected.numpy()
            bb_norm_result = normalize_representations(bb_np)
            log_dict.update(
                normalization_counts_for_logging("backbone", bb_norm_result)
            )

            if bb_norm_result.n_valid >= 2:
                bb_coords, bb_pca = fit_deterministic_pca(
                    bb_norm_result.vectors, n_components=2, seed=config.seed
                )
                bb_dist_matrix = cosine_distance_matrix(bb_norm_result.vectors)

                valid_indices = np.where(bb_norm_result.valid_mask)[0]

                dataset_labels = np.array(
                    [
                        merged.identities[
                            selection.window_indices[i]
                        ].dataset_id
                        for i in valid_indices
                    ]
                )
                subject_labels = np.array(
                    [
                        merged.identities[
                            selection.window_indices[i]
                        ].subject_id
                        for i in valid_indices
                    ]
                )
                session_labels = np.array(
                    [
                        merged.identities[
                            selection.window_indices[i]
                        ].session_id
                        for i in valid_indices
                    ]
                )

                groupings: dict[str, np.ndarray] = {
                    "dataset": dataset_labels,
                    "subject": subject_labels,
                    "session": session_labels,
                }

                for view_name, view_labels in [
                    ("dataset", dataset_labels),
                    ("subject", subject_labels),
                    ("session", session_labels),
                ]:
                    fig = make_backbone_pca_figure(
                        bb_coords,
                        view_labels,
                        view_name.capitalize(),
                        bb_pca,
                        step_label,
                    )
                    if fig is not None and wandb is not None:
                        log_dict[f"{prefix}/backbone/pca_{view_name}"] = (
                            wandb.Image(fig)
                        )
                    if fig is not None:
                        plt.close(fig)

                task_target_labels = self._extract_task_labels(
                    merged, selection, valid_indices
                )
                for task_name, (
                    task_labels,
                    task_mask,
                    class_names,
                ) in task_target_labels.items():
                    task_valid = task_mask
                    if task_valid.sum() < 2:
                        continue
                    task_coords = bb_coords[task_valid]
                    task_lbl = task_labels[task_valid]
                    fig = make_backbone_pca_figure(
                        task_coords,
                        task_lbl,
                        task_name,
                        bb_pca,
                        step_label,
                        class_names=class_names,
                    )
                    if fig is not None and wandb is not None:
                        log_dict[f"{prefix}/backbone/pca_task/{task_name}"] = (
                            wandb.Image(fig)
                        )
                    if fig is not None:
                        plt.close(fig)

                    groupings[f"task/{task_name}"] = np.where(
                        task_valid,
                        task_labels,
                        np.array([-1] * len(task_labels)),
                    )

                silhouettes = compute_backbone_silhouettes(
                    bb_dist_matrix, groupings
                )
                log_dict.update(
                    format_backbone_silhouettes_for_logging(silhouettes)
                )

            fig_norm = make_norm_distribution_figure(
                bb_norm_result.norms, "Backbone", step_label
            )
            if fig_norm is not None and wandb is not None:
                log_dict[f"{prefix}/backbone/norm_distribution"] = wandb.Image(
                    fig_norm
                )
            if fig_norm is not None:
                plt.close(fig_norm)

        # ----- Channel analysis -----
        channel_available = (
            merged.channel_representations is not None
            and channel_mode not in ("disabled", None)
            and len(ch_window_indices) > 0
        )

        if channel_available:
            (
                ch_flat_vecs,
                ch_flat_recording_ids,
                ch_flat_channel_ids,
                ch_flat_window_ids,
            ) = self._flatten_channel_observations(merged, ch_window_indices)

            if len(ch_flat_vecs) > 0:
                ch_norm_result = normalize_representations(ch_flat_vecs)
                log_dict.update(
                    normalization_counts_for_logging("channel", ch_norm_result)
                )

                if ch_norm_result.n_valid >= 2:
                    valid_mask = ch_norm_result.valid_mask
                    ch_recording_valid = ch_flat_recording_ids[valid_mask]
                    ch_channel_valid = ch_flat_channel_ids[valid_mask]
                    ch_window_valid = ch_flat_window_ids[valid_mask]

                    ch_coords, ch_pca = fit_deterministic_pca(
                        ch_norm_result.vectors, n_components=2, seed=config.seed
                    )

                    fig_rec = make_channel_recording_figure(
                        ch_coords,
                        ch_recording_valid,
                        ch_channel_valid,
                        ch_pca,
                        channel_mode,
                        config.max_recording_panels,
                        step_label,
                        config.seed,
                    )
                    if fig_rec is not None and wandb is not None:
                        log_dict[f"{prefix}/channel/pca_by_recording"] = (
                            wandb.Image(fig_rec)
                        )
                    if fig_rec is not None:
                        plt.close(fig_rec)

                    fig_canon = make_channel_canonical_figure(
                        ch_coords,
                        ch_channel_valid,
                        ch_pca,
                        step_label,
                    )
                    if fig_canon is not None and wandb is not None:
                        log_dict[
                            f"{prefix}/channel/pca_canonical_electrode"
                        ] = wandb.Image(fig_canon)
                    if fig_canon is not None:
                        plt.close(fig_canon)

                    positions_3d = get_electrode_positions_3d()
                    canonical_labels = np.array(
                        [
                            normalize_electrode_name(str(ch))
                            for ch in ch_channel_valid
                        ]
                    )
                    n_positioned = len(
                        {c for c in canonical_labels if c in positions_3d}
                    )
                    has_positions = n_positioned >= self.min_positioned_channels

                    if has_positions:
                        fig_anat = make_channel_anatomy_figure(
                            ch_coords,
                            ch_channel_valid,
                            positions_3d,
                            ch_pca,
                            step_label,
                        )
                        if fig_anat is not None and wandb is not None:
                            log_dict[f"{prefix}/channel/pca_anatomy"] = (
                                wandb.Image(fig_anat)
                            )
                        if fig_anat is not None:
                            plt.close(fig_anat)

                    channel_metrics = compute_channel_metrics(
                        ch_norm_result.vectors,
                        ch_recording_valid,
                        ch_channel_valid,
                        channel_mode,
                        window_ids=ch_window_valid,
                        positions_3d=positions_3d if has_positions else None,
                        min_positioned=self.min_positioned_channels,
                    )
                    log_dict.update(
                        format_channel_metrics_for_logging(channel_metrics)
                    )

                fig_ch_norm = make_norm_distribution_figure(
                    ch_norm_result.norms, "Channel", step_label
                )
                if fig_ch_norm is not None and wandb is not None:
                    log_dict[f"{prefix}/channel/norm_distribution"] = (
                        wandb.Image(fig_ch_norm)
                    )
                if fig_ch_norm is not None:
                    plt.close(fig_ch_norm)

        # ----- Availability counters -----
        log_dict[f"{prefix}/availability/backbone"] = int(backbone_available)
        log_dict[f"{prefix}/availability/channel"] = int(channel_available)
        log_dict[f"{prefix}/availability/channel_mode"] = channel_mode or "none"

        # ----- Log to W&B -----
        if wandb_experiment is not None and log_dict:
            wandb_experiment.log(log_dict, commit=False)

        log.info(
            "EmbeddingVisualizationCallback: logged %d items at %s "
            "(fingerprint=%s, windows=%d, channel_obs=%d)",
            len(log_dict),
            step_label,
            selection.fingerprint,
            selection.window_count,
            selection.channel_observation_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_channel_mode(self) -> str:
        """Resolve the channel mode from accumulated metadata."""
        modes = self._sample_metadata_lists.get("channel_mode", [])
        modes = [m for m in modes if m is not None]
        if not modes:
            return "disabled"
        return modes[0]

    def _flatten_channel_observations(
        self,
        merged: RankObservations,
        ch_window_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten (window, channel) tensor to per-observation arrays.

        Returns (vectors, recording_ids, channel_ids, window_ids).
        """
        ch_reps = merged.channel_representations
        if ch_reps is None:
            return np.empty((0, 1)), np.array([]), np.array([]), np.array([])

        full_ch_idx = merged.channel_indices
        full_ch_mask = merged.channel_masks

        vectors_list = []
        recording_ids_list = []
        channel_ids_list = []
        window_ids_list = []

        ch_reps_np = ch_reps.numpy()

        for win_idx in ch_window_indices:
            identity = merged.identities[win_idx]
            n_valid = merged.channel_counts[win_idx]

            if full_ch_mask is not None:
                mask = full_ch_mask[win_idx].numpy().astype(bool)
            else:
                mask = np.zeros(ch_reps_np.shape[1], dtype=bool)
                mask[: min(n_valid, ch_reps_np.shape[1])] = True

            for c in range(ch_reps_np.shape[1]):
                if c >= len(mask) or not mask[c]:
                    continue

                vec = ch_reps_np[win_idx, c]
                vectors_list.append(vec)
                recording_id = (
                    f"{identity.dataset_id}/{identity.subject_id}/"
                    f"{identity.session_id}"
                )
                recording_ids_list.append(recording_id)

                if full_ch_idx is not None and win_idx < full_ch_idx.shape[0]:
                    token_idx = int(full_ch_idx[win_idx, c].item())
                    ch_name = self._idx_to_channel.get(
                        token_idx, f"channel_{token_idx}"
                    )
                else:
                    ch_name = f"channel_{c}"
                channel_ids_list.append(ch_name)

                win_id = (
                    f"{identity.session_id}/"
                    f"{identity.absolute_start:.6f}/"
                    f"{identity.window_duration:.6f}"
                )
                window_ids_list.append(win_id)

        if not vectors_list:
            return np.empty((0, 1)), np.array([]), np.array([]), np.array([])

        if self._resolve_channel_mode() == "static":
            # Static lookup vectors do not vary by window. Keep one point per
            # validation-observed recording/channel pair instead of weighting a
            # channel by how often its recording was sampled.
            keep = []
            seen = set()
            for i, key in enumerate(
                zip(recording_ids_list, channel_ids_list, strict=True)
            ):
                if key not in seen:
                    seen.add(key)
                    keep.append(i)
            vectors_list = [vectors_list[i] for i in keep]
            recording_ids_list = [recording_ids_list[i] for i in keep]
            channel_ids_list = [channel_ids_list[i] for i in keep]
            window_ids_list = [window_ids_list[i] for i in keep]

        return (
            np.array(vectors_list),
            np.array(recording_ids_list),
            np.array(channel_ids_list),
            np.array(window_ids_list),
        )

    def _extract_task_labels(
        self,
        merged: RankObservations,
        selection: SelectedObservations,
        valid_indices: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, list[str] | None]]:
        """Extract per-task class labels for valid backbone windows.

        Returns {task_name: (labels, valid_mask, class_names)}.
        Only includes windows with a single valid, non-negative class label.
        """
        result = {}
        for task_name, all_targets in merged.target_values.items():
            selected_targets = all_targets[selection.window_indices]
            valid_targets = selected_targets[valid_indices]

            target_rows = valid_targets.numpy()
            if target_rows.ndim == 1:
                target_rows = target_rows[:, np.newaxis]
            else:
                target_rows = target_rows.reshape(target_rows.shape[0], -1)

            valid_values = target_rows >= 0
            n_valid_values = valid_values.sum(axis=1)
            labels = (
                np.where(valid_values, target_rows, -1).max(axis=1).astype(int)
            )
            task_valid = n_valid_values > 0
            for i, row in enumerate(target_rows):
                row_values = row[valid_values[i]]
                if len(row_values) > 1 and not np.all(
                    row_values == row_values[0]
                ):
                    task_valid[i] = False
            class_names = self._task_class_names.get(task_name)

            if task_valid.sum() >= 2:
                result[task_name] = (labels, task_valid, class_names)

        return result

    def _clear_buffers(self) -> None:
        self._rank_obs = None
        self._local_identities = []
        self._local_channel_counts = []
        self._sample_metadata_lists = {}

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        setter = getattr(
            pl_module, "set_validation_representation_capture", None
        )
        if setter is not None:
            setter(False)
        self._clear_buffers()
