"""Embedding visualization callback for monitoring representation structure."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning import Trainer

from foundry.training.step_output import extract_step_output

log = logging.getLogger(__name__)


class EmbeddingVisualizationCallback(L.Callback):
    """Visualize backbone embedding structure during validation epochs.

    Registers a forward hook on ``model.backbone.processor`` to capture latent
    representations. At each validation epoch end (respecting ``every_n_epochs``),
    computes PCA, optional t-SNE, and silhouette score on pooled embeddings,
    logging scatter plots and metrics to W&B.

    When no classification labels are available (e.g. during MAE pretraining),
    embeddings are colored by session ID instead.

    Args:
        max_samples: Maximum samples to buffer per validation epoch.
        every_n_epochs: How often to compute and log visualizations.
        compute_tsne: Whether to compute t-SNE (slower, off by default).
        class_names: Optional list of class names for legend labels.
    """

    def __init__(
        self,
        max_samples: int = 2048,
        every_n_epochs: int = 1,
        compute_tsne: bool = False,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.max_samples = max_samples
        self.every_n_epochs = every_n_epochs
        self.compute_tsne = compute_tsne
        self.class_names = class_names

        self._hook = None
        self._ch_hook = None
        self._latents: torch.Tensor | None = None
        self._ch_latents: torch.Tensor | None = None
        self._emb_buffer: list[np.ndarray] = []
        self._label_buffer: list[np.ndarray] = []
        self._session_buffer: list[str] = []
        self._n_collected = 0

        self._ch_emb_buffer: list[np.ndarray] = []
        self._ch_name_buffer: list[str] = []
        self._n_ch_collected = 0
        self._idx_to_channel: dict[int, str] = {}

    def _hook_fn(self, module, input, output):
        self._latents = output.detach()

    def _ch_hook_fn(self, module, input, output):
        self._ch_latents = output.detach()

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if not hasattr(model, "backbone") or not hasattr(
            model.backbone, "processor"
        ):
            log.warning(
                "EmbeddingVisualizationCallback: model has no "
                "backbone.processor — disabling."
            )
            return

        self._hook = model.backbone.processor.register_forward_hook(
            self._hook_fn
        )

        if (
            getattr(model, "channel_emb_mode", None) == "dynamic"
            and getattr(model, "relative_channel_encoder", None) is not None
        ):
            self._ch_hook = (
                model.relative_channel_encoder.register_forward_hook(
                    self._ch_hook_fn
                )
            )
            vocab = getattr(model.channel_emb, "vocab", None)
            if vocab is not None:
                self._idx_to_channel = {
                    idx: name for name, idx in vocab.items()
                }

        if self.class_names is None:
            self._discover_class_names(model)

    def _discover_class_names(self, model) -> None:
        """Auto-discover class names from task_configs if available."""
        if not hasattr(model, "task_configs"):
            return
        for cfg in model.task_configs.values():
            if hasattr(cfg, "class_mapping") and cfg.class_mapping is not None:
                self.class_names = cfg.get_class_names()
                return

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._hook is None or self._latents is None:
            return
        if self._n_collected >= self.max_samples:
            return

        latents = self._latents
        pooled = latents.mean(dim=1).cpu().numpy()  # (B, embed_dim)

        step_output = extract_step_output(outputs)

        labels = None
        if step_output is not None and step_output.target_values:
            for task_name, targets in step_output.target_values.items():
                if "sleep" in task_name:
                    labels = targets.cpu().numpy()
                    break
            if labels is None:
                first_targets = next(iter(step_output.target_values.values()))
                labels = first_targets.cpu().numpy()

        session_ids = None
        if step_output is not None and step_output.session_id is not None:
            session_ids = step_output.session_id

        remaining = self.max_samples - self._n_collected
        n_take = min(pooled.shape[0], remaining)

        self._emb_buffer.append(pooled[:n_take])
        if labels is not None:
            self._label_buffer.append(labels[:n_take])
        if session_ids is not None:
            self._session_buffer.extend(session_ids[:n_take])

        self._n_collected += n_take
        self._latents = None

        if (
            self._ch_hook is not None
            and self._ch_latents is not None
            and self._n_ch_collected < self.max_samples
        ):
            ch_embs = self._ch_latents  # (B, C, channel_emb_dim)
            ch_idx = batch.get("input_channel_index")  # (B, C)
            ch_mask = batch.get("input_mask")  # (B, C)
            if ch_idx is not None and ch_mask is not None:
                ch_embs_np = ch_embs.cpu().numpy()
                ch_idx_np = ch_idx.cpu().numpy()
                ch_mask_np = ch_mask.cpu().numpy().astype(bool)
                for b in range(ch_embs_np.shape[0]):
                    if self._n_ch_collected >= self.max_samples:
                        break
                    for c in range(ch_embs_np.shape[1]):
                        if not ch_mask_np[b, c]:
                            continue
                        if self._n_ch_collected >= self.max_samples:
                            break
                        token_idx = int(ch_idx_np[b, c])
                        name = self._idx_to_channel.get(
                            token_idx, f"channel_{token_idx}"
                        )
                        self._ch_emb_buffer.append(ch_embs_np[b, c])
                        self._ch_name_buffer.append(name)
                        self._n_ch_collected += 1
            self._ch_latents = None

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._hook is None:
            return
        if not self._emb_buffer:
            self._clear_buffers()
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            self._clear_buffers()
            return

        from foundry.training.callbacks import get_wandb_experiment

        wandb_experiment = get_wandb_experiment(trainer)
        if wandb_experiment is None:
            self._clear_buffers()
            return

        embeddings = np.concatenate(self._emb_buffer, axis=0)
        labels = (
            np.concatenate(self._label_buffer, axis=0)
            if self._label_buffer
            else None
        )
        session_ids = self._session_buffer if self._session_buffer else None

        self._compute_and_log(
            embeddings, labels, session_ids, wandb_experiment, trainer
        )
        self._log_channel_embeddings(wandb_experiment, trainer)
        self._clear_buffers()

    def _compute_and_log(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray | None,
        session_ids: list[str] | None,
        wandb_experiment,
        trainer: Trainer,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        try:
            import wandb
        except ImportError:
            return

        n_components_50 = min(50, embeddings.shape[1], embeddings.shape[0])
        pca_50 = PCA(n_components=n_components_50)
        emb_pca50 = pca_50.fit_transform(embeddings)

        n_components_2 = min(2, n_components_50)
        pca_2 = PCA(n_components=n_components_2)
        emb_pca2 = pca_2.fit_transform(embeddings)

        use_class_labels = (
            labels is not None
            and self.class_names is not None
            and labels.max() >= 0
        )

        if use_class_labels:
            valid_mask = labels >= 0
            plot_labels = labels[valid_mask]
            plot_emb2 = emb_pca2[valid_mask]
            plot_emb50 = emb_pca50[valid_mask]
        else:
            valid_mask = np.ones(len(embeddings), dtype=bool)
            plot_emb2 = emb_pca2
            plot_emb50 = emb_pca50
            plot_labels = None

        log_dict = {}

        # Silhouette score
        color_labels, label_names = self._get_color_labels(
            plot_labels, session_ids, valid_mask
        )
        if color_labels is not None and len(np.unique(color_labels)) > 1:
            from sklearn.metrics import silhouette_score

            try:
                sil = silhouette_score(plot_emb50, color_labels)
                log_dict["val/embedding_silhouette"] = sil
            except Exception:
                pass

        # PCA scatter
        fig = self._make_scatter(
            plot_emb2,
            color_labels,
            title=f"PCA (epoch {trainer.current_epoch})",
            xlabel=f"PC1 ({pca_50.explained_variance_ratio_[0]:.1%})",
            ylabel=f"PC2 ({pca_50.explained_variance_ratio_[1]:.1%})",
            label_names=label_names,
        )
        log_dict["val/embedding_pca"] = wandb.Image(fig)
        plt.close(fig)

        # t-SNE scatter (optional)
        if self.compute_tsne:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(plot_emb50) - 1),
                max_iter=1000,
                random_state=42,
                init="pca",
            )
            emb_tsne = tsne.fit_transform(plot_emb50)
            fig_tsne = self._make_scatter(
                emb_tsne,
                color_labels,
                title=f"t-SNE (epoch {trainer.current_epoch})",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
                label_names=label_names,
            )
            log_dict["val/embedding_tsne"] = wandb.Image(fig_tsne)
            plt.close(fig_tsne)

        wandb_experiment.log(log_dict, commit=False)

    @staticmethod
    def _extract_dataset_prefix(namespaced_id: str) -> str:
        """Return dataset prefix from a namespaced ID (everything before first '/')."""
        return (
            namespaced_id.split("/")[0]
            if "/" in namespaced_id
            else namespaced_id
        )

    def _get_color_labels(
        self,
        class_labels: np.ndarray | None,
        session_ids: list[str] | None,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        """Return integer labels for coloring and corresponding display names.

        Prefers class labels when available; otherwise groups by dataset prefix
        extracted from session IDs.
        """
        if class_labels is not None and self.class_names is not None:
            return class_labels, self.class_names
        if session_ids:
            filtered = [s for s, v in zip(session_ids, valid_mask) if v]
            prefixes = [self._extract_dataset_prefix(s) for s in filtered]
            unique_prefixes = sorted(set(prefixes))
            prefix_to_idx = {p: i for i, p in enumerate(unique_prefixes)}
            int_labels = np.array([prefix_to_idx[p] for p in prefixes])
            return int_labels, unique_prefixes
        return None, None

    def _make_scatter(
        self,
        emb_2d: np.ndarray,
        labels: np.ndarray | None,
        title: str,
        xlabel: str,
        ylabel: str,
        label_names: list[str] | None = None,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if labels is None:
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.4, s=8)
        else:
            unique_labels = sorted(np.unique(labels))
            colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 1)))
            for idx, lbl in enumerate(unique_labels):
                mask = labels == lbl
                if label_names is not None and lbl < len(label_names):
                    name = label_names[lbl]
                else:
                    name = f"Group {lbl}"
                ax.scatter(
                    emb_2d[mask, 0],
                    emb_2d[mask, 1],
                    c=[colors[idx]],
                    label=name,
                    alpha=0.4,
                    s=8,
                )
            ax.legend(markerscale=3, fontsize=8)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig

    def _log_channel_embeddings(
        self, wandb_experiment, trainer: Trainer
    ) -> None:
        """Reduce buffered dynamic channel embeddings and log scatter to W&B."""
        if not self._ch_emb_buffer:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        try:
            import wandb
        except ImportError:
            return

        embeddings = np.stack(self._ch_emb_buffer, axis=0)  # (N, ch_emb_dim)
        channel_names = self._ch_name_buffer

        prefixes = [self._extract_dataset_prefix(n) for n in channel_names]
        unique_prefixes = sorted(set(prefixes))
        prefix_to_idx = {p: i for i, p in enumerate(unique_prefixes)}
        color_labels = np.array([prefix_to_idx[p] for p in prefixes])

        n_components = min(2, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=n_components)
        emb_pca2 = pca.fit_transform(embeddings)

        log_dict = {}

        fig = self._make_scatter(
            emb_pca2,
            color_labels,
            title=f"Channel Emb PCA (epoch {trainer.current_epoch})",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            label_names=unique_prefixes,
        )
        log_dict["val/channel_embedding_pca"] = wandb.Image(fig)
        plt.close(fig)

        if self.compute_tsne and embeddings.shape[0] > 2:
            from sklearn.manifold import TSNE

            n_pca = min(50, embeddings.shape[1], embeddings.shape[0])
            pca_50 = PCA(n_components=n_pca)
            emb_pca50 = pca_50.fit_transform(embeddings)

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(emb_pca50) - 1),
                max_iter=1000,
                random_state=42,
                init="pca",
            )
            emb_tsne = tsne.fit_transform(emb_pca50)
            fig_tsne = self._make_scatter(
                emb_tsne,
                color_labels,
                title=f"Channel Emb t-SNE (epoch {trainer.current_epoch})",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
                label_names=unique_prefixes,
            )
            log_dict["val/channel_embedding_tsne"] = wandb.Image(fig_tsne)
            plt.close(fig_tsne)

        fig_scalp = self._make_channel_scalp_scatter(
            embeddings, channel_names, trainer
        )
        if fig_scalp is not None:
            log_dict["val/channel_embedding_pca_scalp"] = wandb.Image(fig_scalp)
            plt.close(fig_scalp)

        wandb_experiment.log(log_dict, commit=False)

    @classmethod
    def _get_electrode_positions_2d(
        cls,
    ) -> dict[str, tuple[float, float]]:
        """2D scalp positions for standard EEG electrodes via MNE montages.

        Cached after first call.  Returns ``{lowercase_name: (x, y)}`` where
        *x* is left(−)/right(+) and *y* is posterior(−)/anterior(+), in
        meters.
        """
        cached = getattr(cls, "_electrode_pos_cache", None)
        if cached is not None:
            return cached

        try:
            import mne

            mne.set_log_level("ERROR")
        except ImportError:
            log.info(
                "MNE not installed — skipping scalp-position channel plot."
            )
            cls._electrode_pos_cache = {}
            return cls._electrode_pos_cache

        positions: dict[str, tuple[float, float]] = {}
        for montage_name in ("standard_1020", "standard_1005"):
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                for ch, xyz in montage.get_positions()["ch_pos"].items():
                    key = ch.lower()
                    if key not in positions:
                        positions[key] = (float(xyz[0]), float(xyz[1]))
            except Exception:
                continue

        cls._electrode_pos_cache = positions
        return positions

    @staticmethod
    def _extract_electrode_name(namespaced_id: str) -> str:
        """Bare electrode name from a namespaced channel ID.

        ``'dataset_name/sub-01/Fp1'`` → ``'Fp1'``
        """
        parts = namespaced_id.split("/")
        return parts[-1] if len(parts) > 1 else namespaced_id

    def _make_channel_scalp_scatter(
        self,
        embeddings: np.ndarray,
        channel_names: list[str],
        trainer: "Trainer",
    ):
        """Averaged-per-electrode PCA colored by 2D scalp position.

        Groups buffered channel embeddings by bare electrode name,
        averages, runs PCA, and colors each dot with an HSV colorwheel
        encoding of the electrode's scalp coordinates.  A colorwheel
        legend is drawn alongside.

        Returns the figure, or ``None`` when positions cannot be resolved.
        """
        from collections import defaultdict

        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb as _hsv
        from sklearn.decomposition import PCA

        electrode_pos = self._get_electrode_positions_2d()
        if not electrode_pos:
            return None

        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for name, emb in zip(channel_names, embeddings):
            groups[self._extract_electrode_name(name)].append(emb)

        names = sorted(groups)
        avg = np.array([np.mean(groups[n], axis=0) for n in names])

        if avg.shape[0] < 3:
            return None

        pca = PCA(n_components=min(2, *avg.shape))
        coords = pca.fit_transform(avg)

        xs = np.zeros(len(names))
        ys = np.zeros(len(names))
        matched = np.zeros(len(names), dtype=bool)
        for i, n in enumerate(names):
            key = n.lower()
            if key in electrode_pos:
                xs[i], ys[i] = electrode_pos[key]
                matched[i] = True

        if matched.sum() < 3:
            return None

        max_d = np.sqrt(xs[matched] ** 2 + ys[matched] ** 2).max() or 1.0
        angles = np.arctan2(ys, xs)
        hues = (angles + np.pi) / (2 * np.pi)
        sats = np.clip(np.sqrt(xs**2 + ys**2) / max_d, 0.15, 1.0)
        hsv = np.stack([hues, sats, np.full_like(hues, 0.85)], axis=-1)
        rgb = _hsv(hsv.reshape(-1, 1, 3)).reshape(-1, 3)

        fig = plt.figure(figsize=(11, 8), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.12)
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])

        ax.scatter(
            coords[matched, 0],
            coords[matched, 1],
            c=rgb[matched],
            s=50,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.85,
            zorder=3,
        )
        if (~matched).any():
            ax.scatter(
                coords[~matched, 0],
                coords[~matched, 1],
                c="lightgray",
                s=25,
                alpha=0.5,
                label="no position",
                zorder=2,
            )
            ax.legend(markerscale=1.5, fontsize=8)

        _LABEL_SET = {
            "fp1",
            "fp2",
            "f7",
            "f3",
            "fz",
            "f4",
            "f8",
            "t7",
            "t3",
            "c3",
            "cz",
            "c4",
            "t4",
            "t8",
            "p7",
            "t5",
            "p3",
            "pz",
            "p4",
            "t6",
            "p8",
            "o1",
            "oz",
            "o2",
        }
        for i, n in enumerate(names):
            if n.lower() in _LABEL_SET or len(names) <= 40:
                ax.annotate(
                    n,
                    (coords[i, 0], coords[i, 1]),
                    fontsize=6,
                    alpha=0.8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        ax.set_title(
            f"Channel Emb PCA — Scalp Position (epoch {trainer.current_epoch})"
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

        self._draw_scalp_colorwheel(ax_leg, electrode_pos, max_d)
        return fig

    @staticmethod
    def _draw_scalp_colorwheel(ax, electrode_pos, max_dist):
        """Circular colorwheel legend matching the HSV scalp encoding."""
        from matplotlib.colors import hsv_to_rgb as _hsv

        n = 256
        lin = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(lin, lin)
        R = np.sqrt(X**2 + Y**2)
        T = np.arctan2(Y, X)

        H = (T + np.pi) / (2 * np.pi)
        S = np.clip(R, 0.15, 1.0)
        V = np.full_like(H, 0.85)
        rgb_img = _hsv(np.stack([H, S, V], axis=-1))

        alpha = np.where(R <= 1.0, 1.0, 0.0)
        rgba = np.concatenate([rgb_img, alpha[..., np.newaxis]], axis=-1)
        ax.imshow(rgba, extent=[-1.3, 1.3, -1.3, 1.3], origin="lower")

        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.4)
        ax.plot(
            [-0.1, 0, 0.1],
            [1.0, 1.12, 1.0],
            "k-",
            linewidth=0.8,
            alpha=0.4,
        )

        ax.text(0, 1.25, "Front", ha="center", va="bottom", fontsize=8)
        ax.text(0, -1.22, "Back", ha="center", va="top", fontsize=8)
        ax.text(
            -1.25,
            0,
            "L",
            ha="right",
            va="center",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            1.25,
            0,
            "R",
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
        )

        for _, (x, y) in electrode_pos.items():
            xn = x / max_dist if max_dist > 0 else 0
            yn = y / max_dist if max_dist > 0 else 0
            if xn**2 + yn**2 <= 1.05:
                ax.plot(xn, yn, "k.", markersize=1.5, alpha=0.3)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Scalp Position\n(color key)", fontsize=9)

    def _clear_buffers(self) -> None:
        self._emb_buffer = []
        self._label_buffer = []
        self._session_buffer = []
        self._n_collected = 0
        self._ch_emb_buffer = []
        self._ch_name_buffer = []
        self._n_ch_collected = 0

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self._ch_hook is not None:
            self._ch_hook.remove()
            self._ch_hook = None
