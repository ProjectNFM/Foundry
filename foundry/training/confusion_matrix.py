"""Confusion matrix computation and tracking for classification tasks."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw counts and row-normalized confusion matrices.

    Args:
        preds: Predicted class indices (1D integer tensor).
        targets: Ground-truth class indices (1D integer tensor).
        num_classes: Total number of classes.

    Returns:
        Tuple of (counts, normalized) where:
        - counts: (num_classes, num_classes) integer matrix, rows=true, cols=pred
        - normalized: (num_classes, num_classes) float matrix, row-normalized.
          Rows with zero samples are left as zeros.
    """
    counts = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        counts[t, p] += 1

    row_sums = counts.sum(dim=1, keepdim=True).float()
    normalized = torch.zeros(num_classes, num_classes)
    nonzero_rows = row_sums.squeeze() > 0
    normalized[nonzero_rows] = (
        counts[nonzero_rows].float() / row_sums[nonzero_rows]
    )

    return counts, normalized


class ConfusionMatrixTracker:
    """Accumulates predictions across batches for epoch-level confusion matrix.

    Args:
        num_classes: Number of classification classes.
        class_names: Optional display names for each class. Defaults to class_<i>.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            f"class_{i}" for i in range(num_classes)
        ]
        self._all_preds: list[torch.Tensor] = []
        self._all_targets: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Out-of-bounds or negative targets are dropped with a warning.
        """
        valid = (targets >= 0) & (targets < self.num_classes)
        if not valid.all():
            n_invalid = (~valid).sum().item()
            logger.warning(
                "ConfusionMatrixTracker: %d invalid targets dropped", n_invalid
            )
        self._all_preds.append(preds[valid].detach().cpu())
        self._all_targets.append(targets[valid].detach().cpu())

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute confusion matrix from all accumulated batches."""
        if not self._all_preds:
            return (
                torch.zeros(
                    self.num_classes, self.num_classes, dtype=torch.long
                ),
                torch.zeros(self.num_classes, self.num_classes),
            )
        all_preds = torch.cat(self._all_preds)
        all_targets = torch.cat(self._all_targets)
        return compute_confusion_matrix(
            all_preds, all_targets, self.num_classes
        )

    def reset(self) -> None:
        """Clear accumulated state for next epoch."""
        self._all_preds.clear()
        self._all_targets.clear()

    def log_wandb(
        self,
        experiment,
        task_name: str,
        epoch: int,
        counts: torch.Tensor,
        normalized: torch.Tensor,
    ) -> None:
        """Log confusion matrix as a W&B image so the media panel gets a step slider.

        Rendering a matplotlib heatmap and logging it as ``wandb.Image``
        gives an automatic epoch slider in the W&B UI.
        """
        if counts.sum() == 0:
            return

        try:
            import wandb

            fig = self._render_confusion_figure(
                counts, normalized, task_name, epoch
            )
            experiment.log(
                {f"val/{task_name}_confusion_matrix": wandb.Image(fig)},
                commit=False,
            )
            fig.clear()
        except Exception:
            logger.warning(
                "Failed to log W&B confusion matrix for %s",
                task_name,
                exc_info=True,
            )

    def _render_confusion_figure(
        self,
        counts: torch.Tensor,
        normalized: torch.Tensor,
        task_name: str,
        epoch: int,
    ):
        """Render a matplotlib heatmap with counts, percentages, and per-class totals.

        Uses the OOP Figure API so no display/backend is required.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        counts_np = counts.numpy()
        norm_np = normalized.numpy()
        n = self.num_classes
        row_sums = counts_np.sum(axis=1)
        total_samples = int(counts_np.sum())

        fig = Figure(figsize=(max(6, n + 2), max(6, n + 2)))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(norm_np, vmin=0, vmax=1, cmap="Blues")

        for i in range(n):
            for j in range(n):
                count = int(counts_np[i, j])
                pct = norm_np[i, j] * 100
                color = "white" if norm_np[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{count}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )

        y_labels = [
            f"{name} (n={int(row_sums[i])})"
            for i, name in enumerate(self.class_names)
        ]

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{task_name} (N={total_samples})")

        fig.colorbar(im, ax=ax, label="Row-normalized ratio", shrink=0.8)
        fig.tight_layout()
        return fig
