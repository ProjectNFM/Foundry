"""Confusion matrix computation and tracking for classification tasks."""

from __future__ import annotations

import torch


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
        """Accumulate a batch of predictions and targets."""
        self._all_preds.append(preds.detach().cpu())
        self._all_targets.append(targets.detach().cpu())

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
