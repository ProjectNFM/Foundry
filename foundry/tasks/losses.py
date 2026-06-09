"""Task loss functions for the training layer.

Each loss is an ``nn.Module`` with a uniform signature::

    (predictions, targets, sample_weights) -> scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyTaskLoss(nn.Module):
    """Cross-entropy loss for classification tasks.

    Wraps :func:`torch.nn.functional.cross_entropy` with per-sample weighting.
    Class weights and label smoothing are configured at construction time so they
    can be set from YAML and serialized in checkpoints.

    Args:
        label_smoothing: Smoothing factor passed to cross-entropy. ``0.0``
            disables smoothing.
        class_weights: Per-class weights of length ``num_classes``. Registered
            as a buffer when provided.

    Shape:
        - ``predictions``: ``(N, num_classes)`` unnormalized logits.
        - ``targets``: ``(N,)`` integer class indices.
        - ``sample_weights``: scalar or ``(N,)`` tensor; multiplied per sample
            before the batch mean. A scalar is a no-op.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            predictions,
            targets.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights
        return loss.mean()


class MSETaskLoss(nn.Module):
    """Mean squared error loss for regression tasks.

    Computes element-wise MSE between predictions and targets, optionally
    weighting each sample before averaging over the full batch.

    Shape:
        - ``predictions``: ``(N, D)`` predicted values.
        - ``targets``: ``(N, D)`` ground-truth values (same shape as predictions).
        - ``sample_weights``: scalar or ``(N,)`` tensor; broadcast across
            target dimensions when a tensor. A scalar is a no-op.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.mse_loss(predictions, targets, reduction="none")
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights.unsqueeze(-1)
        return loss.mean()
