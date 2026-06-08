import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyTaskLoss(nn.Module):
    """Cross-entropy loss with optional class weights and label smoothing."""

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
    """MSE loss with optional per-sample weighting."""

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


class FocalTaskLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        ce = F.cross_entropy(predictions, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        if isinstance(sample_weights, torch.Tensor):
            focal = focal * sample_weights
        return focal.mean()
