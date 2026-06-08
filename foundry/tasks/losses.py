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
