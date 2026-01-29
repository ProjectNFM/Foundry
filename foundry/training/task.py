from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGTask(L.LightningModule):
    """
    PyTorch Lightning wrapper for EEG model training.

    Handles training and validation loops, loss computation, and optimizer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        """
        Args:
            model: EEGModel instance to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Forward pass through the model."""
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step that computes loss for the batch.

        Args:
            batch: Batch dictionary containing model_inputs, target_values, and target_weights
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        outputs = self.model(**batch, unpack_output=False)

        total_loss = 0.0
        num_tasks = 0

        for task_name, task_output in outputs.items():
            if task_name not in batch["target_values"]:
                continue

            target = batch["target_values"][task_name]
            weight = batch["target_weights"][task_name]

            if target.numel() == 0:
                continue

            logits = task_output["logits"]
            loss = F.cross_entropy(logits, target, reduction="none")
            weighted_loss = (loss * weight).sum() / weight.sum()

            total_loss += weighted_loss
            num_tasks += 1

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                accuracy = (
                    (pred == target).float() * weight
                ).sum() / weight.sum()
                self.log(
                    f"train/{task_name}_loss", weighted_loss, prog_bar=False
                )
                self.log(f"train/{task_name}_acc", accuracy, prog_bar=False)

        if num_tasks > 0:
            total_loss = total_loss / num_tasks

        self.log("train/loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step that computes loss and metrics for the batch.

        Args:
            batch: Batch dictionary containing model_inputs, target_values, and target_weights
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        outputs = self.model(**batch, unpack_output=False)

        total_loss = 0.0
        num_tasks = 0

        for task_name, task_output in outputs.items():
            if task_name not in batch["target_values"]:
                continue

            target = batch["target_values"][task_name]
            weight = batch["target_weights"][task_name]

            if target.numel() == 0:
                continue

            logits = task_output["logits"]
            loss = F.cross_entropy(logits, target, reduction="none")
            weighted_loss = (loss * weight).sum() / weight.sum()

            total_loss += weighted_loss
            num_tasks += 1

            pred = logits.argmax(dim=-1)
            accuracy = ((pred == target).float() * weight).sum() / weight.sum()

            self.log(f"val/{task_name}_loss", weighted_loss, prog_bar=False)
            self.log(f"val/{task_name}_acc", accuracy, prog_bar=False)

        if num_tasks > 0:
            total_loss = total_loss / num_tasks

        self.log("val/loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
