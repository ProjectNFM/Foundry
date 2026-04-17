from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)


def _create_task_metrics(num_classes: int, prefix: str) -> MetricCollection:
    """
    Create a MetricCollection for a given task based on number of classes.

    Args:
        num_classes: Number of classes for the task
        prefix: Prefix for metric names (e.g., "train/task_name_")

    Returns:
        MetricCollection with accuracy, F1, AUROC, precision, and recall
    """
    task_type = "binary" if num_classes == 2 else "multiclass"
    return MetricCollection(
        {
            "acc": Accuracy(task=task_type, num_classes=num_classes),
            "f1": F1Score(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "auroc": AUROC(task=task_type, num_classes=num_classes),
            "precision": Precision(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task=task_type, num_classes=num_classes, average="macro"
            ),
        },
        prefix=prefix,
    )


class ClassificationModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for classification model training.

    Handles training and validation loops, loss computation, and optimizer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_names: dict[str, list[str]] | None = None,
        class_weights: dict[str, list[float]] | None = None,
        class_weight_smoothing: float = 1.0,
    ):
        """
        Args:
            model: EEGModel instance to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            class_names: Optional mapping of readout name -> list of class labels
                         for confusion matrix display (e.g. from DataModule.get_class_names_for_task)
            class_weights: Optional mapping of readout name -> per-class weight list
                           for loss balancing (e.g. inverse frequency weights).
                           Weights are indexed by class index [0, num_classes-1].
            class_weight_smoothing: Exponent applied to class weights before use.
                           Stored for hyperparameter logging only — the actual
                           smoothing is applied during weight computation.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._class_names = class_names or {}

        self._class_weights: dict[str, torch.Tensor] = {}
        if class_weights:
            for name, weights in class_weights.items():
                self._class_weights[name] = torch.tensor(
                    weights, dtype=torch.float32
                )

        self.save_hyperparameters(ignore=["model"])

        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.val_confusion_matrices = nn.ModuleDict()

        for task_name, spec in model.readout_specs.items():
            num_classes = spec.dim
            task_type = "binary" if num_classes == 2 else "multiclass"

            self.train_metrics[task_name] = _create_task_metrics(
                num_classes, f"train/{task_name}_"
            )
            self.val_metrics[task_name] = _create_task_metrics(
                num_classes, f"val/{task_name}_"
            )
            self.val_confusion_matrices[task_name] = ConfusionMatrix(
                task=task_type, num_classes=num_classes
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Downcast float64 tensors to float32 for MPS compatibility."""
        # TODO: Remove this once MPS supports float64
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.float() if t.dtype == torch.float64 else t,
        )
        return move_data_to_device(batch, device)

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
        model_inputs, target_values, target_weights, output_decoder_index = (
            self._unpack_batch(batch)
        )

        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_multitask_loss(
            outputs, target_values, target_weights, output_decoder_index
        )

        self.log("train/loss", total_loss, prog_bar=True)

        for task_name, task_output in outputs.items():
            if task_name not in target_values:
                continue

            target = target_values[task_name]

            if target.numel() == 0:
                continue

            probs = self._get_metric_preds(task_output, task_name)
            mapped_target = self._apply_label_mapping(target, task_name)

            self.train_metrics[task_name].update(probs, mapped_target)

            self.log(
                f"train/{task_name}_loss",
                taskwise_loss[task_name],
                prog_bar=False,
            )

            self.log_dict(
                self.train_metrics[task_name],
                on_step=False,
                on_epoch=True,
            )

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
        model_inputs, target_values, target_weights, output_decoder_index = (
            self._unpack_batch(batch)
        )

        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_multitask_loss(
            outputs, target_values, target_weights, output_decoder_index
        )

        self.log("val/loss", total_loss, prog_bar=True)

        for task_name, task_output in outputs.items():
            if task_name not in target_values:
                continue

            target = target_values[task_name]

            if target.numel() == 0:
                continue

            probs = self._get_metric_preds(task_output, task_name)
            mapped_target = self._apply_label_mapping(target, task_name)

            self.val_metrics[task_name].update(probs, mapped_target)
            self.val_confusion_matrices[task_name].update(probs, mapped_target)

            self.log(
                f"val/{task_name}_loss",
                taskwise_loss[task_name],
                prog_bar=False,
            )

            self.log_dict(
                self.val_metrics[task_name],
                on_step=False,
                on_epoch=True,
            )

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Uses AdamW optimizer with optional cosine annealing learning rate scheduler
        for improved convergence on EEG tasks.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Optional: add learning rate scheduler for better convergence
        # Cosine annealing is commonly used and often improves EEG model training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_fit_start(self):
        self._configure_wandb_metric_summaries()

    def _configure_wandb_metric_summaries(self):
        """Tell W&B to track best (not last) values in run summaries.

        Losses get summary="min", classification metrics get summary="max".
        This matters for W&B sweep optimization which reads from run.summary.
        """
        from lightning.pytorch.loggers import WandbLogger

        if not isinstance(self.logger, WandbLogger):
            return

        experiment = self.logger.experiment

        for prefix in ("train", "val"):
            experiment.define_metric(f"{prefix}/loss", summary="min")
            for task_name in self.model.readout_specs:
                experiment.define_metric(
                    f"{prefix}/{task_name}_loss", summary="min"
                )

        for metrics in (
            *self.train_metrics.values(),
            *self.val_metrics.values(),
        ):
            for metric_name in metrics:
                experiment.define_metric(metric_name, summary="max")

    def on_validation_epoch_end(self):
        """Log confusion matrices at the end of validation epoch."""
        import matplotlib.pyplot as plt
        from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

        for task_name, cm in self.val_confusion_matrices.items():
            matrix = cm.compute()
            class_names = self._class_names.get(task_name)
            fig = self._plot_confusion_matrix(matrix, task_name, class_names)

            if self.logger:
                if isinstance(self.logger, WandbLogger):
                    import wandb

                    self.logger.experiment.log(
                        {f"val/{task_name}_confusion_matrix": wandb.Image(fig)}
                    )
                elif isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(
                        f"val/{task_name}_confusion_matrix",
                        fig,
                        self.current_epoch,
                    )

            plt.close(fig)
            cm.reset()

    def _unpack_batch(self, batch: Dict[str, Any]):
        """
        Separate model inputs from targets and metadata.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Tuple of (model_inputs, target_values, target_weights, output_decoder_index)
        """
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)

        output_decoder_index = batch["output_decoder_index"]
        return batch, target_values, target_weights, output_decoder_index

    def _get_metric_preds(
        self, task_output: torch.Tensor, task_name: str
    ) -> torch.Tensor:
        """Convert raw logits to the format expected by torchmetrics.

        Binary metrics expect a 1-D tensor of positive-class probabilities,
        while multiclass metrics expect the full [N, C] probability matrix.
        """
        probs = torch.softmax(task_output, dim=-1)
        spec = self.model.readout_specs[task_name]
        if spec.dim == 2:
            return probs[:, 1]
        return probs

    def _apply_label_mapping(
        self, target: torch.Tensor, task_name: str
    ) -> torch.Tensor:
        """
        Apply label mapping if the task uses MappedCrossEntropyLoss.

        Args:
            target: Target tensor with original label IDs
            task_name: Name of the task

        Returns:
            Mapped target tensor with class indices [0, num_classes-1]
        """
        from foundry.data.datasets.modalities import MappedCrossEntropyLoss

        spec = self.model.readout_specs[task_name]
        loss_fn = spec.loss_fn

        if isinstance(loss_fn, MappedCrossEntropyLoss):
            mapped_target = torch.zeros_like(target)
            for i, key in enumerate(loss_fn._keys):
                mask = target == key
                mapped_target[mask] = loss_fn._values[i]
            return mapped_target
        return target

    def _compute_multitask_loss(
        self,
        outputs: Dict[str, Any],
        target_values: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        output_decoder_index: torch.Tensor,
    ):
        """
        Compute multitask loss using model's readout specs.

        Args:
            outputs: Model outputs dictionary
            target_values: Target values per task
            target_weights: Target weights per task
            output_decoder_index: Decoder indices indicating which task each output belongs to

        Returns:
            Tuple of (total_loss, taskwise_loss_dict)
        """
        loss_mt = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        taskwise_loss = {}
        total_sequences = 0

        for readout_id, task_output in outputs.items():
            if readout_id not in target_values:
                continue

            target = target_values[readout_id]
            if target.numel() == 0:
                continue

            spec = self.model.readout_specs[readout_id]
            weights = target_weights.get(readout_id, 1.0)

            if readout_id in self._class_weights:
                cw = self._class_weights[readout_id].to(target.device)
                mapped_target = self._apply_label_mapping(target, readout_id)
                sample_cw = cw[mapped_target.long()]
                if isinstance(weights, torch.Tensor):
                    weights = weights * sample_cw
                else:
                    weights = sample_cw

            taskwise_loss[readout_id] = spec.loss_fn(
                task_output, target, weights
            )

            num_sequences = torch.any(
                output_decoder_index == spec.id, dim=1
            ).sum()

            loss_mt = loss_mt + taskwise_loss[readout_id] * num_sequences
            total_sequences += num_sequences

        if total_sequences > 0:
            loss_mt = loss_mt / total_sequences

        return loss_mt, taskwise_loss

    def _plot_confusion_matrix(
        self,
        matrix: torch.Tensor,
        task_name: str,
        class_names: list[str] | None = None,
    ):
        """Create a matplotlib figure for the confusion matrix.

        Displays per-cell counts and row-normalized percentages, with class
        labels and per-class sample totals on the axes.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        cm = matrix.cpu().numpy()
        n_classes = cm.shape[0]
        row_sums = cm.sum(axis=1)
        total_samples = int(cm.sum())

        cm_normalized = np.zeros_like(cm, dtype=float)
        for i in range(n_classes):
            if row_sums[i] > 0:
                cm_normalized[i] = cm[i] / row_sums[i]

        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        fig, ax = plt.subplots(
            figsize=(max(6, n_classes + 2), max(6, n_classes + 2))
        )
        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

        for i in range(n_classes):
            for j in range(n_classes):
                count = int(cm[i, j])
                pct = cm_normalized[i, j] * 100
                color = "white" if cm_normalized[i, j] > 0.5 else "black"
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
            for i, name in enumerate(class_names)
        ]

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{task_name} (N={total_samples})")

        fig.colorbar(im, ax=ax, label="Row-normalized ratio", shrink=0.8)
        plt.tight_layout()
        return fig
