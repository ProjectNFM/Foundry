from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def _create_regression_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(multioutput="uniform_average"),
        },
        prefix=prefix,
    )


def _create_classification_metrics(
    num_classes: int, prefix: str
) -> MetricCollection:
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
            "balanced_acc": Accuracy(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "cohen_kappa": CohenKappa(task=task_type, num_classes=num_classes),
        },
        prefix=prefix,
    )


class BaseMultitaskModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

    def _initialize_task_modules(self) -> None:
        for task_name, spec in self.model.readout_specs.items():
            self.train_metrics[task_name] = self._build_task_metrics(
                task_name, spec, f"train/{task_name}_"
            )
            self.val_metrics[task_name] = self._build_task_metrics(
                task_name, spec, f"val/{task_name}_"
            )
            self._initialize_task_state(task_name, spec)

    def _build_task_metrics(
        self, task_name: str, spec: Any, prefix: str
    ) -> MetricCollection:
        raise NotImplementedError

    def _initialize_task_state(self, task_name: str, spec: Any) -> None:
        return None

    def _prepare_metric_inputs(
        self,
        task_output: torch.Tensor,
        target: torch.Tensor,
        task_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return task_output, target

    def _prepare_task_loss_inputs(
        self,
        task_output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float,
        task_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        return target, weights

    def _metric_summary_mode(self, metric_name: str) -> str:
        return "min"

    def _update_validation_task_state(
        self,
        task_name: str,
        metric_preds: torch.Tensor,
        metric_target: torch.Tensor,
    ) -> None:
        return None

    def _on_validation_epoch_end_task(self, task_name: str) -> None:
        return None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda tensor: (
                tensor.float() if tensor.dtype == torch.float64 else tensor
            ),
        )
        return move_data_to_device(batch, device)

    def forward(self, **kwargs) -> Dict[str, Any]:
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step("train", batch)

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step("val", batch)

    def _shared_step(self, stage: str, batch: Dict[str, Any]) -> torch.Tensor:
        model_inputs, target_values, target_weights, output_decoder_index = (
            self._unpack_batch(batch)
        )
        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_multitask_loss(
            outputs, target_values, target_weights, output_decoder_index
        )
        self.log(f"{stage}/loss", total_loss, prog_bar=True)

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        for task_name, task_output in outputs.items():
            target = target_values.get(task_name)
            if target is None or target.numel() == 0:
                continue

            metric_preds, metric_target = self._prepare_metric_inputs(
                task_output, target, task_name
            )
            metrics[task_name].update(metric_preds, metric_target)
            self.log(f"{stage}/{task_name}_loss", taskwise_loss[task_name])
            self.log_dict(
                metrics[task_name],
                on_step=False,
                on_epoch=True,
            )

            if stage == "val":
                self._update_validation_task_state(
                    task_name, metric_preds, metric_target
                )

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
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
                experiment.define_metric(
                    metric_name,
                    summary=self._metric_summary_mode(metric_name),
                )

    def on_validation_epoch_end(self):
        for task_name in self.model.readout_specs:
            self._on_validation_epoch_end_task(task_name)

    def _unpack_batch(self, batch: Dict[str, Any]):
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)

        output_decoder_index = batch["output_decoder_index"]
        return batch, target_values, target_weights, output_decoder_index

    def _compute_multitask_loss(
        self,
        outputs: Dict[str, Any],
        target_values: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        output_decoder_index: torch.Tensor,
    ):
        multitask_loss = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        taskwise_loss: dict[str, torch.Tensor] = {}
        total_sequences = 0

        for readout_id, task_output in outputs.items():
            target = target_values.get(readout_id)
            if target is None or target.numel() == 0:
                continue

            spec = self.model.readout_specs[readout_id]
            weights = target_weights.get(readout_id, 1.0)
            target_for_loss, weights_for_loss = self._prepare_task_loss_inputs(
                task_output, target, weights, readout_id
            )
            taskwise_loss[readout_id] = spec.loss_fn(
                task_output, target_for_loss, weights_for_loss
            )

            num_sequences = torch.any(
                output_decoder_index == spec.id, dim=1
            ).sum()
            multitask_loss = (
                multitask_loss + taskwise_loss[readout_id] * num_sequences
            )
            total_sequences += num_sequences

        if total_sequences > 0:
            multitask_loss = multitask_loss / total_sequences

        return multitask_loss, taskwise_loss


class RegressionModule(BaseMultitaskModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_names: dict[str, list[str]] | None = None,
        class_weights: dict[str, list[float]] | None = None,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self._initialize_task_modules()
        self.save_hyperparameters(
            ignore=["model", "class_names", "class_weights"]
        )

    def _build_task_metrics(
        self, task_name: str, spec: Any, prefix: str
    ) -> MetricCollection:
        return _create_regression_metrics(prefix)

    def _metric_summary_mode(self, metric_name: str) -> str:
        return "max" if metric_name.endswith("_r2") else "min"


class ClassificationModule(BaseMultitaskModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_names: dict[str, list[str]] | None = None,
        class_weights: dict[str, list[float]] | None = None,
        class_weight_smoothing: float = 1.0,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self._class_names = class_names or {}

        self._class_weights: dict[str, torch.Tensor] = {}
        if class_weights:
            for name, weights in class_weights.items():
                self._class_weights[name] = torch.tensor(
                    weights, dtype=torch.float32
                )

        self.val_confusion_matrices = nn.ModuleDict()
        self._initialize_task_modules()
        self.save_hyperparameters(ignore=["model"])

    def _build_task_metrics(
        self, task_name: str, spec: Any, prefix: str
    ) -> MetricCollection:
        return _create_classification_metrics(spec.dim, prefix)

    def _initialize_task_state(self, task_name: str, spec: Any) -> None:
        task_type = "binary" if spec.dim == 2 else "multiclass"
        self.val_confusion_matrices[task_name] = ConfusionMatrix(
            task=task_type, num_classes=spec.dim
        )

    def _prepare_metric_inputs(
        self,
        task_output: torch.Tensor,
        target: torch.Tensor,
        task_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(task_output, dim=-1)
        spec = self.model.readout_specs[task_name]
        metric_preds = probs[:, 1] if spec.dim == 2 else probs
        metric_target = self._apply_label_mapping(target, task_name)
        return metric_preds, metric_target

    def _prepare_task_loss_inputs(
        self,
        task_output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float,
        task_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        if task_name not in self._class_weights:
            return target, weights

        class_weights = self._class_weights[task_name].to(target.device)
        mapped_target = self._apply_label_mapping(target, task_name)
        sample_weights = class_weights[mapped_target.long()]

        if isinstance(weights, torch.Tensor):
            weights = weights * sample_weights
        else:
            weights = sample_weights

        return target, weights

    def _metric_summary_mode(self, metric_name: str) -> str:
        return "max"

    def _update_validation_task_state(
        self,
        task_name: str,
        metric_preds: torch.Tensor,
        metric_target: torch.Tensor,
    ) -> None:
        self.val_confusion_matrices[task_name].update(
            metric_preds, metric_target
        )

    def _on_validation_epoch_end_task(self, task_name: str) -> None:
        if task_name not in self.val_confusion_matrices:
            return

        import matplotlib.pyplot as plt
        from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

        confusion_matrix = self.val_confusion_matrices[task_name]
        matrix = confusion_matrix.compute()
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
        confusion_matrix.reset()

    def _apply_label_mapping(
        self, target: torch.Tensor, task_name: str
    ) -> torch.Tensor:
        from foundry.data.datasets.modalities import MappedCrossEntropyLoss

        spec = self.model.readout_specs[task_name]
        loss_fn = spec.loss_fn

        if isinstance(loss_fn, MappedCrossEntropyLoss):
            mapped_target = torch.zeros_like(target)
            for i, key in enumerate(loss_fn._keys):
                mapped_target[target == key] = loss_fn._values[i]
            return mapped_target
        return target

    def _plot_confusion_matrix(
        self,
        matrix: torch.Tensor,
        task_name: str,
        class_names: list[str] | None = None,
    ):
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
        image = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

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

        fig.colorbar(image, ax=ax, label="Row-normalized ratio", shrink=0.8)
        plt.tight_layout()
        return fig
