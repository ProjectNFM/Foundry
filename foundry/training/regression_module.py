from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def _create_task_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(multioutput="uniform_average"),
        },
        prefix=prefix,
    )


class RegressionModule(L.LightningModule):
    """PyTorch Lightning wrapper for multitask regression models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_names: dict[str, list[str]] | None = None,
        class_weights: dict[str, list[float]] | None = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

        for task_name in model.readout_specs:
            self.train_metrics[task_name] = _create_task_metrics(
                f"train/{task_name}_"
            )
            self.val_metrics[task_name] = _create_task_metrics(
                f"val/{task_name}_"
            )

        self.save_hyperparameters(
            ignore=["model", "class_names", "class_weights"]
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Downcast float64 tensors to float32 for MPS compatibility."""
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
        model_inputs, target_values, target_weights, output_decoder_index = (
            self._unpack_batch(batch)
        )
        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_multitask_loss(
            outputs, target_values, target_weights, output_decoder_index
        )
        self.log("train/loss", total_loss, prog_bar=True)

        for task_name, task_output in outputs.items():
            target = target_values.get(task_name)
            if target is None or target.numel() == 0:
                continue
            self.train_metrics[task_name].update(task_output, target)
            self.log(f"train/{task_name}_loss", taskwise_loss[task_name])
            self.log_dict(
                self.train_metrics[task_name],
                on_step=False,
                on_epoch=True,
            )

        return total_loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        model_inputs, target_values, target_weights, output_decoder_index = (
            self._unpack_batch(batch)
        )
        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_multitask_loss(
            outputs, target_values, target_weights, output_decoder_index
        )
        self.log("val/loss", total_loss, prog_bar=True)

        for task_name, task_output in outputs.items():
            target = target_values.get(task_name)
            if target is None or target.numel() == 0:
                continue
            self.val_metrics[task_name].update(task_output, target)
            self.log(f"val/{task_name}_loss", taskwise_loss[task_name])
            self.log_dict(
                self.val_metrics[task_name],
                on_step=False,
                on_epoch=True,
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
                summary_mode = "max" if metric_name.endswith("_r2") else "min"
                experiment.define_metric(metric_name, summary=summary_mode)

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
            taskwise_loss[readout_id] = spec.loss_fn(
                task_output, target, weights
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
