"""Unified Lightning training module for all task types."""

from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.ssl_meta import ModelOutput
from foundry.tasks.config import TaskConfig
from foundry.training.confusion_matrix import ConfusionMatrixTracker
from foundry.training.step_output import StepOutput


def _squeeze_scalar_predictions(
    preds: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Squeeze trailing dim-1 from predictions when targets are 1-D."""
    if preds.dim() == 2 and preds.shape[1] == 1 and target.dim() == 1:
        return preds.squeeze(-1)
    return preds


class FoundryModule(L.LightningModule):
    """Single training module for classification, regression, and multitask runs.

    Per-task loss functions and metrics are built from :class:`~foundry.tasks.config.TaskConfig`
    entries on ``model.task_configs``. Sequence-weighted multitask loss aggregation,
    CWT LR param groups, and WandB metric summaries match the previous
    Classification/Regression module behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        cwt_lr_multiplier: float = 1.0,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cwt_lr_multiplier = cwt_lr_multiplier
        self.warmup_epochs = warmup_epochs
        self.save_hyperparameters(ignore=["model"])

        self._task_losses = nn.ModuleDict()
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self._val_confusion_trackers: dict[str, ConfusionMatrixTracker] = {}

        for name, cfg in model.task_configs.items():
            self._task_losses[name] = instantiate(cfg.loss)

            if cfg.metrics is not None:
                metrics = instantiate(cfg.metrics)
                self.train_metrics[name] = metrics.clone(
                    prefix=f"train/{name}_"
                )
                self.val_metrics[name] = metrics.clone(prefix=f"val/{name}_")

            if (
                cfg.kind in ("binary", "multiclass")
                and cfg.class_mapping is not None
            ):
                self._val_confusion_trackers[name] = ConfusionMatrixTracker(
                    num_classes=cfg.output_dim,
                    class_names=cfg.get_class_names(),
                )

    def _metric_summary_mode(
        self, task_name: str, metric_name: str, cfg: Any
    ) -> str:
        short_name = metric_name.removeprefix(
            f"train/{task_name}_"
        ).removeprefix(f"val/{task_name}_")
        return cfg.metric_summary_modes.get(short_name, "min")

    def _prepare_for_metrics(
        self,
        cfg: TaskConfig,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg.kind == "multiclass":
            return torch.softmax(predictions, dim=-1), targets
        if cfg.kind == "binary":
            return torch.softmax(predictions, dim=-1)[:, 1], targets
        return _squeeze_scalar_predictions(predictions, targets), targets

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
    ) -> Dict[str, Any]:
        step_output = self._shared_step("train", batch)
        return {"loss": step_output.loss, "step_output": step_output}

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        step_output = self._shared_step("val", batch)
        return {"loss": step_output.loss, "step_output": step_output}

    def _shared_step(self, stage: str, batch: Dict[str, Any]) -> StepOutput:
        model_inputs, target_values, target_weights, task_index, session_id = (
            self._unpack_batch(batch)
        )
        model_output = self.model(**model_inputs, unpack_output=False)

        if isinstance(model_output, ModelOutput):
            outputs = model_output.task_outputs
            ssl_meta = model_output.ssl_meta
            reconstruction_viz = model_output.viz
        else:
            outputs = model_output
            ssl_meta = outputs.pop("_ssl_meta", None)
            reconstruction_viz = outputs.pop("_reconstruction_viz", None)
        ssl_task_names: set[str] = set()
        if ssl_meta is not None:
            for task_name, meta in ssl_meta.items():
                target_values[task_name] = meta.targets
                target_weights[task_name] = meta.weights
                ssl_task_names.add(task_name)

        total_loss, taskwise_loss = self._compute_task_losses(
            outputs, target_values, target_weights, task_index, ssl_task_names
        )
        self.log(f"{stage}/loss", total_loss, prog_bar=True)

        if stage == "train" and getattr(self, "_trainer", None) is not None:
            opt = self.optimizers()
            if opt is not None:
                current_lr = opt.param_groups[0]["lr"]
                self.log("train/lr", current_lr, prog_bar=False)

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        for name, cfg in self.model.task_configs.items():
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            if name in taskwise_loss:
                self.log(f"{stage}/{name}_loss", taskwise_loss[name])

            if cfg.kind in ("binary", "multiclass"):
                valid_mask = target >= 0
                if not valid_mask.all():
                    preds = preds[valid_mask]
                    target = target[valid_mask]
                    if target.numel() == 0:
                        continue

            if name in metrics:
                metric_preds, metric_target = self._prepare_for_metrics(
                    cfg, preds, target
                )
                metrics[name].update(metric_preds, metric_target)
                self.log_dict(
                    metrics[name],
                    on_step=False,
                    on_epoch=True,
                )

            if stage == "val" and name in self._val_confusion_trackers:
                if cfg.kind == "multiclass":
                    pred_classes = preds.argmax(dim=-1)
                elif cfg.kind == "binary":
                    pred_classes = (preds[:, 1] > preds[:, 0]).long()
                else:
                    continue
                self._val_confusion_trackers[name].update(pred_classes, target)

        return StepOutput(
            loss=total_loss,
            task_outputs=outputs,
            target_values=target_values,
            target_weights=target_weights,
            task_index=task_index,
            session_id=session_id,
            ssl_task_names=ssl_task_names,
            reconstruction_viz=reconstruction_viz,
            reconstruction_targets=model_inputs.get("reconstruction_targets"),
            input_mask=model_inputs.get("input_mask"),
        )

    def _build_param_groups(self) -> list[dict]:
        if self.cwt_lr_multiplier == 1.0:
            return [
                {
                    "params": list(self.parameters()),
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            ]

        cwt_params = []
        other_params = []
        for name, param in self.named_parameters():
            if ".cwt." in name:
                cwt_params.append(param)
            else:
                other_params.append(param)

        groups = [
            {
                "params": other_params,
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
        ]
        if cwt_params:
            cwt_lr = self.learning_rate * self.cwt_lr_multiplier
            groups.append(
                {
                    "params": cwt_params,
                    "lr": cwt_lr,
                    "weight_decay": self.weight_decay,
                }
            )
            n_cwt = sum(
                p.numel()
                for p in cwt_params
                if not p.__class__.__name__.startswith("Uninitialized")
            )
            n_other = sum(
                p.numel()
                for p in other_params
                if not p.__class__.__name__.startswith("Uninitialized")
            )
            print(
                f"CWT LR multiplier: {self.cwt_lr_multiplier}x "
                f"(cwt_lr={cwt_lr:.2e}, {n_cwt} params) | "
                f"base_lr={self.learning_rate:.2e} ({n_other} params)"
            )

        return groups

    def configure_optimizers(self):
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups)

        max_epochs = self.trainer.max_epochs if self.trainer else 100
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - self.warmup_epochs)
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = cosine

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

        for name, cfg in self.model.task_configs.items():
            for prefix in ("train", "val"):
                experiment.define_metric(f"{prefix}/{name}_loss", summary="min")
            for metric_name, mode in cfg.metric_summary_modes.items():
                if metric_name == "loss":
                    continue
                for prefix in ("train", "val"):
                    experiment.define_metric(
                        f"{prefix}/{name}_{metric_name}", summary=mode
                    )

        for name in self.model.task_configs:
            cfg = self.model.task_configs[name]
            for metrics_dict in (self.train_metrics, self.val_metrics):
                if name not in metrics_dict:
                    continue
                for metric_name in metrics_dict[name]:
                    experiment.define_metric(
                        metric_name,
                        summary=self._metric_summary_mode(
                            name, metric_name, cfg
                        ),
                    )

    def _unpack_batch(self, batch: Dict[str, Any]):
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        session_id = batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)

        task_index = batch["task_index"]
        return batch, target_values, target_weights, task_index, session_id

    def _compute_task_losses(
        self,
        outputs: dict[str, torch.Tensor],
        target_values: dict[str, torch.Tensor],
        target_weights: dict[str, torch.Tensor | float],
        task_index: torch.Tensor,
        ssl_task_names: set[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        multitask_loss = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        taskwise_loss: dict[str, torch.Tensor] = {}
        total_sequences = 0
        if ssl_task_names is None:
            ssl_task_names = set()

        for name in self.model.task_configs:
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            preds = _squeeze_scalar_predictions(preds, target)

            weights = target_weights.get(name, 1.0)
            loss = self._task_losses[name](preds, target, weights)
            taskwise_loss[name] = loss

            if name in ssl_task_names:
                num_sequences = task_index.shape[0]
            else:
                idx = self.model.router.get_task_index_by_name(name) + 1
                num_sequences = torch.any(task_index == idx, dim=1).sum()
            multitask_loss = multitask_loss + loss * num_sequences
            total_sequences += num_sequences

        if total_sequences > 0:
            multitask_loss = multitask_loss / total_sequences

        return multitask_loss, taskwise_loss
