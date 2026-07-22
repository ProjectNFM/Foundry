"""Unified Lightning training module for all task types."""

from __future__ import annotations
import math
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.tasks.config import TaskConfig
from foundry.training.confusion_matrix import ConfusionMatrixTracker


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
        scheduler: str = "warmup_hold_cosine",
        warmup_steps: int = 100,
        hold_steps: int = 10,
        cosine_steps: int = 0,
        decay_steps: int = 100,
        min_lr_factor: float = 0.1,
        cwt_lr_multiplier: float = 1.0,
    ):
        super().__init__()
        if scheduler not in ("warmup_hold_cosine", "cosine_anneal_decay"):
            raise ValueError(
                f"Unknown scheduler={scheduler!r}. Expected "
                "'warmup_hold_cosine' or 'cosine_anneal_decay'."
            )
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.cosine_steps = cosine_steps
        self.decay_steps = decay_steps
        self.min_lr_factor = min_lr_factor
        self.weight_decay = weight_decay
        self.cwt_lr_multiplier = cwt_lr_multiplier
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
        return predictions, targets

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
        model_inputs, target_values, target_weights, task_index, session_id = (
            self._unpack_batch(batch)
        )
        outputs = self.model(**model_inputs, unpack_output=False)

        total_loss, taskwise_loss = self._compute_task_losses(
            outputs, target_values, target_weights, task_index
        )
        self.log(f"{stage}/loss", total_loss, prog_bar=True)

        metrics = self.train_metrics if stage == "train" else self.val_metrics

        accumulate_sessions = (
            stage == "val"
            and session_id is not None
            and getattr(self, "_val_session_buffers", None) is not None
        )

        for name, cfg in self.model.task_configs.items():
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            if name in taskwise_loss:
                self.log(f"{stage}/{name}_loss", taskwise_loss[name])

            if accumulate_sessions and cfg.metrics is not None:
                self._accumulate_session_preds(
                    name, preds, target, task_index, session_id
                )

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

        return total_loss

    def _accumulate_session_preds(
        self,
        task_name: str,
        preds: torch.Tensor,
        target: torch.Tensor,
        task_index: torch.Tensor,
        session_id: list[str],
    ) -> None:
        """Buffer per-session predictions/targets for epoch-end metric computation.

        Uses ``task_index`` (B, n_out) to count how many output positions each
        batch item contributes to this task, then splits the flat concatenated
        ``preds``/``target`` tensors back into per-item chunks grouped by session.
        """
        router_idx = self.model.router.get_task_index_by_name(task_name) + 1
        counts = (task_index == router_idx).sum(dim=1)

        per_item_preds = torch.split(preds, counts.tolist())
        per_item_targets = torch.split(target, counts.tolist())

        if task_name not in self._val_session_buffers:
            self._val_session_buffers[task_name] = {}

        task_buf = self._val_session_buffers[task_name]
        for sid, item_p, item_t in zip(
            session_id, per_item_preds, per_item_targets
        ):
            if item_p.numel() == 0:
                continue
            if sid not in task_buf:
                task_buf[sid] = {"preds": [], "targets": []}
            task_buf[sid]["preds"].append(item_p.detach().cpu())
            task_buf[sid]["targets"].append(item_t.detach().cpu())

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

    @staticmethod
    def _cosine_anneal_factor(
        progress: float, start: float, end: float
    ) -> float:
        """Cosine interpolate from ``start`` to ``end`` for ``progress`` in [0, 1].

        Matches ``torch.optim.lr_scheduler.CosineAnnealingLR`` when
        ``start=1.0`` and ``end=eta_min/base_lr``.
        """
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * cosine


    def lr_lambda(self, current_step: int) -> float:
        """Multiplicative LR schedule for :class:`~torch.optim.lr_scheduler.LambdaLR`.

        ``warmup_hold_cosine``
          1. Linear warmup 0 → 1 over ``warmup_steps``
          2. Hold at 1 for ``hold_steps``
          3. Cosine decay 1 → ``min_lr_factor`` over ``decay_steps``
          4. Hold at ``min_lr_factor``

        ``cosine_anneal_decay``
          1. Smooth cosine oscillation (max → 0 → max) over ``cosine_steps``.
             Period T_0 = trainer.max_epochs if available, else 100.
             Formula: 0.5 * (1 + cos(2π * t / T_0))
          2. Cosine decay 1 → ``min_lr_factor`` over ``decay_steps``
          3. Hold at ``min_lr_factor``
        """
        min_lr_factor = self.min_lr_factor
        decay_steps = self.decay_steps

        if self.scheduler == "warmup_hold_cosine":
            warmup_steps = self.warmup_steps
            hold_steps = self.hold_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < warmup_steps + hold_steps or decay_steps == 0:
                return 1.0
            decay_start = warmup_steps + hold_steps
            if current_step < decay_start + decay_steps:
                # Cosine decay phase
                progress = float(current_step - decay_start) / float(
                    decay_steps
                )
                return self._cosine_anneal_factor(progress, 1.0, min_lr_factor)
            return min_lr_factor

        # cosine_anneal_decay
        cosine_steps = self.cosine_steps
        # If no cosine or decay steps, return 1.0 (no LR decay)
        if cosine_steps == 0 and decay_steps == 0:
            return 1.0

        # Phase 1: Smooth cosine oscillation
        if current_step < cosine_steps:
            # Period T_0 = trainer.max_epochs if available, else 100
            trainer = getattr(self, "_trainer", None)
            T_0 = float(trainer.max_epochs if trainer else 100)
            t = float(current_step)
            # Smooth oscillation: 0.5 * (1 + cos(2π * t / T_0))
            # Starts at 1, goes to 0 at T_0/2, back to 1 at T_0, etc.
            oscillation = 0.5 * (1.0 + math.cos(2.0 * math.pi * t / T_0))
            return oscillation

        # Phase 2: Cosine decay from 1 → min_lr_factor
        if decay_steps == 0:
            return min_lr_factor
        if current_step < cosine_steps + decay_steps:
            progress = float(current_step - cosine_steps) / float(decay_steps)
            return self._cosine_anneal_factor(
                progress, 1.0, min_lr_factor
            )
        return min_lr_factor

    def configure_optimizers(self):
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.lr_lambda
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        multitask_loss = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        taskwise_loss: dict[str, torch.Tensor] = {}
        total_sequences = 0

        for name in self.model.task_configs:
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            weights = target_weights.get(name, 1.0)
            loss = self._task_losses[name](preds, target, weights)
            taskwise_loss[name] = loss

            idx = self.model.router.get_task_index_by_name(name) + 1
            num_sequences = torch.any(task_index == idx, dim=1).sum()
            multitask_loss = multitask_loss + loss * num_sequences
            total_sequences += num_sequences

        if total_sequences > 0:
            multitask_loss = multitask_loss / total_sequences

        return multitask_loss, taskwise_loss
