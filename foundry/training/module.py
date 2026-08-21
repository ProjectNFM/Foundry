"""Unified Lightning training module for all task types."""

from __future__ import annotations

import logging
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.ssl_meta import ModelOutput
from foundry.tasks.config import TaskConfig
from foundry.training.confusion_matrix import ConfusionMatrixTracker
from foundry.training.step_output import StepOutput

logger = logging.getLogger(__name__)


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

    Args:
        learning_rate (float): Base learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) used in the optimizer.
        cwt_lr_multiplier (float): Multiplier to apply to learning rate for CWT parameter groups.
        warmup (int): Number of steps for the learning rate warmup phase.
        start_lr_factor (float): Starting learning rate as a fraction of `learning_rate` during warmup.
        hold (int): Number of steps to hold the learning rate after warmup.
        hold_scheduler_type (str): Type of scheduler to use during the hold phase, e.g. "constant" or "cosine".
        decay (int): Number of steps for cosine learning rate decay after the hold phase.
        end_lr_factor (float): Fraction of `learning_rate` for the final learning rate at the end of decay.
        scheduler_interval (str): Scheduler update interval (e.g. "step" or "epoch").
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        cwt_lr_multiplier: float = 1.0,
        backbone_learning_rate: float | None = None,
        scheduler_name: str = "phased",
        warmup: int = 0,
        start_lr_factor: float = 1e-4,
        hold: int = 0,
        hold_scheduler_type: str = "constant",
        decay: int = 0,
        end_lr_factor: float = 0.1,
        scheduler_interval: str = "step",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cwt_lr_multiplier = cwt_lr_multiplier
        self.backbone_learning_rate = backbone_learning_rate
        self.scheduler_name = scheduler_name
        self.warmup = warmup
        self.hold = hold
        self.hold_scheduler_type = hold_scheduler_type
        self.decay = decay
        self.end_lr_factor = end_lr_factor
        self.start_lr_factor = start_lr_factor
        self.scheduler_interval = scheduler_interval
        self.save_hyperparameters(ignore=["model"])

        self._task_losses = nn.ModuleDict()
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        self._val_confusion_trackers: dict[str, ConfusionMatrixTracker] = {}

        for name, cfg in model.task_configs.items():
            self._task_losses[name] = instantiate(cfg.loss)

            if cfg.metrics is not None:
                metrics = instantiate(cfg.metrics)
                self.train_metrics[name] = metrics.clone(
                    prefix=f"train/{name}_"
                )
                self.val_metrics[name] = metrics.clone(prefix=f"val/{name}_")
                self.test_metrics[name] = metrics.clone(prefix=f"test/{name}_")

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
        """Resolve the WandB summary mode (``"min"``/``"max"``) for a metric."""
        short_name = (
            metric_name.removeprefix(f"train/{task_name}_")
            .removeprefix(f"val/{task_name}_")
            .removeprefix(f"test/{task_name}_")
        )
        return cfg.metric_summary_modes.get(short_name, "min")

    def _prepare_for_metrics(
        self,
        cfg: TaskConfig,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw logits to the format expected by torchmetrics.

        All classification metrics now use ``task="multiclass"`` (even for
        2-class problems) so that ``average="macro"`` is respected for
        balanced accuracy.  This means the full softmax probability vector
        is always passed.
        """
        if cfg.kind in ("multiclass", "binary"):
            return torch.softmax(predictions, dim=-1), targets
        return _squeeze_scalar_predictions(predictions, targets), targets

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move batch tensors to *device*, casting float64 to float32."""
        from lightning_utilities.core.apply_func import apply_to_collection

        def _move_and_convert(tensor):
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.to(device, non_blocking=True)

        return apply_to_collection(
            batch, dtype=torch.Tensor, function=_move_and_convert
        )

    def forward(self, **kwargs) -> Dict[str, Any]:
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Lightning training step — delegates to :meth:`_shared_step`."""
        step_output = self._shared_step("train", batch)
        return {"loss": step_output.loss, "step_output": step_output}

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Lightning validation step — delegates to :meth:`_shared_step`."""
        step_output = self._shared_step("val", batch)
        return {"loss": step_output.loss, "step_output": step_output}

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Lightning test step — evaluates the selected checkpoint on held-out data."""
        step_output = self._shared_step("test", batch)
        return {"loss": step_output.loss, "step_output": step_output}

    def _shared_step(self, stage: str, batch: Dict[str, Any]) -> StepOutput:
        """Run a single training or validation step.

        Unpacks the batch, runs the model forward pass, computes the
        sequence-weighted multitask loss, updates per-task metrics, and
        optionally tracks confusion matrices for classification tasks.

        Args:
            stage: ``"train"`` or ``"val"``.
            batch: Collated batch dict containing model inputs, target
                values/weights, task indices, and metadata.

        Returns:
            :class:`StepOutput` with the total loss, per-task outputs,
            targets, weights, and optional SSL/reconstruction metadata.
        """
        model_inputs, target_values, target_weights, task_index, session_id = (
            self._unpack_batch(batch)
        )
        batch_size = task_index.shape[0]
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
        self.log(
            f"{stage}/loss", total_loss, prog_bar=True, batch_size=batch_size
        )

        metrics_by_stage = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        metrics = metrics_by_stage[stage]

        for name, cfg in self.model.task_configs.items():
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            if name in taskwise_loss:
                self.log(
                    f"{stage}/{name}_loss",
                    taskwise_loss[name],
                    batch_size=batch_size,
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
                    batch_size=batch_size,
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
        """Build optimizer parameter groups with per-component learning rates.

        Three strategies, selected by constructor arguments:

        1. **Backbone/head split** — when ``backbone_learning_rate`` is set and
           the model exposes ``transferable_components()``, backbone parameters
           get ``backbone_learning_rate`` and all other (head) parameters get
           ``learning_rate``.
        2. **CWT split** — when ``cwt_lr_multiplier != 1.0``, parameters whose
           names contain ``".cwt."`` are grouped with a scaled learning rate.
        3. **Uniform** — all parameters share ``learning_rate``.

        Returns:
            List of param-group dicts suitable for :class:`torch.optim.AdamW`.
        """
        if self.backbone_learning_rate is not None and hasattr(
            self.model, "transferable_components"
        ):
            return self._build_backbone_head_param_groups()

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
            logger.info(
                "CWT LR multiplier: %sx (cwt_lr=%.2e, %d params) | "
                "base_lr=%.2e (%d params)",
                self.cwt_lr_multiplier,
                cwt_lr,
                n_cwt,
                self.learning_rate,
                n_other,
            )

        return groups

    def _build_backbone_head_param_groups(self) -> list[dict]:
        """Build param groups splitting backbone (transferable) from head params.

        Backbone parameters (matched by ``model.transferable_components()``)
        receive ``backbone_learning_rate``; remaining head parameters receive
        ``learning_rate``.

        Returns:
            List of one or two param-group dicts.
        """
        component_prefixes = tuple(
            f"{name}." for name in self.model.transferable_components()
        )
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if name.startswith(component_prefixes):
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": self.backbone_learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )

        n_backbone = sum(
            p.numel()
            for p in backbone_params
            if not p.__class__.__name__.startswith("Uninitialized")
        )
        n_head = sum(
            p.numel()
            for p in head_params
            if not p.__class__.__name__.startswith("Uninitialized")
        )
        logger.info(
            "Discriminative LR: backbone_lr=%.2e (%d params) | "
            "head_lr=%.2e (%d params)",
            self.backbone_learning_rate,
            n_backbone,
            self.learning_rate,
            n_head,
        )
        return groups

    def configure_optimizers(self):
        """Build AdamW optimizer and multi-phase LR scheduler.

        ``scheduler_name="onecycle"`` uses PyTorch's :class:`OneCycleLR` with
        its native defaults, matching NeuralBench's EEGNet recipe. Otherwise,
        constructs a :class:`SequentialLR` with up to three phases: linear
        warmup, constant/cosine hold, and cosine decay.
        """
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups)

        if self.scheduler_name == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                },
            }
        if self.scheduler_name != "phased":
            raise ValueError(
                f"Unknown scheduler_name: {self.scheduler_name}. "
                "Must be 'phased' or 'onecycle'."
            )

        schedulers = []
        milestones = []
        current_step = 0

        # Warmup phase
        if self.warmup > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.start_lr_factor,
                end_factor=1.0,
                total_iters=self.warmup,
            )
            schedulers.append(warmup)
            current_step += self.warmup
            milestones.append(current_step)

        # Hold phase
        if self.hold > 0:
            if self.hold_scheduler_type == "constant":
                hold = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=self.hold,
                )
            elif self.hold_scheduler_type == "cosine":
                hold = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hold
                    / 10,  # 10 is the default cosine annealing period
                    eta_min=self.end_lr_factor * self.learning_rate,
                )
            else:
                raise ValueError(
                    f"Unknown hold_scheduler_type: {self.hold_scheduler_type}. "
                    f"Must be 'constant' or 'cosine'."
                )
            schedulers.append(hold)
            current_step += self.hold
            if self.decay > 0:  # Only add milestone if there's a next phase
                milestones.append(current_step)

        # Decay phase
        if self.decay > 0:
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay,
                eta_min=self.end_lr_factor * self.learning_rate,
            )

            schedulers.append(decay)

        # If no schedulers are active, use a default constant scheduler
        if not schedulers:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0
            )
        elif len(schedulers) == 1:
            # Single scheduler, no need for SequentialLR
            scheduler = schedulers[0]
        else:
            # Multiple schedulers, use SequentialLR
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }

    def on_fit_start(self):
        self._configure_wandb_metric_summaries()

    def _configure_wandb_metric_summaries(self):
        """Register per-task metric summary modes (min/max) with WandB."""
        from lightning.pytorch.loggers import WandbLogger

        if not isinstance(self.logger, WandbLogger):
            return

        experiment = self.logger.experiment
        for prefix in ("train", "val", "test"):
            experiment.define_metric(f"{prefix}/loss", summary="min")

        for name, cfg in self.model.task_configs.items():
            for prefix in ("train", "val", "test"):
                experiment.define_metric(f"{prefix}/{name}_loss", summary="min")
            for metric_name, mode in cfg.metric_summary_modes.items():
                if metric_name == "loss":
                    continue
                for prefix in ("train", "val", "test"):
                    experiment.define_metric(
                        f"{prefix}/{name}_{metric_name}", summary=mode
                    )

        for name in self.model.task_configs:
            cfg = self.model.task_configs[name]
            for metrics_dict in (
                self.train_metrics,
                self.val_metrics,
                self.test_metrics,
            ):
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
        """Separate target/metadata keys from model-input keys in the batch.

        Pops ``target_values``, ``target_weights``, ``session_id``,
        ``absolute_start``, and ``eval_mask`` from *batch* (mutating it
        in-place) so that the remaining dict can be passed directly as
        ``**model_inputs``.

        Returns:
            ``(model_inputs, target_values, target_weights, task_index,
            session_id)`` where *model_inputs* is the modified *batch*.
        """
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
        """Compute sequence-weighted multitask loss.

        Each task's loss is weighted by the number of sequences in the batch
        that contain that task (determined from ``task_index``).  SSL tasks
        (e.g. masked reconstruction) bypass the task-index lookup and use
        the full batch size as their sequence count.

        Args:
            outputs: Per-task prediction tensors from the model.
            target_values: Per-task ground-truth tensors.
            target_weights: Per-task sample weights (tensor or scalar).
            task_index: (B, n_out) padded task indices (0 = padding).
            ssl_task_names: Task names injected by the SSL head, which use
                batch-level weighting instead of per-sequence counting.

        Returns:
            ``(total_loss, taskwise_loss)`` where *total_loss* is the
            sequence-weighted mean and *taskwise_loss* maps task name to
            its individual (unweighted) loss scalar.
        """
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
