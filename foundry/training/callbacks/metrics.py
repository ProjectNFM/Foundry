"""Session-level and confusion matrix metric callbacks."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from lightning import Trainer

from foundry.training.confusion_matrix import ConfusionMatrixTracker
from foundry.training.step_output import extract_step_output

log = logging.getLogger(__name__)


class SessionMetricsCallback(L.Callback):
    """Log per-session (per-recording) validation metrics at epoch end.

    Accumulation buffers live on this callback instance, not on the
    Lightning module.  During each validation batch the callback splits
    flat predictions back to per-item chunks (using ``task_index``) and
    groups them by ``session_id``.  At epoch end it instantiates fresh
    metric collections per session, computes results, and logs them.

    Logged metric keys follow the pattern::

        val_session/{short_session_id}/{task_name}_{metric}

    where ``short_session_id`` strips the ``task-*`` and ``desc-*``
    segments from the full BIDS-style recording ID for readability.
    """

    def __init__(self) -> None:
        super().__init__()
        self._val_session_buffers: dict = {}

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._val_session_buffers = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        step_output = extract_step_output(outputs)
        if step_output is None or step_output.session_id is None:
            return

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        for name, cfg in model.task_configs.items():
            if cfg.metrics is None or name in step_output.ssl_task_names:
                continue
            preds = step_output.task_outputs.get(name)
            target = step_output.target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue
            self._accumulate_session_preds(
                name,
                preds,
                target,
                step_output.task_index,
                step_output.session_id,
                model.router,
            )

    def _accumulate_session_preds(
        self,
        task_name: str,
        preds: torch.Tensor,
        target: torch.Tensor,
        task_index: torch.Tensor,
        session_id: list[str],
        router,
    ) -> None:
        """Buffer per-session predictions/targets for epoch-end metric computation."""
        router_idx = router.get_task_index_by_name(task_name) + 1
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

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if not self._val_session_buffers:
            return

        all_metrics: dict[str, float] = {}
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        for task_name, session_data in self._val_session_buffers.items():
            cfg = model.task_configs.get(task_name)
            if cfg is None or cfg.metrics is None:
                continue

            for session_id, data in session_data.items():
                preds = torch.cat(data["preds"])
                targets = torch.cat(data["targets"])

                if cfg.kind in ("binary", "multiclass"):
                    valid = targets >= 0
                    if not valid.all():
                        preds = preds[valid]
                        targets = targets[valid]

                if targets.numel() == 0:
                    continue

                try:
                    metric_collection = instantiate(cfg.metrics)
                    metric_preds, metric_targets = (
                        pl_module._prepare_for_metrics(cfg, preds, targets)
                    )
                    metric_collection.update(metric_preds, metric_targets)
                    result = metric_collection.compute()
                except Exception:
                    log.debug(
                        "SessionMetrics: failed to compute metrics for "
                        "task=%s session=%s, skipping.",
                        task_name,
                        session_id,
                        exc_info=True,
                    )
                    continue

                short = self._shorten_session_id(session_id)
                for metric_name, value in result.items():
                    key = f"val_session/{short}/{task_name}_{metric_name}"
                    all_metrics[key] = (
                        value.item() if torch.is_tensor(value) else value
                    )

        if all_metrics and trainer.logger is not None:
            trainer.logger.log_metrics(all_metrics, step=trainer.current_epoch)

        self._val_session_buffers = {}

    @staticmethod
    def _shorten_session_id(session_id: str) -> str:
        """Keep only subject, session, and acquisition segments."""
        parts = session_id.split("_")
        keep = [p for p in parts if p.startswith(("sub-", "ses-", "acq-"))]
        return "_".join(keep) if keep else session_id


class SourceMetricsCallback(L.Callback):
    """Log validation metrics and confusion matrices separately by source.

    Samples are grouped using ``StepOutput.source_id`` while model parameters
    and readout heads remain shared. Logged keys follow
    ``val/{source_id}/{task_name}_{metric}``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._val_source_buffers: dict = {}

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._val_source_buffers = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        step_output = extract_step_output(outputs)
        if step_output is None or step_output.source_id is None:
            return

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        for task_name, cfg in model.task_configs.items():
            if cfg.metrics is None or task_name in step_output.ssl_task_names:
                continue
            preds = step_output.task_outputs.get(task_name)
            targets = step_output.target_values.get(task_name)
            if preds is None or targets is None or targets.numel() == 0:
                continue
            self._accumulate_source_preds(
                task_name,
                preds,
                targets,
                step_output.task_index,
                step_output.source_id,
                model.router,
            )

    def _accumulate_source_preds(
        self,
        task_name: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
        task_index: torch.Tensor,
        source_ids: list[str],
        router,
    ) -> None:
        """Split flat task outputs by sample and buffer them by source."""
        if len(source_ids) != task_index.shape[0]:
            raise ValueError(
                "Expected one source ID per validation sample, got "
                f"{len(source_ids)} IDs for batch size {task_index.shape[0]}"
            )

        router_idx = router.get_task_index_by_name(task_name) + 1
        counts = (task_index == router_idx).sum(dim=1)
        per_item_preds = torch.split(preds, counts.tolist())
        per_item_targets = torch.split(targets, counts.tolist())
        task_buffer = self._val_source_buffers.setdefault(task_name, {})

        for source_id, item_preds, item_targets in zip(
            source_ids, per_item_preds, per_item_targets
        ):
            if item_preds.numel() == 0:
                continue
            source_buffer = task_buffer.setdefault(
                str(source_id), {"preds": [], "targets": []}
            )
            source_buffer["preds"].append(item_preds.detach().cpu())
            source_buffer["targets"].append(item_targets.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if not self._val_source_buffers:
            return

        from foundry.training.callbacks import get_wandb_experiment

        all_metrics: dict[str, Any] = {}
        confusion_matrices = []
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        for task_name, source_data in self._val_source_buffers.items():
            cfg = model.task_configs.get(task_name)
            if cfg is None or cfg.metrics is None:
                continue

            for source_id, data in source_data.items():
                preds = torch.cat(data["preds"])
                targets = torch.cat(data["targets"])
                if cfg.kind in ("binary", "multiclass"):
                    valid = targets >= 0
                    preds = preds[valid]
                    targets = targets[valid]
                if targets.numel() == 0:
                    continue

                try:
                    metric_collection = instantiate(cfg.metrics)
                    metric_preds, metric_targets = (
                        pl_module._prepare_for_metrics(cfg, preds, targets)
                    )
                    metric_collection.update(metric_preds, metric_targets)
                    results = metric_collection.compute()
                except Exception:
                    log.debug(
                        "SourceMetrics: failed to compute metrics for "
                        "task=%s source=%s, skipping.",
                        task_name,
                        source_id,
                        exc_info=True,
                    )
                    continue

                prefix = f"val/{source_id}/{task_name}"
                for metric_name, value in results.items():
                    all_metrics[f"{prefix}_{metric_name}"] = (
                        value.item() if torch.is_tensor(value) else value
                    )

                if cfg.kind in ("binary", "multiclass"):
                    tracker = ConfusionMatrixTracker(
                        num_classes=cfg.output_dim,
                        class_names=cfg.get_class_names(),
                    )
                    if cfg.kind == "multiclass":
                        predicted_classes = preds.argmax(dim=-1)
                    else:
                        predicted_classes = (preds[:, 1] > preds[:, 0]).long()
                    tracker.update(predicted_classes, targets)
                    counts, normalized = tracker.compute()
                    all_metrics[f"{prefix}_confusion_counts"] = counts.tolist()
                    all_metrics[f"{prefix}_confusion_normalized"] = (
                        normalized.tolist()
                    )
                    all_metrics[f"{prefix}_confusion_class_names"] = (
                        tracker.class_names
                    )
                    confusion_matrices.append(
                        (source_id, task_name, tracker, counts, normalized)
                    )

        if all_metrics and trainer.logger is not None:
            trainer.logger.log_metrics(all_metrics, step=trainer.current_epoch)

        wandb_experiment = get_wandb_experiment(trainer)
        if wandb_experiment is not None:
            for (
                source_id,
                task_name,
                tracker,
                counts,
                normalized,
            ) in confusion_matrices:
                tracker.log_wandb(
                    wandb_experiment,
                    f"{source_id}/{task_name}",
                    trainer.current_epoch,
                    counts,
                    normalized,
                )

        self._val_source_buffers = {}


class ConfusionMatrixCallback(L.Callback):
    """Log confusion matrices for classification tasks at validation epoch end.

    Reads ``pl_module._val_confusion_trackers`` (populated by
    :class:`~foundry.training.module.FoundryModule` during validation steps)
    and handles all logging — generic metric payloads plus W&B-native
    interactive visualizations.
    """

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        trackers: dict[str, ConfusionMatrixTracker] = getattr(
            pl_module, "_val_confusion_trackers", {}
        )
        if not trackers:
            return

        from foundry.training.callbacks import get_wandb_experiment

        wandb_experiment = get_wandb_experiment(trainer)

        for name, tracker in trackers.items():
            counts, normalized = tracker.compute()
            if counts.sum() == 0:
                tracker.reset()
                continue

            payload = {
                f"val/{name}_confusion_counts": counts.tolist(),
                f"val/{name}_confusion_normalized": normalized.tolist(),
                f"val/{name}_confusion_class_names": tracker.class_names,
            }

            if trainer.logger is not None:
                trainer.logger.log_metrics(payload, step=trainer.current_epoch)

            if wandb_experiment is not None:
                tracker.log_wandb(
                    wandb_experiment,
                    name,
                    trainer.current_epoch,
                    counts,
                    normalized,
                )

            tracker.reset()
