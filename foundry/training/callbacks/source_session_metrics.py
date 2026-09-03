"""Source-session aggregate validation metrics for pretraining checkpoint selection."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from lightning import Trainer
from torchmetrics.classification import F1Score

from foundry.training.step_output import extract_step_output

log = logging.getLogger(__name__)


class SourceSessionMetricsCallback(L.Callback):
    """Aggregate per-source-session supported-class macro-F1 for checkpoint selection.

    During multi-session source pretraining, validation batches contain items
    from many source sessions.  This callback buffers predictions and targets
    per session across validation batches, computes supported-class macro-F1
    independently for each session, and logs the unweighted arithmetic mean of
    those session scores via ``pl_module.log()`` so that EarlyStopping and
    ModelCheckpoint can monitor ``metric_key`` through ``trainer.callback_metrics``.

    Per-session breakdowns are logged separately through ``trainer.logger`` for
    analysis.  Distributed validation is not supported: each rank would only
    see a subset of sessions, so gathering is required but not yet implemented.
    """

    def __init__(
        self,
        monitor_task: str,
        num_classes: int,
        metric_key: str = "val/source_session_mean_supported_f1",
    ) -> None:
        super().__init__()
        self.monitor_task = monitor_task
        self.num_classes = num_classes
        self.metric_key = metric_key
        self._val_session_buffers: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._latest_session_scores: dict[str, float] = {}
        self._latest_mean_f1: float | None = None
        self._best_mean_f1: float | None = None
        self._best_session_scores: dict[str, float] = {}

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.world_size > 1:
            raise NotImplementedError(
                "SourceSessionMetricsCallback does not support distributed "
                "validation; per-session prediction gather is not implemented."
            )
        self._clear_buffers()

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

        preds = step_output.task_outputs.get(self.monitor_task)
        target = step_output.target_values.get(self.monitor_task)
        if preds is None or target is None or target.numel() == 0:
            return

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        self._accumulate_session_preds(
            preds,
            target,
            step_output.task_index,
            step_output.session_id,
            model.router,
        )

    def _accumulate_session_preds(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        task_index: torch.Tensor,
        session_id: list[str],
        router,
    ) -> None:
        """Buffer per-session predictions/targets for epoch-end metric computation."""
        router_idx = router.get_task_index_by_name(self.monitor_task) + 1
        counts = (task_index == router_idx).sum(dim=1)

        per_item_preds = torch.split(preds, counts.tolist())
        per_item_targets = torch.split(target, counts.tolist())

        for sid, item_p, item_t in zip(
            session_id, per_item_preds, per_item_targets
        ):
            if item_p.numel() == 0:
                continue
            if sid not in self._val_session_buffers:
                self._val_session_buffers[sid] = {"preds": [], "targets": []}
            self._val_session_buffers[sid]["preds"].append(item_p.detach().cpu())
            self._val_session_buffers[sid]["targets"].append(item_t.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            self._clear_buffers()
            return

        if not self._val_session_buffers:
            self._clear_buffers()
            raise RuntimeError(
                "SourceSessionMetrics received no validation predictions for "
                f"task {self.monitor_task!r}; cannot select a source checkpoint"
            )

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        cfg = model.task_configs.get(self.monitor_task)
        if cfg is None:
            self._clear_buffers()
            raise RuntimeError(
                "SourceSessionMetrics monitor task is not configured: "
                f"{self.monitor_task!r}"
            )

        session_f1_values: list[float] = []
        epoch_scores: dict[str, float] = {}
        logger_metrics: dict[str, Any] = {}

        for session_id, data in self._val_session_buffers.items():
            preds = torch.cat(data["preds"])
            targets = torch.cat(data["targets"])
            short = self._shorten_session_id(session_id)

            had_predictions = preds.numel() > 0
            valid = targets >= 0
            if not valid.all():
                preds = preds[valid]
                targets = targets[valid]

            if targets.numel() == 0:
                self._clear_buffers()
                raise RuntimeError(
                    "SourceSessionMetrics found an undefined validation session "
                    f"for {session_id!r}: predictions={had_predictions}, "
                    "valid targets=0"
                )

            metric_preds, metric_targets = pl_module._prepare_for_metrics(
                cfg, preds, targets
            )
            session_result = self._compute_supported_f1(
                metric_preds, metric_targets
            )
            if session_result is None:
                self._clear_buffers()
                raise RuntimeError(
                    "SourceSessionMetrics found no supported classes for "
                    f"validation session {session_id!r}"
                )

            supported_f1, per_class_f1, support, class_mask = session_result
            session_f1_values.append(supported_f1)
            epoch_scores[session_id] = supported_f1

            logger_metrics[f"val/source_session/{short}/supported_f1"] = (
                supported_f1
            )
            logger_metrics[f"val/source_session/{short}/class_mask"] = (
                class_mask.tolist()
            )
            logger_metrics[f"val/source_session/{short}/support"] = (
                support.tolist()
            )
            for class_idx, value in enumerate(per_class_f1.tolist()):
                logger_metrics[
                    f"val/source_session/{short}/f1_class_{class_idx}"
                ] = value

        session_count = len(session_f1_values)
        if session_count == 0:
            self._clear_buffers()
            raise RuntimeError(
                "SourceSessionMetrics produced no valid source-session F1 "
                f"values for task {self.monitor_task!r} at epoch "
                f"{trainer.current_epoch}"
            )
        mean_f1 = sum(session_f1_values) / session_count

        self._latest_session_scores = epoch_scores
        self._latest_mean_f1 = mean_f1
        if self._best_mean_f1 is None or mean_f1 > self._best_mean_f1:
            self._best_mean_f1 = mean_f1
            self._best_session_scores = dict(epoch_scores)

        pl_module.log(
            self.metric_key,
            mean_f1,
            logger=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )
        pl_module.log(
            "val/source_session_count",
            float(session_count),
            logger=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )
        log.info(
            "SourceSessionMetrics: %s=%.4f over %d sessions (task=%r, epoch=%d).",
            self.metric_key,
            mean_f1,
            session_count,
            self.monitor_task,
            trainer.current_epoch,
        )

        logger_metrics["val/source_session_count"] = session_count
        if logger_metrics and trainer.logger is not None:
            trainer.logger.log_metrics(
                logger_metrics, step=trainer.current_epoch
            )

        self._clear_buffers()

    def _compute_supported_f1(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return supported macro-F1 and per-class diagnostics for one session."""
        support = torch.bincount(
            targets.reshape(-1), minlength=self.num_classes
        )
        class_mask = support > 0
        if not class_mask.any():
            return None

        per_class_f1 = F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average=None,
        )(preds, targets)
        supported_f1 = per_class_f1[class_mask].mean().item()
        return supported_f1, per_class_f1, support, class_mask

    def _clear_buffers(self) -> None:
        self._val_session_buffers = {}

    @staticmethod
    def _shorten_session_id(session_id: str) -> str:
        """Keep only subject, session, and acquisition segments."""
        namespace, separator, raw_id = session_id.partition(":")
        parts = raw_id.split("_") if separator else session_id.split("_")
        keep = [p for p in parts if p.startswith(("sub-", "ses-", "acq-"))]
        short = "_".join(keep) if keep else raw_id
        return f"{namespace}:{short}" if separator else short
