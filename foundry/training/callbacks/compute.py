"""Compute accounting callback for training throughput and FLOP tracking."""

from __future__ import annotations

import logging
import time
from typing import Any

import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


class ComputeTrackingCallback(L.Callback):
    """Track processed windows, wall time, and optional FLOPs during training.

    Counts training windows from the actual batch tensor on every training
    batch (including gradient-accumulation micro-batches and short final
    batches). Counters and elapsed wall time persist through
    ``state_dict`` / ``load_state_dict`` so Slurm requeues do not reset them.

    At each real validation epoch logs compute metrics via
    ``pl_module.log_dict``; fit-end metrics are sent directly to the logger,
    because Lightning forbids ``self.log`` in ``on_fit_end``. When
    ``flops_per_window`` or ``session_flops`` is set, also logs
    cumulative FLOPs. Tracks the configured ``monitor`` metric and, at fit
    end, logs best-checkpoint compute stats (verifying against
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` when present).

    Args:
        monitor: Metric key to track for best-checkpoint compute stats
            (e.g. ``"val/neurosoft_acoustic_stim_8band_supported_f1"``).
        mode: ``"min"`` or ``"max"`` for ``monitor`` improvement.
        sequence_length: Window duration in seconds (for signal-seconds accounting).
        flops_per_window: Validated forward+backward FLOPs per training window.
            Mutually exclusive with ``session_flops``.
        flop_method: Identifier for the FLOP validation method/version.
        require_flops: Refuse to start without both validated FLOP fields.
        session_flops: Per-canonical-session FLOPs per window for multi-session
            source pretraining. Maps canonical session ID to FLOPs per window.
            Mutually exclusive with ``flops_per_window``.
        realized_train_windows_per_epoch: Number of train windows per nominal
            epoch (from the source manifest) for effective-epoch computation.
    """

    def __init__(
        self,
        monitor: str,
        mode: str,
        sequence_length: float,
        flops_per_window: int | None = None,
        flop_method: str | None = None,
        require_flops: bool = False,
        session_flops: dict[str, int] | None = None,
        realized_train_windows_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if flops_per_window is not None and session_flops is not None:
            raise ValueError(
                "flops_per_window and session_flops are mutually exclusive"
            )
        self.monitor = monitor
        self.mode = mode
        self.sequence_length = sequence_length
        self.flops_per_window = flops_per_window
        self.flop_method = flop_method
        self.require_flops = require_flops
        self.session_flops = dict(session_flops) if session_flops else None
        self.realized_train_windows_per_epoch = realized_train_windows_per_epoch
        if flops_per_window is not None and flops_per_window <= 0:
            raise ValueError("flops_per_window must be a positive integer")
        if session_flops is not None:
            for sid, flops in session_flops.items():
                if flops <= 0:
                    raise ValueError(
                        f"session_flops values must be positive, got {flops} "
                        f"for session {sid!r}"
                    )
        if require_flops and (
            flops_per_window is None
            and session_flops is None
            or not flop_method
            or not flop_method.strip()
        ):
            raise ValueError(
                "require_flops=True requires validated flops_per_window or "
                "session_flops and a non-empty flop_method"
            )

        self._processed_windows = 0
        self._per_session_windows: dict[str, int] = {}
        self._restored_wall_time_s = 0.0
        self._fit_start_monotonic: float | None = None
        self._last_batch_size = 0

        self._best_monitor_value: float | None = None
        self._best_step = 0
        self._best_examples = 0
        self._best_windows = 0
        self._best_flops: int | None = None
        self._best_wall_time_s = 0.0
        self._best_per_session_windows: dict[str, int] = {}
        self._best_effective_epochs: float | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "processed_windows": self._processed_windows,
            "per_session_windows": dict(self._per_session_windows),
            "elapsed_wall_time_s": self._elapsed_wall_time_s(),
            "best_monitor_value": self._best_monitor_value,
            "best_step": self._best_step,
            "best_examples": self._best_examples,
            "best_windows": self._best_windows,
            "best_flops": self._best_flops,
            "best_wall_time_s": self._best_wall_time_s,
            "best_per_session_windows": dict(self._best_per_session_windows),
            "best_effective_epochs": self._best_effective_epochs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._processed_windows = int(state_dict["processed_windows"])
        self._per_session_windows = {
            str(k): int(v)
            for k, v in state_dict.get("per_session_windows", {}).items()
        }
        self._restored_wall_time_s = float(state_dict["elapsed_wall_time_s"])
        best_value = state_dict.get("best_monitor_value")
        self._best_monitor_value = (
            None if best_value is None else float(best_value)
        )
        self._best_step = int(state_dict.get("best_step", 0))
        self._best_examples = int(state_dict.get("best_examples", 0))
        self._best_windows = int(state_dict.get("best_windows", 0))
        best_flops = state_dict.get("best_flops")
        self._best_flops = None if best_flops is None else int(best_flops)
        self._best_wall_time_s = float(state_dict.get("best_wall_time_s", 0.0))
        self._best_per_session_windows = {
            str(k): int(v)
            for k, v in state_dict.get("best_per_session_windows", {}).items()
        }
        best_ee = state_dict.get("best_effective_epochs")
        self._best_effective_epochs = (
            None if best_ee is None else float(best_ee)
        )
        self._fit_start_monotonic = None

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._fit_start_monotonic = time.monotonic()
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(
                self._build_compute_metadata(trainer)
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        batch_windows = self._count_batch_windows(batch)
        self._processed_windows += batch_windows
        self._last_batch_size = batch_windows

        if self.session_flops is not None:
            session_ids = self._extract_session_ids(batch)
            if session_ids is not None:
                for sid in session_ids:
                    self._per_session_windows[sid] = (
                        self._per_session_windows.get(sid, 0) + 1
                    )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        self._log_compute_metrics(trainer, pl_module)
        self._maybe_update_best(trainer)

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._log_compute_metrics(trainer, pl_module, at_fit_end=True)
        self._verify_and_log_best(trainer)

    @staticmethod
    def _count_batch_windows(batch: Any) -> int:
        if isinstance(batch, dict) and "task_index" in batch:
            return int(batch["task_index"].shape[0])
        task_index = getattr(batch, "task_index", None)
        if task_index is not None:
            return int(task_index.shape[0])
        raise TypeError(
            "ComputeTrackingCallback expected batch with 'task_index' tensor"
        )

    def _elapsed_wall_time_s(self) -> float:
        elapsed = self._restored_wall_time_s
        if self._fit_start_monotonic is not None:
            elapsed += time.monotonic() - self._fit_start_monotonic
        return elapsed

    def _global_windows(self, trainer: Trainer) -> int:
        return self._processed_windows * trainer.world_size

    def _global_examples(self, trainer: Trainer) -> int:
        return self._global_windows(trainer)

    def _cumulative_flops(self, trainer: Trainer) -> int | None:
        if self.session_flops is not None:
            world_size = trainer.world_size
            total = 0
            for sid, count in self._per_session_windows.items():
                flops = self.session_flops.get(sid, 0)
                total += count * world_size * flops
            return total if total > 0 else None
        if self.flops_per_window is not None:
            return self._global_windows(trainer) * self.flops_per_window
        return None

    def _effective_epochs(self, trainer: Trainer) -> float | None:
        if self.realized_train_windows_per_epoch is None:
            return None
        if self.realized_train_windows_per_epoch <= 0:
            return None
        return (
            self._global_windows(trainer)
            / self.realized_train_windows_per_epoch
        )

    @staticmethod
    def _extract_session_ids(batch: Any) -> list[str] | None:
        """Extract per-item canonical session IDs from the batch."""
        if isinstance(batch, dict):
            sids = batch.get("input_session_ids")
            if sids is not None:
                return list(sids)
        sids = getattr(batch, "input_session_ids", None)
        if sids is not None:
            return list(sids)
        return None

    def get_compute_snapshot(self, trainer: Trainer) -> dict[str, Any]:
        """Return a snapshot of current compute counters for manifest emission."""
        return {
            "processed_windows": self._global_windows(trainer),
            "processed_examples": self._global_examples(trainer),
            "signal_seconds": self._global_windows(trainer)
            * self.sequence_length,
            "wall_time_seconds": self._elapsed_wall_time_s(),
            "cumulative_flops": self._cumulative_flops(trainer),
            "flop_method": self.flop_method,
            "optimizer_steps": trainer.global_step,
            "effective_epochs": self._effective_epochs(trainer),
            "gpu": self._gpu_model_name(),
            "gpu_compute_capability": self._gpu_compute_capability(),
            "precision": str(trainer.precision),
            "per_session_windows": dict(self._per_session_windows),
        }

    def get_best_compute_snapshot(self) -> dict[str, Any]:
        """Return compute counters at the best checkpoint step."""
        effective_epochs = self._best_effective_epochs
        if (
            effective_epochs is None
            and self.realized_train_windows_per_epoch
            and self.realized_train_windows_per_epoch > 0
        ):
            effective_epochs = (
                self._best_windows / self.realized_train_windows_per_epoch
            )
        return {
            "processed_windows": self._best_windows,
            "processed_examples": self._best_examples,
            "signal_seconds": self._best_windows * self.sequence_length,
            "wall_time_seconds": self._best_wall_time_s,
            "cumulative_flops": self._best_flops,
            "flop_method": self.flop_method,
            "optimizer_steps": self._best_step,
            "effective_epochs": effective_epochs,
            "gpu": self._gpu_model_name(),
            "gpu_compute_capability": self._gpu_compute_capability(),
            "precision": "unknown",
            "per_session_windows": dict(self._best_per_session_windows),
            "monitor_value": self._best_monitor_value,
        }

    def _resolve_model(self, pl_module: L.LightningModule) -> torch.nn.Module:
        return pl_module.model if hasattr(pl_module, "model") else pl_module

    def _get_batch_size(self, trainer: Trainer) -> int:
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "batch_size"):
            return int(datamodule.batch_size)
        return max(self._last_batch_size, 1)

    @staticmethod
    def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = 0
        trainable = 0
        for param in model.parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
        return total, trainable

    @staticmethod
    def _gpu_model_name() -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "cpu"

    @staticmethod
    def _gpu_compute_capability() -> str:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
        return "n/a"

    @staticmethod
    def _peak_memory_gb() -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.max_memory_allocated() / 1e9,
            torch.cuda.max_memory_reserved() / 1e9,
        )

    def _build_compute_metadata(self, trainer: Trainer) -> dict[str, str]:
        """Return non-scalar compute metadata for logger hyperparameters."""
        metadata = {
            "compute/gpu_model": self._gpu_model_name(),
            "compute/gpu_compute_capability": self._gpu_compute_capability(),
            "compute/precision": str(trainer.precision),
        }
        if self.flop_method is not None:
            metadata["compute/flop_method"] = self.flop_method
        return metadata

    def _build_compute_metrics(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> dict[str, float | int]:
        model = self._resolve_model(pl_module)
        total_parameters, trainable_parameters = self._count_parameters(model)
        batch_size = self._get_batch_size(trainer)
        world_size = trainer.world_size
        accumulate_grad_batches = int(trainer.accumulate_grad_batches)
        processed_windows = self._global_windows(trainer)
        processed_examples = self._global_examples(trainer)
        peak_allocated_gb, peak_reserved_gb = self._peak_memory_gb()

        metrics: dict[str, float | int] = {
            "compute/optimizer_steps": trainer.global_step,
            "compute/processed_windows": processed_windows,
            "compute/processed_examples": processed_examples,
            "compute/signal_seconds": processed_windows * self.sequence_length,
            "compute/elapsed_wall_time_s": self._elapsed_wall_time_s(),
            "compute/epoch": float(trainer.current_epoch),
            "compute/world_size": world_size,
            "compute/effective_batch_size": (
                batch_size * world_size * accumulate_grad_batches
            ),
            "compute/total_parameters": total_parameters,
            "compute/trainable_parameters": trainable_parameters,
            "compute/peak_memory_allocated_gb": peak_allocated_gb,
            "compute/peak_memory_reserved_gb": peak_reserved_gb,
        }

        cumulative_flops = self._cumulative_flops(trainer)
        if cumulative_flops is not None:
            metrics["compute/cumulative_flops"] = cumulative_flops
            if self.flops_per_window is not None:
                metrics["compute/flops_per_window"] = self.flops_per_window

        effective_epochs = self._effective_epochs(trainer)
        if effective_epochs is not None:
            metrics["compute/effective_epochs"] = effective_epochs

        return metrics

    def _log_compute_metrics(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        *,
        at_fit_end: bool = False,
    ) -> None:
        metrics = self._build_compute_metrics(trainer, pl_module)
        if at_fit_end:
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
        else:
            pl_module.log_dict(
                metrics,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        log.info(
            "ComputeTracking: logged compute metrics at step=%d "
            "(processed_windows=%d, elapsed_wall_time_s=%.1f)",
            trainer.global_step,
            metrics["compute/processed_windows"],
            metrics["compute/elapsed_wall_time_s"],
        )

    @staticmethod
    def _metric_value(trainer: Trainer, monitor: str) -> float | None:
        value = trainer.callback_metrics.get(monitor)
        if value is None:
            value = trainer.logged_metrics.get(monitor)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.item())
        return float(value)

    def _is_improvement(self, current: float) -> bool:
        if self._best_monitor_value is None:
            return True
        if self.mode == "max":
            return current > self._best_monitor_value
        return current < self._best_monitor_value

    def _maybe_update_best(self, trainer: Trainer) -> None:
        current = self._metric_value(trainer, self.monitor)
        if current is None:
            log.debug(
                "ComputeTracking: monitor metric %r not available at step=%d",
                self.monitor,
                trainer.global_step,
            )
            return
        if not self._is_improvement(current):
            return

        self._best_monitor_value = current
        self._best_step = trainer.global_step
        self._best_examples = self._global_examples(trainer)
        self._best_windows = self._global_windows(trainer)
        self._best_flops = self._cumulative_flops(trainer)
        self._best_wall_time_s = self._elapsed_wall_time_s()
        self._best_per_session_windows = dict(self._per_session_windows)
        self._best_effective_epochs = self._effective_epochs(trainer)
        log.info(
            "ComputeTracking: new best %s=%.6f at step=%d",
            self.monitor,
            current,
            trainer.global_step,
        )

    @staticmethod
    def _find_model_checkpoint(trainer: Trainer) -> ModelCheckpoint | None:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback
        return None

    def _verify_and_log_best(self, trainer: Trainer) -> None:
        checkpoint_cb = self._find_model_checkpoint(trainer)
        if (
            checkpoint_cb is not None
            and checkpoint_cb.best_model_score is not None
            and self._best_monitor_value is not None
        ):
            ckpt_score = checkpoint_cb.best_model_score
            if torch.is_tensor(ckpt_score):
                ckpt_score = float(ckpt_score.item())
            else:
                ckpt_score = float(ckpt_score)
            if abs(ckpt_score - self._best_monitor_value) > 1e-6:
                log.warning(
                    "ComputeTracking: best monitor value %.6f does not match "
                    "ModelCheckpoint.best_model_score %.6f for %r",
                    self._best_monitor_value,
                    ckpt_score,
                    self.monitor,
                )

        if self._best_monitor_value is None:
            log.info(
                "ComputeTracking: no best-checkpoint compute stats "
                "(monitor %r never improved)",
                self.monitor,
            )
            return

        best_metrics: dict[str, float | int] = {
            "compute/best_step": self._best_step,
            "compute/best_examples": self._best_examples,
            "compute/best_windows": self._best_windows,
            "compute/best_wall_time_s": self._best_wall_time_s,
            "compute/best_monitor_value": self._best_monitor_value,
        }
        if self._best_flops is not None:
            best_metrics["compute/best_flops"] = self._best_flops
        if self._best_effective_epochs is not None:
            best_metrics["compute/best_effective_epochs"] = (
                self._best_effective_epochs
            )

        if trainer.logger is not None:
            trainer.logger.log_metrics(best_metrics, step=trainer.global_step)
        log.info(
            "ComputeTracking: logged best-checkpoint compute stats "
            "(%s=%.6f at step=%d)",
            self.monitor,
            self._best_monitor_value,
            self._best_step,
        )
