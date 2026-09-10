"""Low-overhead metrics for controlled local training benchmarks."""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning import Trainer

logger = logging.getLogger(__name__)
_PROCESS_STARTED_AT = time.perf_counter()


class StepPerformanceBenchmarkCallback(L.Callback):
    """Record synchronized post-warmup step latency and memory to JSON."""

    def __init__(
        self, warmup_steps: int = 25, filename: str = "benchmark.json"
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = warmup_steps
        self.filename = filename
        self._fit_started_at = 0.0
        self._step_started_at = 0.0
        self._step_times: list[float] = []
        self._finite_losses = True

    @staticmethod
    def _synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._fit_started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._synchronize()
        self._step_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._synchronize()
        elapsed = time.perf_counter() - self._step_started_at
        if trainer.global_step > self.warmup_steps:
            self._step_times.append(elapsed)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor):
            self._finite_losses &= bool(torch.isfinite(loss.detach()).all())

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        samples = np.asarray(self._step_times, dtype=np.float64)
        gpu_name = "cpu"
        capability = None
        allocated_gb = reserved_gb = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            major, minor = torch.cuda.get_device_capability()
            capability = f"{major}.{minor}"
            allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        report = {
            "startup_time_s": self._fit_started_at - _PROCESS_STARTED_AT,
            "training_time_s": time.perf_counter() - self._fit_started_at,
            "measured_steps": len(samples),
            "warmup_steps": self.warmup_steps,
            "median_step_time_s": (
                statistics.median(samples.tolist()) if len(samples) else None
            ),
            "p95_step_time_s": (
                float(np.percentile(samples, 95)) if len(samples) else None
            ),
            "peak_memory_allocated_gb": allocated_gb,
            "peak_memory_reserved_gb": reserved_gb,
            "effective_precision": str(trainer.precision),
            "gpu": gpu_name,
            "gpu_compute_capability": capability,
            "finite_losses": self._finite_losses,
        }
        output_path = Path(trainer.default_root_dir) / self.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(report, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, output_path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        logger.info(
            "Wrote controlled benchmark metrics to %s: %s", output_path, report
        )
