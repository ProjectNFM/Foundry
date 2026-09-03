"""Optimizer-step milestone checkpoint callback for NeuroSoft pretraining."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import lightning as L
from lightning import Trainer

log = logging.getLogger(__name__)

DEFAULT_MILESTONE_FRACTIONS = (0.01, 0.03, 0.10, 0.30, 1.00)


class ComputeMilestoneCheckpointCallback(L.Callback):
    """Save model checkpoints at fixed fractions of ``trainer.max_steps``.

    Milestone optimizer steps are computed with ``round(fraction * max_steps)``,
    deduplicated when rounding causes collisions (common at very small budgets),
    and recorded with their realized percentages. Checkpoints are written
    atomically and callback state survives Slurm requeues so milestones are
    neither lost nor overwritten.

    Args:
        milestone_fractions: Fractions of ``trainer.max_steps`` at which to save.
        checkpoint_dir: Directory for milestone checkpoints. Defaults to the
            trainer's :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
            ``dirpath`` when present, otherwise ``<default_root_dir>/checkpoints``.
    """

    def __init__(
        self,
        milestone_fractions: list[float] | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        super().__init__()
        fractions = (
            list(DEFAULT_MILESTONE_FRACTIONS)
            if milestone_fractions is None
            else list(milestone_fractions)
        )
        if not fractions:
            raise ValueError("milestone_fractions must contain at least one value")
        for fraction in fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    "milestone_fractions entries must satisfy 0 < fraction <= 1, "
                    f"got {fraction!r}"
                )
        self.milestone_fractions = fractions
        self.checkpoint_dir = checkpoint_dir

        self._milestone_steps: dict[int, dict[str, Any]] = {}
        self._max_steps_at_init: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "milestone_steps": {
                step: dict(info) for step, info in self._milestone_steps.items()
            },
            "max_steps_at_init": self._max_steps_at_init,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raw_steps = state_dict.get("milestone_steps", {})
        self._milestone_steps = {
            int(step): dict(info) for step, info in raw_steps.items()
        }
        max_steps = state_dict.get("max_steps_at_init")
        self._max_steps_at_init = None if max_steps is None else int(max_steps)

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        max_steps = trainer.max_steps
        if max_steps is None or max_steps <= 0:
            log.warning(
                "ComputeMilestoneCheckpointCallback: trainer.max_steps=%r; "
                "milestone schedule disabled",
                max_steps,
            )
            return

        schedule = self._build_milestone_schedule(int(max_steps))
        if (
            self._max_steps_at_init == int(max_steps)
            and self._milestone_steps
        ):
            for step, info in schedule.items():
                prior = self._milestone_steps.get(step)
                if prior is not None and prior.get("saved"):
                    info["saved"] = True
                    info["path"] = prior.get("path")
                    info["not_reached"] = False
        self._milestone_steps = schedule
        self._max_steps_at_init = int(max_steps)
        self._log_schedule(int(max_steps))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._milestone_steps or not trainer.is_global_zero:
            return

        step = int(trainer.global_step)
        info = self._milestone_steps.get(step)
        if info is None or info.get("saved"):
            return

        checkpoint_dir = self._resolve_checkpoint_dir(trainer)
        realized_pct = float(info["realized_pct"])
        filename = f"milestone-{realized_pct:.0f}pct-step{step}.ckpt"
        destination = checkpoint_dir / filename
        self._save_checkpoint_atomically(trainer, destination)
        info["saved"] = True
        info["not_reached"] = False
        info["path"] = str(destination)
        log.info(
            "ComputeMilestoneCheckpointCallback: saved %s at optimizer step %d "
            "(%.2f%% of max_steps=%d)",
            destination,
            step,
            realized_pct,
            self._max_steps_at_init,
        )

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        if not self._milestone_steps:
            return

        saved: list[str] = []
        not_reached: list[str] = []
        for step, info in sorted(self._milestone_steps.items()):
            if info.get("saved"):
                saved.append(
                    f"step={step} ({info['realized_pct']:.2f}%) -> {info['path']}"
                )
                continue
            info["not_reached"] = True
            not_reached.append(
                f"step={step} ({info['realized_pct']:.2f}%, "
                f"requested={info['fraction']:.2%})"
            )

        log.info(
            "ComputeMilestoneCheckpointCallback summary: %d saved, %d not_reached",
            len(saved),
            len(not_reached),
        )
        for line in saved:
            log.info("  saved: %s", line)
        for line in not_reached:
            log.info("  not_reached: %s", line)

    def _build_milestone_schedule(self, max_steps: int) -> dict[int, dict[str, Any]]:
        schedule: dict[int, dict[str, Any]] = {}
        seen_steps: set[int] = set()
        for fraction in self.milestone_fractions:
            step = round(fraction * max_steps)
            if step <= 0:
                continue
            step = min(step, max_steps)
            if step in seen_steps:
                continue
            seen_steps.add(step)
            realized_pct = (step / max_steps) * 100.0
            schedule[step] = {
                "fraction": float(fraction),
                "realized_pct": realized_pct,
                "saved": False,
                "not_reached": False,
                "path": None,
            }
        return schedule

    def _log_schedule(self, max_steps: int) -> None:
        entries = []
        for step, info in sorted(self._milestone_steps.items()):
            entries.append(
                f"step={step} "
                f"(requested={info['fraction']:.2%}, "
                f"realized={info['realized_pct']:.2f}%)"
            )
        log.info(
            "ComputeMilestoneCheckpointCallback schedule for max_steps=%d: %s",
            max_steps,
            "; ".join(entries) if entries else "(empty)",
        )

    def _resolve_checkpoint_dir(self, trainer: Trainer) -> Path:
        if self.checkpoint_dir is not None:
            directory = Path(self.checkpoint_dir)
        else:
            from lightning.pytorch.callbacks import ModelCheckpoint

            directory = None
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint) and callback.dirpath:
                    directory = Path(callback.dirpath)
                    break
            if directory is None:
                directory = Path(trainer.default_root_dir) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _save_checkpoint_atomically(
        trainer: Trainer, destination: Path
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temp = Path(temp_path)
        try:
            trainer.save_checkpoint(str(temp))
            os.replace(temp, destination)
        finally:
            if temp.exists():
                temp.unlink()
