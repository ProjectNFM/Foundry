"""Checkpoint/compute tests for WP4 (compute milestones and checkpoint manifests).

Covers:
- milestone rounding, deduplication, naming, and exact global-step trigger
- gradient accumulation counts optimizer steps correctly
- requeue/resume preserves counters and avoids duplicate milestone writes
- best checkpoint score/counters match the monitored aggregate
- effective epochs use processed windows / realized train windows per epoch
- per-session FLOP sums match a hand calculation and counters are monotonic
- checkpoint and source-manifest hashes detect tampering
- Markdown is generated from JSON and shows source data plus epochs
- an unreached milestone is explicit
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from foundry.training.callbacks.compute import ComputeTrackingCallback
from foundry.training.callbacks.compute_milestone import (
    DEFAULT_MILESTONE_FRACTIONS,
    ComputeMilestoneCheckpointCallback,
)
from foundry.training.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_MANIFEST_VERSION,
    CheckpointManifestError,
    CheckpointManifestWriter,
    generate_checkpoint_markdown,
    load_checkpoint_manifest,
    verify_checkpoint_integrity,
    write_checkpoint_manifest,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _mock_trainer(
    global_step: int = 0,
    max_steps: int = 500,
    current_epoch: int = 0,
    world_size: int = 1,
    is_global_zero: bool = True,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    callbacks: list | None = None,
    default_root_dir: str = "/tmp/test",
) -> MagicMock:
    trainer = MagicMock(spec_set=[
        "global_step", "max_steps", "current_epoch", "world_size",
        "is_global_zero", "precision", "accumulate_grad_batches",
        "callbacks", "default_root_dir", "callback_metrics",
        "logged_metrics", "logger", "save_checkpoint", "datamodule",
        "sanity_checking",
    ])
    trainer.global_step = global_step
    trainer.max_steps = max_steps
    trainer.current_epoch = current_epoch
    trainer.world_size = world_size
    trainer.is_global_zero = is_global_zero
    trainer.precision = precision
    trainer.accumulate_grad_batches = accumulate_grad_batches
    trainer.callbacks = callbacks or []
    trainer.default_root_dir = default_root_dir
    trainer.callback_metrics = {}
    trainer.logged_metrics = {}
    trainer.logger = None
    trainer.datamodule = None
    trainer.sanity_checking = False
    return trainer


def _mock_batch(
    batch_size: int = 4,
    session_ids: list[str] | None = None,
) -> dict:
    batch = {
        "task_index": torch.ones(batch_size, 1, dtype=torch.long),
    }
    if session_ids is not None:
        batch["input_session_ids"] = session_ids
    return batch


def _write_fake_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"weight": torch.randn(4, 4)}}, path)


def _write_manifest_for_checkpoint(
    checkpoint_path: Path,
    manifest_dir: Path,
    kind: str = "best",
    **overrides: Any,
) -> tuple[Path, Path]:
    defaults = {
        "trained_on": {
            "source_selection_id": "test_sel",
            "source_manifest_path": "pool.json",
            "source_manifest_hash": "abc123",
            "excluded_target": {"species": "minipigs", "subject": "sub-06"},
            "subjects": ["minipigs:sub-01"],
            "recordings": ["minipigs:sub-01_ses-01"],
            "selected_train_examples": 100,
            "available_train_windows": 100,
            "realized_train_windows_per_epoch": 96,
            "processed_windows": 500,
            "completed_effective_epochs": 5.2,
            "optimizer_steps": 250,
            "class_union": ["low_bass", "midrange"],
            "class_intersection": ["low_bass"],
        },
        "selection": {
            "monitor": "val/source_session_mean_supported_f1",
            "monitor_value": 0.45,
            "source_session_scores": {
                "minipigs:sub-01_ses-01": 0.42,
                "minipigs:sub-02_ses-01": 0.48,
            },
        },
        "compute": {
            "cumulative_flops": 384049152000,
            "flop_method": "analytic-v1",
            "signal_seconds": 250.0,
            "wall_time_seconds": 120.5,
            "gpu": "NVIDIA A100",
            "precision": "bf16-mixed",
        },
        "recipe": {"model": {"adapter_dim": 64}},
        "normalization_artifact_hashes": {"sub-01_ses-01": "norm_hash_1"},
        "git_sha": "abc123def",
        "snapshot_bundle": "/snapshots/test",
        "slurm_job_id": "12345",
        "wandb_info": {
            "project": "test_project",
            "group": "test_group",
            "run_id": "test_run",
        },
    }
    defaults.update(overrides)
    return write_checkpoint_manifest(
        checkpoint_path, manifest_dir, kind=kind, **defaults
    )


# ── ComputeMilestoneCheckpointCallback tests ─────────────────────────────────


class TestMilestoneScheduleComputation:
    """Milestone rounding, deduplication, naming."""

    def test_default_fractions(self):
        cb = ComputeMilestoneCheckpointCallback()
        assert cb.milestone_fractions == list(DEFAULT_MILESTONE_FRACTIONS)

    def test_schedule_500_steps(self):
        cb = ComputeMilestoneCheckpointCallback()
        schedule = cb._build_milestone_schedule(500)
        expected_steps = {5, 15, 50, 150, 500}
        assert set(schedule.keys()) == expected_steps
        for step, info in schedule.items():
            assert info["saved"] is False
            assert info["not_reached"] is False
            assert info["path"] is None
            assert 0 < info["realized_pct"] <= 100.0

    def test_schedule_5000_steps(self):
        cb = ComputeMilestoneCheckpointCallback()
        schedule = cb._build_milestone_schedule(5000)
        expected_steps = {50, 150, 500, 1500, 5000}
        assert set(schedule.keys()) == expected_steps

    def test_deduplication_at_small_budget(self):
        cb = ComputeMilestoneCheckpointCallback()
        schedule = cb._build_milestone_schedule(10)
        steps = sorted(schedule.keys())
        assert len(steps) == len(set(steps)), "Duplicate milestone steps"
        assert all(s <= 10 for s in steps)
        assert all(s > 0 for s in steps)

    def test_deduplication_extreme_small_budget(self):
        cb = ComputeMilestoneCheckpointCallback()
        schedule = cb._build_milestone_schedule(3)
        steps = sorted(schedule.keys())
        assert len(steps) == len(set(steps))
        assert all(0 < s <= 3 for s in steps)

    def test_custom_fractions(self):
        cb = ComputeMilestoneCheckpointCallback(
            milestone_fractions=[0.5, 1.0]
        )
        schedule = cb._build_milestone_schedule(100)
        assert set(schedule.keys()) == {50, 100}

    def test_realized_pct_accuracy(self):
        cb = ComputeMilestoneCheckpointCallback()
        schedule = cb._build_milestone_schedule(500)
        for step, info in schedule.items():
            expected_pct = (step / 500) * 100.0
            assert abs(info["realized_pct"] - expected_pct) < 1e-10

    def test_empty_fractions_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            ComputeMilestoneCheckpointCallback(milestone_fractions=[])

    def test_invalid_fraction_rejected(self):
        with pytest.raises(ValueError, match="0 < fraction"):
            ComputeMilestoneCheckpointCallback(
                milestone_fractions=[0.0, 0.5]
            )
        with pytest.raises(ValueError, match="0 < fraction"):
            ComputeMilestoneCheckpointCallback(
                milestone_fractions=[1.5]
            )


class TestMilestoneExactTrigger:
    """Milestone checkpoints trigger at exact global_step values."""

    def test_saves_at_correct_step(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        cb.on_fit_start(trainer, pl_module)
        assert 5 in cb._milestone_steps

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        info = cb._milestone_steps[5]
        assert info["saved"] is True
        assert info["not_reached"] is False
        assert Path(info["path"]).exists()
        assert "milestone-1pct-step5" in info["path"]

    def test_does_not_save_at_wrong_step(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        cb.on_fit_start(trainer, pl_module)
        trainer.save_checkpoint = MagicMock()
        trainer.global_step = 6
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        trainer.save_checkpoint.assert_not_called()

    def test_does_not_save_twice(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        save_count = 0

        def mock_save(path):
            nonlocal save_count
            save_count += 1
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)
        cb.on_train_batch_end(trainer, pl_module, None, {}, 1)

        assert save_count == 1

    def test_non_global_zero_skips(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(
            max_steps=500, is_global_zero=False
        )
        pl_module = MagicMock()

        cb.on_fit_start(trainer, pl_module)
        trainer.save_checkpoint = MagicMock()
        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        trainer.save_checkpoint.assert_not_called()


class TestMilestoneNaming:
    """Milestone files follow the naming convention."""

    def test_filename_format(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        for step in [5, 15, 50, 150, 500]:
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        expected_files = {
            "milestone-1pct-step5.ckpt",
            "milestone-3pct-step15.ckpt",
            "milestone-10pct-step50.ckpt",
            "milestone-30pct-step150.ckpt",
            "milestone-100pct-step500.ckpt",
        }
        actual_files = {f.name for f in tmp_path.iterdir()}
        assert actual_files == expected_files


class TestMilestoneResumeRequeue:
    """Requeue/resume preserves counters and avoids duplicate milestone writes."""

    def test_state_dict_round_trip(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        state = cb.state_dict()
        assert state["milestone_steps"][5]["saved"] is True

        cb2 = ComputeMilestoneCheckpointCallback(
            checkpoint_dir=str(tmp_path)
        )
        cb2.load_state_dict(state)
        cb2.on_fit_start(trainer, pl_module)

        assert cb2._milestone_steps[5]["saved"] is True
        assert cb2._milestone_steps[5]["path"] is not None

    def test_resume_does_not_duplicate_saves(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        state = cb.state_dict()
        cb2 = ComputeMilestoneCheckpointCallback(
            checkpoint_dir=str(tmp_path)
        )
        cb2.load_state_dict(state)

        save_count = [0]
        original_save = mock_save

        def counting_save(path):
            save_count[0] += 1
            original_save(path)

        trainer.save_checkpoint = counting_save
        cb2.on_fit_start(trainer, pl_module)

        trainer.global_step = 5
        cb2.on_train_batch_end(trainer, pl_module, None, {}, 0)
        assert save_count[0] == 0, "Resumed callback re-saved an existing milestone"


class TestMilestoneUnreached:
    """Unreached milestones are marked explicitly."""

    def test_unreached_milestones_logged(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        cb.on_fit_end(trainer, pl_module)

        saved_count = sum(
            1 for info in cb._milestone_steps.values() if info["saved"]
        )
        not_reached_count = sum(
            1 for info in cb._milestone_steps.values() if info["not_reached"]
        )
        assert saved_count == 1
        assert not_reached_count == len(cb._milestone_steps) - 1

    def test_get_saved_milestones(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500)
        pl_module = MagicMock()

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        cb.on_fit_start(trainer, pl_module)

        for step in [5, 15]:
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        saved = cb.get_saved_milestones()
        assert set(saved.keys()) == {5, 15}
        assert all(info["saved"] for info in saved.values())


class TestMilestoneComputeSnapshot:
    """Milestones snapshot compute counters at save time."""

    def test_snapshot_captures_counters(self, tmp_path):
        compute_cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
        )
        cb = ComputeMilestoneCheckpointCallback(checkpoint_dir=str(tmp_path))
        trainer = _mock_trainer(max_steps=500, callbacks=[compute_cb])

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        pl_module = MagicMock()

        compute_cb.on_fit_start(trainer, pl_module)
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=16)
        for i in range(5):
            trainer.global_step = i + 1
            compute_cb.on_train_batch_end(
                trainer, pl_module, None, batch, i
            )
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        info = cb._milestone_steps[5]
        assert info["saved"] is True
        snap = info["compute_snapshot"]
        assert snap["optimizer_steps"] == 5
        assert snap["processed_windows"] == 80


# ── ComputeTrackingCallback tests ────────────────────────────────────────────


class TestComputeTrackingPerSessionFlops:
    """Per-session FLOP sums match a hand calculation and are monotonic."""

    def test_session_flops_accumulation(self):
        session_flops = {
            "minipigs:sub-01_ses-01": 1000,
            "minipigs:sub-02_ses-01": 2000,
        }
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops=session_flops,
            flop_method="analytic-v1",
        )

        trainer = _mock_trainer()
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch1 = _mock_batch(
            batch_size=3,
            session_ids=[
                "minipigs:sub-01_ses-01",
                "minipigs:sub-01_ses-01",
                "minipigs:sub-02_ses-01",
            ],
        )
        cb.on_train_batch_end(trainer, pl_module, None, batch1, 0)

        assert cb._per_session_windows["minipigs:sub-01_ses-01"] == 2
        assert cb._per_session_windows["minipigs:sub-02_ses-01"] == 1

        expected_flops = 2 * 1000 + 1 * 2000
        assert cb._cumulative_flops(trainer) == expected_flops

    def test_session_flops_monotonic(self):
        session_flops = {"sess_a": 500, "sess_b": 700}
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops=session_flops,
            flop_method="test",
        )

        trainer = _mock_trainer()
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        prev_flops = 0
        for i in range(10):
            batch = _mock_batch(
                batch_size=2,
                session_ids=["sess_a", "sess_b"],
            )
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)
            current_flops = cb._cumulative_flops(trainer)
            assert current_flops > prev_flops, "FLOPs must be monotonically increasing"
            prev_flops = current_flops

    def test_session_flops_and_flops_per_window_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ComputeTrackingCallback(
                monitor="val/f1",
                mode="max",
                sequence_length=0.5,
                flops_per_window=1000,
                session_flops={"sess_a": 500},
            )

    def test_negative_session_flops_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ComputeTrackingCallback(
                monitor="val/f1",
                mode="max",
                sequence_length=0.5,
                session_flops={"sess_a": -100},
            )

    def test_session_flops_state_dict_roundtrip(self):
        session_flops = {"sess_a": 500, "sess_b": 700}
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops=session_flops,
            flop_method="test",
        )

        trainer = _mock_trainer()
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(
            batch_size=2,
            session_ids=["sess_a", "sess_b"],
        )
        cb.on_train_batch_end(trainer, pl_module, None, batch, 0)

        state = cb.state_dict()
        assert state["per_session_windows"] == {"sess_a": 1, "sess_b": 1}

        cb2 = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops=session_flops,
            flop_method="test",
        )
        cb2.load_state_dict(state)
        assert cb2._per_session_windows == {"sess_a": 1, "sess_b": 1}

    def test_per_session_windows_snapshot_at_best(self):
        session_flops = {"sess_a": 500, "sess_b": 700}
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops=session_flops,
            flop_method="test",
        )

        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=2, session_ids=["sess_a", "sess_b"])
        for i in range(5):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.6)}
        cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._best_per_session_windows == {"sess_a": 5, "sess_b": 5}
        assert cb._best_flops == 5 * 500 + 5 * 700

        for i in range(3):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        assert cb._per_session_windows == {"sess_a": 8, "sess_b": 8}
        assert cb._best_per_session_windows == {"sess_a": 5, "sess_b": 5}


class TestComputeTrackingEffectiveEpochs:
    """Effective epochs = processed windows / realized train windows per epoch."""

    def test_effective_epochs_basic(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=100,
        )

        trainer = _mock_trainer()
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=10)
        for i in range(25):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        assert cb._effective_epochs(trainer) == 2.5

    def test_effective_epochs_none_without_config(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
        )
        trainer = _mock_trainer()
        assert cb._effective_epochs(trainer) is None

    def test_effective_epochs_at_best(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=100,
        )

        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=20)
        for i in range(5):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.5)}
        cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._best_effective_epochs == 1.0

    def test_effective_epochs_in_metrics(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=50,
        )
        trainer = _mock_trainer()
        pl_module = MagicMock()
        pl_module.model = MagicMock()
        pl_module.model.parameters.return_value = []
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=10)
        for i in range(10):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        metrics = cb._build_compute_metrics(trainer, pl_module)
        assert metrics["compute/effective_epochs"] == 2.0


class TestComputeTrackingBestCheckpoint:
    """Best checkpoint score/counters match the monitored aggregate."""

    def test_best_tracks_improvement(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
        )

        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=4)
        for i in range(10):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.3)}
        cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._best_monitor_value == pytest.approx(0.3)
        assert cb._best_step == 10
        assert cb._best_windows == 40

        for i in range(10):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.global_step = 20
        trainer.callback_metrics = {"val/f1": torch.tensor(0.5)}
        cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._best_monitor_value == pytest.approx(0.5)
        assert cb._best_step == 20
        assert cb._best_windows == 80

    def test_best_does_not_regress(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
        )

        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=4)
        for i in range(10):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.5)}
        cb.on_validation_epoch_end(trainer, pl_module)

        trainer.global_step = 20
        trainer.callback_metrics = {"val/f1": torch.tensor(0.3)}
        cb.on_validation_epoch_end(trainer, pl_module)

        assert cb._best_monitor_value == pytest.approx(0.5)
        assert cb._best_step == 10

    def test_compute_snapshot_methods(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=100,
        )

        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=10)
        for i in range(10):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.6)}
        cb.on_validation_epoch_end(trainer, pl_module)

        snap = cb.get_compute_snapshot(trainer)
        assert snap["processed_windows"] == 100
        assert snap["signal_seconds"] == 50.0
        assert snap["optimizer_steps"] == 10
        assert snap["effective_epochs"] == 1.0

        best_snap = cb.get_best_compute_snapshot()
        assert best_snap["processed_windows"] == 100
        assert best_snap["optimizer_steps"] == 10
        assert best_snap["effective_epochs"] == 1.0
        assert best_snap["monitor_value"] == pytest.approx(0.6)


class TestComputeTrackingStatePersistence:
    """State dict preserves all counters across requeue."""

    def test_full_state_roundtrip(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=100,
        )
        trainer = _mock_trainer(global_step=10)
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        batch = _mock_batch(batch_size=8)
        for i in range(5):
            cb.on_train_batch_end(trainer, pl_module, None, batch, i)

        trainer.callback_metrics = {"val/f1": torch.tensor(0.7)}
        cb.on_validation_epoch_end(trainer, pl_module)

        state = cb.state_dict()

        cb2 = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            realized_train_windows_per_epoch=100,
        )
        cb2.load_state_dict(state)

        assert cb2._processed_windows == 40
        assert cb2._best_monitor_value == pytest.approx(0.7)
        assert cb2._best_step == 10
        assert cb2._best_windows == 40
        assert cb2._best_effective_epochs == pytest.approx(0.4)


# ── Checkpoint manifest tests ────────────────────────────────────────────────


class TestCheckpointManifestWriteAndLoad:
    """Write, load, and verify round-trip for checkpoint manifests."""

    def test_write_and_load_roundtrip(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_manifest_for_checkpoint(
            ckpt, manifest_dir
        )

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

        manifest = load_checkpoint_manifest(json_path)
        assert manifest["schema"] == CHECKPOINT_MANIFEST_SCHEMA
        assert manifest["version"] == CHECKPOINT_MANIFEST_VERSION
        assert manifest["checkpoint"]["kind"] == "best"
        assert len(manifest["checkpoint"]["sha256"]) == 64
        assert manifest["trained_on"]["source_selection_id"] == "test_sel"

    def test_manifest_hash_detects_tampering(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        data = json.loads(json_path.read_text())
        data["trained_on"]["optimizer_steps"] = 999999
        json_path.write_text(json.dumps(data, indent=2))

        with pytest.raises(CheckpointManifestError, match="hash mismatch"):
            load_checkpoint_manifest(json_path)

    def test_checkpoint_sha256_detects_tampering(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)

        torch.save({"tampered": True}, ckpt)

        with pytest.raises(CheckpointManifestError, match="SHA-256 mismatch"):
            verify_checkpoint_integrity(manifest, str(tmp_path))

    def test_valid_checkpoint_passes_integrity(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)
        verify_checkpoint_integrity(manifest, str(tmp_path))

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(CheckpointManifestError, match="not found"):
            load_checkpoint_manifest(tmp_path / "nonexistent.json")

    def test_missing_checkpoint_raises(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)
        ckpt.unlink()

        with pytest.raises(CheckpointManifestError, match="not found"):
            verify_checkpoint_integrity(manifest, str(tmp_path))

    def test_valid_checkpoint_passes_integrity_with_correct_root(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)
        verify_checkpoint_integrity(manifest, str(tmp_path))

    def test_invalid_schema_raises(self, tmp_path):
        bad_manifest = {
            "schema": "wrong-schema",
            "version": 1,
            "manifest_hash": "abc",
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_manifest))

        with pytest.raises(CheckpointManifestError, match="Unsupported schema"):
            load_checkpoint_manifest(path)


class TestCheckpointManifestMarkdown:
    """Markdown is generated from JSON and shows source data plus epochs."""

    def test_markdown_contains_required_sections(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        md_text = md_path.read_text()

        assert "# NeuroSoft Pretraining Checkpoint Manifest" in md_text
        assert "**Kind:** best" in md_text
        assert "**Species:** minipigs" in md_text
        assert "**Subject:** sub-06" in md_text
        assert "**Selection ID:** test_sel" in md_text
        assert "minipigs:sub-01_ses-01" in md_text
        assert "0.45" in md_text
        assert "**Completed effective epochs:** 5.2" in md_text
        assert "**Optimizer steps:** 250" in md_text
        assert "**SHA-256:**" in md_text
        assert "## Source Data" in md_text
        assert "## Compute" in md_text
        assert "## Provenance" in md_text

    def test_markdown_generated_from_json(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_manifest_for_checkpoint(ckpt, manifest_dir)

        manifest = json.loads(json_path.read_text())
        regenerated_md = generate_checkpoint_markdown(manifest)
        assert regenerated_md == md_path.read_text()


class TestCheckpointManifestWriter:
    """The CheckpointManifestWriter facade delegates correctly."""

    def test_writer_facade(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        _write_fake_checkpoint(ckpt)

        json_path, md_path = CheckpointManifestWriter.write(
            ckpt,
            tmp_path / "manifests",
            kind="best",
            trained_on={"excluded_target": {"species": "test", "subject": "t"}},
            selection={"monitor": "val/f1", "monitor_value": 0.5},
            compute={"cumulative_flops": 0, "flop_method": "none"},
            recipe={},
            normalization_artifact_hashes={},
            git_sha="abc",
            snapshot_bundle="/test",
            slurm_job_id="123",
            wandb_info={"project": "t", "group": "g", "run_id": "r"},
        )

        manifest = CheckpointManifestWriter.load(json_path)
        assert manifest["checkpoint"]["kind"] == "best"
        CheckpointManifestWriter.verify_integrity(
            manifest, str(tmp_path)
        )


class TestCheckpointManifestAtomicWrite:
    """Manifest writes are atomic (no partial files on failure)."""

    def test_missing_checkpoint_raises_before_write(self, tmp_path):
        fake_path = tmp_path / "nonexistent.ckpt"
        manifest_dir = tmp_path / "manifests"

        with pytest.raises(CheckpointManifestError, match="not found"):
            write_checkpoint_manifest(
                fake_path,
                manifest_dir,
                kind="best",
                trained_on={"excluded_target": {"species": "t", "subject": "s"}},
                selection={},
                compute={},
                recipe={},
                normalization_artifact_hashes={},
                git_sha="x",
                snapshot_bundle="x",
                slurm_job_id="x",
                wandb_info={},
            )

        assert not manifest_dir.exists() or not list(manifest_dir.iterdir())


class TestMilestoneManifestEmission:
    """Best and milestone checkpoints both produce valid manifests."""

    def test_milestone_kind_in_manifest(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "milestone-10pct-step50.ckpt"
        _write_fake_checkpoint(ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_manifest_for_checkpoint(
            ckpt, manifest_dir, kind="milestone-10pct"
        )

        manifest = load_checkpoint_manifest(json_path)
        assert manifest["checkpoint"]["kind"] == "milestone-10pct"

        md_text = md_path.read_text()
        assert "milestone-10pct" in md_text

    def test_multiple_manifests_for_different_checkpoints(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        manifest_dir = tmp_path / "manifests"

        names = ["best", "milestone-1pct-step5", "milestone-100pct-step500"]
        for name in names:
            ckpt = checkpoint_dir / f"{name}.ckpt"
            _write_fake_checkpoint(ckpt)
            _write_manifest_for_checkpoint(ckpt, manifest_dir, kind=name)

        json_files = sorted(manifest_dir.glob("*.json"))
        md_files = sorted(manifest_dir.glob("*.md"))
        assert len(json_files) == 3
        assert len(md_files) == 3

        for jf in json_files:
            manifest = load_checkpoint_manifest(jf)
            verify_checkpoint_integrity(manifest, str(tmp_path))


class TestGradientAccumulationStepCounting:
    """Gradient accumulation correctly counts optimizer steps for milestones."""

    def test_accumulation_does_not_change_milestone_trigger(self, tmp_path):
        cb = ComputeMilestoneCheckpointCallback(
            checkpoint_dir=str(tmp_path),
            milestone_fractions=[0.5, 1.0],
        )
        trainer = _mock_trainer(max_steps=10, accumulate_grad_batches=4)

        def mock_save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"test": True}, path)

        trainer.save_checkpoint = mock_save
        pl_module = MagicMock()

        cb.on_fit_start(trainer, pl_module)
        schedule = cb._milestone_steps
        assert 5 in schedule
        assert 10 in schedule

        for step in range(1, 11):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, None, {}, 0)

        assert schedule[5]["saved"] is True
        assert schedule[10]["saved"] is True


class TestComputeTrackingRequireFlops:
    """require_flops validation with session_flops."""

    def test_require_flops_with_session_flops(self):
        cb = ComputeTrackingCallback(
            monitor="val/f1",
            mode="max",
            sequence_length=0.5,
            session_flops={"sess_a": 1000},
            flop_method="analytic-v1",
            require_flops=True,
        )
        assert cb.require_flops is True

    def test_require_flops_without_any_flops_fails(self):
        with pytest.raises(ValueError, match="require_flops"):
            ComputeTrackingCallback(
                monitor="val/f1",
                mode="max",
                sequence_length=0.5,
                require_flops=True,
                flop_method="test",
            )
