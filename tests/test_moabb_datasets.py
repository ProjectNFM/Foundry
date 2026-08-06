"""Tests for the Foundry MOABB dataset wrappers (PhysionetMI, BrainInvadersP300)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry.data.datasets import BrainInvadersP300, PhysionetMI
from foundry.data.datasets.brain_invaders_p300 import _keep_anchor_trial
from foundry.tasks.config import TaskConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "configs" / "tasks"


class TestPhysionetMIDatasetWrapper:
    def test_motor_imagery_binary_task_config_loads(self):
        cfg = TaskConfig.from_yaml(TASKS_DIR / "motor_imagery_binary.yaml")

        assert isinstance(cfg, TaskConfig)
        assert cfg.name == "motor_imagery_binary"
        assert cfg.kind == "binary"
        assert cfg.output_dim == 2

    def test_fold_and_split_type_forwarded(self, tmp_path):
        ds = PhysionetMI(
            root=str(tmp_path),
            fold=1,
            split_type="intrasession",
            recording_ids=[],
        )
        assert ds.fold_number == 1
        assert ds.fold_type == "intrasession"

    def test_task_type_kwarg_does_not_raise(self, tmp_path):
        ds = PhysionetMI(
            root=str(tmp_path),
            task_type="motor_imagery",
            recording_ids=[],
        )
        assert ds.fold_number == 0

    def test_invalid_fold_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Fold number must be"):
            PhysionetMI(root=str(tmp_path), fold=5, recording_ids=[])

    def test_invalid_split_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid split_type"):
            PhysionetMI(
                root=str(tmp_path), split_type="bogus", recording_ids=[]
            )

    def test_get_channel_ids_empty_when_no_recordings(self, tmp_path):
        ds = PhysionetMI(root=str(tmp_path), recording_ids=[])
        assert ds.get_channel_ids() == []

    def test_get_required_transforms_empty(self):
        assert PhysionetMI.get_required_transforms("motor_imagery") == []


class TestBrainInvadersP300DatasetWrapper:
    def test_p300_binary_task_config_loads(self):
        cfg = TaskConfig.from_yaml(TASKS_DIR / "p300_binary.yaml")

        assert isinstance(cfg, TaskConfig)
        assert cfg.name == "p300_binary"
        assert cfg.kind == "binary"
        assert cfg.output_dim == 2

    def test_fold_and_split_type_forwarded(self, tmp_path):
        ds = BrainInvadersP300(
            root=str(tmp_path),
            fold=2,
            split_type="intersubject",
            recording_ids=[],
        )
        assert ds.fold_number == 2
        assert ds.fold_type == "intersubject"

    def test_task_type_kwarg_does_not_raise(self, tmp_path):
        ds = BrainInvadersP300(
            root=str(tmp_path),
            task_type="p300",
            recording_ids=[],
        )
        assert ds.fold_number == 0

    def test_invalid_fold_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Fold number must be"):
            BrainInvadersP300(root=str(tmp_path), fold=-1, recording_ids=[])

    def test_invalid_split_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid split_type"):
            BrainInvadersP300(
                root=str(tmp_path), split_type="bogus", recording_ids=[]
            )

    def test_get_channel_ids_empty_when_no_recordings(self, tmp_path):
        ds = BrainInvadersP300(root=str(tmp_path), recording_ids=[])
        assert ds.get_channel_ids() == []

    def test_get_required_transforms_returns_keep_anchor_trial(self):
        transforms = BrainInvadersP300.get_required_transforms("p300")

        assert len(transforms) == 1
        assert transforms[0] is _keep_anchor_trial

    def test_get_required_transforms_empty_for_unknown_task(self):
        assert BrainInvadersP300.get_required_transforms("other_task") == []
