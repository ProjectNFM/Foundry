"""Tests for TaskConfig and configs/tasks/*.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from foundry.tasks.config import TaskConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_CONFIG_DIR = REPO_ROOT / "configs" / "tasks"


class TestTaskConfigProperties:
    def test_output_dim_reads_from_head(self):
        cfg = TaskConfig(
            name="example",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 5,
            },
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "trials.timestamps",
                "value_key": "trials.values",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )

        assert cfg.output_dim == 5

    def test_kind_infers_binary_multiclass_and_continuous(self):
        binary = TaskConfig(
            name="binary",
            head={"output_dim": 2},
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        multiclass = TaskConfig(
            name="multiclass",
            head={"output_dim": 5},
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        continuous = TaskConfig(
            name="continuous",
            head={"output_dim": 18},
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.MSETaskLoss"},
        )

        assert binary.kind == "binary"
        assert multiclass.kind == "multiclass"
        assert continuous.kind == "continuous"


def _task_yaml_paths() -> list[Path]:
    return sorted(TASKS_CONFIG_DIR.glob("*.yaml"))


class TestTaskYamlConfigs:
    def test_motor_imagery_5class_yaml_loads_and_matches_modality(self):
        cfg = TaskConfig.from_yaml(
            TASKS_CONFIG_DIR / "motor_imagery_5class.yaml"
        )

        assert cfg.name == "motor_imagery_5class"
        assert cfg.output_dim == 5
        assert cfg.kind == "multiclass"
        assert (
            cfg.target_extractor["timestamp_key"]
            == "motor_imagery_trials.timestamps"
        )
        assert (
            cfg.target_extractor["value_key"]
            == "motor_imagery_trials.movement_ids"
        )
        assert cfg.class_names == [
            "Rest",
            "Left hand",
            "Right hand",
            "Feet",
            "Tongue",
        ]

    def test_yaml_loss_params_can_be_overridden(self):
        base = OmegaConf.load(TASKS_CONFIG_DIR / "motor_imagery_5class.yaml")
        merged = OmegaConf.merge(
            base,
            OmegaConf.create({"loss": {"label_smoothing": 0.2}}),
        )
        cfg = TaskConfig.from_dict(OmegaConf.to_container(merged, resolve=True))

        assert cfg.loss["label_smoothing"] == 0.2

    @pytest.mark.parametrize(
        "yaml_path", _task_yaml_paths(), ids=lambda p: p.stem
    )
    def test_each_task_yaml_parses_with_expected_kind_and_output_dim(
        self, yaml_path: Path
    ):
        expected = EXPECTED_TASK_SPECS[yaml_path.stem]
        cfg = TaskConfig.from_yaml(yaml_path)

        assert cfg.name == yaml_path.stem
        assert cfg.output_dim == expected["output_dim"]
        assert cfg.kind == expected["kind"]
        assert (
            cfg.target_extractor["timestamp_key"] == expected["timestamp_key"]
        )
        assert cfg.target_extractor["value_key"] == expected["value_key"]

    @pytest.mark.parametrize(
        "yaml_path", _task_yaml_paths(), ids=lambda p: p.stem
    )
    def test_each_task_yaml_components_instantiate(self, yaml_path: Path):
        cfg = TaskConfig.from_yaml(yaml_path)
        embed_dim = 64

        head = instantiate({**cfg.head, "embed_dim": embed_dim})
        assert head.output_dim == cfg.output_dim

        extractor = instantiate(cfg.target_extractor)
        assert callable(extractor)

        loss = instantiate(cfg.loss)
        assert loss is not None

        if cfg.metrics is not None:
            metrics = instantiate(cfg.metrics)
            assert metrics is not None


EXPECTED_TASK_SPECS = {
    "motor_imagery_5class": {
        "output_dim": 5,
        "kind": "multiclass",
        "timestamp_key": "motor_imagery_trials.timestamps",
        "value_key": "motor_imagery_trials.movement_ids",
    },
    "motor_imagery_left_right": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "motor_imagery_trials.timestamps",
        "value_key": "motor_imagery_trials.movement_ids",
    },
    "motor_imagery_right_feet": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "motor_imagery_trials.timestamps",
        "value_key": "motor_imagery_trials.movement_ids",
    },
    "ajile_pose_estimation": {
        "output_dim": 18,
        "kind": "continuous",
        "timestamp_key": "pose_trajectories.timestamps",
        "value_key": "pose_trajectories.values",
    },
    "ajile_active_behavior": {
        "output_dim": 5,
        "kind": "multiclass",
        "timestamp_key": "active_behavior_trials.timestamps",
        "value_key": "active_behavior_trials.behavior_id",
    },
    "ajile_inactive_active": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "active_vs_inactive_trials.timestamps",
        "value_key": "active_vs_inactive_trials.behavior_id",
    },
    "sleep_stage_5class": {
        "output_dim": 5,
        "kind": "multiclass",
        "timestamp_key": "sleep_stages.timestamps",
        "value_key": "sleep_stages.values",
    },
    "neurosoft_on_vs_off": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "on_vs_off_trials.timestamps",
        "value_key": "on_vs_off_trials.behavior_ids",
    },
    "neurosoft_acoustic_stim": {
        "output_dim": 26,
        "kind": "multiclass",
        "timestamp_key": "acoustic_stim_trials.timestamps",
        "value_key": "acoustic_stim_trials.behavior_ids",
    },
}
