"""Tests for TaskConfig and configs/tasks/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    def test_ajile_active_behavior_yaml_loads_and_matches_modality(self):
        cfg = TaskConfig.from_yaml(
            TASKS_CONFIG_DIR / "ajile_active_behavior.yaml"
        )

        assert cfg.name == "ajile_active_behavior"
        assert cfg.output_dim == 5
        assert cfg.kind == "multiclass"
        assert (
            cfg.target_extractor["timestamp_key"]
            == "active_behavior_trials.timestamps"
        )
        assert (
            cfg.target_extractor["value_key"]
            == "active_behavior_trials.behavior_id"
        )
        assert cfg.class_names == [
            "Eat",
            "Talk",
            "TV",
            "Computer/Phone",
            "Other Activity",
        ]

    def test_yaml_loss_params_can_be_overridden(self):
        base = OmegaConf.load(TASKS_CONFIG_DIR / "ajile_active_behavior.yaml")
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

        head_kwargs = {**cfg.head, "embed_dim": embed_dim}
        if "output_dim" not in head_kwargs:
            head_kwargs["output_dim"] = cfg.output_dim
        head = instantiate(head_kwargs)
        assert head.output_dim == cfg.output_dim

        extractor = instantiate(cfg.target_extractor)
        assert callable(extractor)

        loss = instantiate(cfg.loss)
        assert loss is not None

        if cfg.metrics is not None:
            metrics = instantiate(cfg.metrics)
            assert metrics is not None


def _make_task_data(**overrides: Any) -> dict[str, Any]:
    """Minimal valid task data dict, with optional overrides."""
    base: dict[str, Any] = {
        "name": "test_task",
        "head": {"_target_": "foundry.tasks.heads.ReadoutHead"},
        "target_extractor": {
            "_target_": "foundry.tasks.targets.TargetExtractor",
            "timestamp_key": "t",
            "value_key": "v",
        },
        "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
    }
    base.update(overrides)
    return base


class TestFromDictMetricsNumClasses:
    """Auto-injection and validation of metrics.num_classes from mapping."""

    def test_num_classes_injected_when_mapping_present(self):
        data = _make_task_data(
            classification_mapping={
                "mapping": {0: "A", 1: "B", 2: "C"},
            },
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
            },
        )
        cfg = TaskConfig.from_dict(data)

        assert cfg.metrics["num_classes"] == 3

    def test_num_classes_injected_overwrites_matching_value(self):
        data = _make_task_data(
            classification_mapping={
                "mapping": {0: "A", 1: "B", 2: "C"},
            },
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 3,
            },
        )
        cfg = TaskConfig.from_dict(data)

        assert cfg.metrics["num_classes"] == 3

    def test_raises_on_conflicting_num_classes(self):
        data = _make_task_data(
            classification_mapping={
                "mapping": {0: "A", 1: "B", 2: "C"},
            },
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 7,
            },
        )
        with pytest.raises(
            ValueError, match=r"metrics\.num_classes \(7\).*conflicts"
        ):
            TaskConfig.from_dict(data)

    def test_no_mapping_leaves_metrics_unchanged(self):
        data = _make_task_data(
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 5,
            },
        )
        cfg = TaskConfig.from_dict(data)

        assert cfg.metrics["num_classes"] == 5

    def test_no_metrics_with_mapping_leaves_metrics_none(self):
        data = _make_task_data(
            classification_mapping={
                "mapping": {0: "A", 1: "B"},
            },
        )
        cfg = TaskConfig.from_dict(data)

        assert cfg.metrics is None


EXPECTED_TASK_SPECS = {
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
        "timestamp_key": "stages.start",
        "value_key": "stages.names",
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
    "neurosoft_acoustic_stim_2band": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "acoustic_stim_trials.timestamps",
        "value_key": "acoustic_stim_trials.behavior_labels",
    },
    "neurosoft_acoustic_stim_2band_rnd_a": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "acoustic_stim_trials.timestamps",
        "value_key": "acoustic_stim_trials.behavior_labels",
    },
    "neurosoft_acoustic_stim_2band_rnd_b": {
        "output_dim": 2,
        "kind": "binary",
        "timestamp_key": "acoustic_stim_trials.timestamps",
        "value_key": "acoustic_stim_trials.behavior_labels",
    },
    "neurosoft_acoustic_stim_3band": {
        "output_dim": 3,
        "kind": "multiclass",
        "timestamp_key": "acoustic_stim_trials.timestamps",
        "value_key": "acoustic_stim_trials.behavior_labels",
    },
}
