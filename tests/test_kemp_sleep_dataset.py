"""Tests for the Foundry KempSleepEDF2013 dataset wrapper."""

from __future__ import annotations


from foundry.data.datasets import KempSleepEDF2013
from foundry.data.datasets.mixins import TaskMixin
from foundry.data.transforms import SelectEEGChannels, PrepareSleepStages
from foundry.tasks.config import TaskConfig


class TestKempDatasetWrapper:
    def test_uses_task_mixin(self):
        assert issubclass(KempSleepEDF2013, TaskMixin)

    def test_sleep_stage_5class_in_available_tasks(self):
        assert "sleep_stage_5class" in KempSleepEDF2013.AVAILABLE_TASKS

    def test_sleep_stage_maps_to_sleep_stage_5class(self):
        task_configs = KempSleepEDF2013.get_tasks_for_experiment("sleep_stage")

        assert set(task_configs.keys()) == {"sleep_stage_5class"}

    def test_sleep_stage_task_config_is_multiclass_five(self):
        task_configs = KempSleepEDF2013.get_tasks_for_experiment("sleep_stage")
        cfg = task_configs["sleep_stage_5class"]

        assert isinstance(cfg, TaskConfig)
        assert cfg.kind == "multiclass"
        assert cfg.output_dim == 5

    def test_get_required_transforms_returns_select_then_prepare(self):
        transforms = KempSleepEDF2013.get_required_transforms("sleep_stage")

        assert len(transforms) == 2
        assert isinstance(transforms[0], SelectEEGChannels)
        assert isinstance(transforms[1], PrepareSleepStages)

    def test_get_required_transforms_empty_for_unknown_task(self):
        transforms = KempSleepEDF2013.get_required_transforms("other_task")

        assert transforms == []

    def test_fold_and_split_type_forwarded_to_torch_brain(self, tmp_path):
        """fold/split_type are translated to fold_number/fold_type."""
        ds = KempSleepEDF2013(
            root=str(tmp_path),
            fold=1,
            split_type="intrasession",
            recording_ids=[],
        )

        assert ds.fold_number == 1
        assert ds.fold_type == "intrasession"

    def test_task_type_kwarg_does_not_raise(self, tmp_path):
        """task_type is consumed by the wrapper and does not propagate to torch_brain."""
        ds = KempSleepEDF2013(
            root=str(tmp_path),
            task_type="sleep_stage",
            recording_ids=[],
        )

        assert ds.fold_number == 0

    def test_get_channel_ids_empty_when_no_recordings(self, tmp_path):
        ds = KempSleepEDF2013(root=str(tmp_path), recording_ids=[])

        assert ds.get_channel_ids() == []
