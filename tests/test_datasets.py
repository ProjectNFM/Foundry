import pytest
import numpy as np

from foundry.data.datasets import (
    KorczowskiBrainInvaders2014a,
    SchalkWolpawPhysionet2009,
)
from foundry.data.datasets.mixins import EEGDatasetMixin
from foundry.data.datamodules.neurosoft import NeurosoftMinipigsMonkeys2026
from torch_brain.dataset.dataset import DatasetIndex

from .conftest import skip_if_missing_dataset


class TestEEGDatasetMixin:
    def test_get_channel_ids_returns_unique_sorted_vocab(self):
        class Channels:
            def __init__(self, ids):
                self.id = np.array(ids)

        class Recording:
            def __init__(self, ids):
                self.channels = Channels(ids)

        class Dataset(EEGDatasetMixin):
            recording_ids = ["r1", "r2"]

            def __init__(self):
                self.recordings = {
                    "r1": Recording(["z", "a", "b"]),
                    "r2": Recording(["b", "a", "c"]),
                }

            def get_recording(self, recording_id):
                return self.recordings[recording_id]

        assert Dataset().get_channel_ids() == ["a", "b", "c", "z"]


class TestNeurosoftMinipigsMonkeys2026:
    def test_namespaces_recordings_channels_and_samples(self):
        class FakeDataset:
            def __init__(
                self,
                root,
                transform=None,
                fold_num=None,
                split_type=None,
                task_type=None,
                recording_ids=None,
            ):
                self.root = root
                self.transform = transform
                self.fold_num = fold_num
                self.split_type = split_type
                self.task_type = task_type
                self.recording_ids = recording_ids or ["default"]

            def get_sampling_intervals(self, split=None):
                return {
                    recording_id: f"{split}:{recording_id}"
                    for recording_id in self.recording_ids
                }

            def get_channel_ids(self):
                return ["sub/ch2", "sub/ch1", "sub/ch1"]

            def __getitem__(self, index):
                return (
                    index.recording_id,
                    index.start,
                    index.end,
                    index._namespace,
                )

        old_sources = NeurosoftMinipigsMonkeys2026.SOURCES
        NeurosoftMinipigsMonkeys2026.SOURCES = {
            "minipigs": FakeDataset,
            "monkeys": FakeDataset,
        }
        try:
            dataset = NeurosoftMinipigsMonkeys2026(
                root="/tmp/data",
                fold_num=2,
                split_type="intrasession-causal",
                task_type="acoustic_stim",
                minipigs_recording_ids=["pig_a"],
                monkeys_recording_ids=["monkey_b"],
            )
        finally:
            NeurosoftMinipigsMonkeys2026.SOURCES = old_sources

        assert dataset.recording_ids == ["minipigs/pig_a", "monkeys/monkey_b"]
        assert dataset.get_sampling_intervals("train") == {
            "minipigs/pig_a": "train:pig_a",
            "monkeys/monkey_b": "train:monkey_b",
        }
        assert dataset.get_channel_ids() == [
            "minipigs/sub/ch1",
            "minipigs/sub/ch2",
            "monkeys/sub/ch1",
            "monkeys/sub/ch2",
        ]
        assert dataset[DatasetIndex("monkeys/monkey_b", 0.0, 0.5)] == (
            "monkey_b",
            0.0,
            0.5,
            "monkeys",
        )


class TestSchalkWolpawPhysionet2009:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset(
            "schalk_wolpaw_physionet_2009", data_root
        )
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_dataset_initialization(self, data_root):
        dataset = SchalkWolpawPhysionet2009(root=str(data_root))
        assert dataset is not None
        assert hasattr(dataset, "recording_ids")

    def test_get_channel_ids(self, data_root):
        dataset = SchalkWolpawPhysionet2009(root=str(data_root))
        channel_ids = dataset.get_channel_ids()
        assert isinstance(channel_ids, list)

    def test_get_sampling_intervals_motor_imagery(self, data_root):
        dataset = SchalkWolpawPhysionet2009(
            root=str(data_root), task_type="MotorImagery", fold_number=0
        )
        intervals = dataset.get_sampling_intervals(split="train")
        assert isinstance(intervals, dict)

    def test_get_sampling_intervals_left_right(self, data_root):
        dataset = SchalkWolpawPhysionet2009(
            root=str(data_root), task_type="LeftRightImagery", fold_number=0
        )
        intervals = dataset.get_sampling_intervals(split="train")
        assert isinstance(intervals, dict)

    def test_task_config_validation(self, data_root):
        with pytest.raises(ValueError, match="Invalid task_type"):
            SchalkWolpawPhysionet2009(
                root=str(data_root), task_type="InvalidTask"
            )


class TestKorczowskiBrainInvaders2014a:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset(
            "korczowski_brain_invaders_2014a", data_root
        )
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_dataset_initialization(self, data_root):
        dataset = KorczowskiBrainInvaders2014a(root=str(data_root))
        assert dataset is not None
        assert hasattr(dataset, "recording_ids")

    def test_get_channel_ids(self, data_root):
        dataset = KorczowskiBrainInvaders2014a(root=str(data_root))
        channel_ids = dataset.get_channel_ids()
        assert isinstance(channel_ids, list)

    def test_get_sampling_intervals_inter_subject(self, data_root):
        dataset = KorczowskiBrainInvaders2014a(
            root=str(data_root), fold_number=0, fold_type="inter-subject"
        )
        intervals = dataset.get_sampling_intervals(split="train")
        assert isinstance(intervals, dict)

    def test_get_sampling_intervals_intra_subject(self, data_root):
        dataset = KorczowskiBrainInvaders2014a(
            root=str(data_root), fold_number=0, fold_type="intra-subject"
        )
        intervals = dataset.get_sampling_intervals(split="train")
        assert isinstance(intervals, dict)

    def test_fold_type_validation(self, data_root):
        with pytest.raises(ValueError, match="Invalid fold_type"):
            KorczowskiBrainInvaders2014a(
                root=str(data_root), fold_type="invalid"
            )
