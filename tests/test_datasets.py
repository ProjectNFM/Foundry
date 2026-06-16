import pytest
import numpy as np
from temporaldata import Data, Interval
from torch_brain.nn.loss import MSELoss
from torch_brain.nn.multitask_readout import prepare_for_multitask_readout
from torch_brain.dataset.dataset import DatasetIndex
from torch_brain.registry import DataType, MODALITY_REGISTRY

from foundry.data.datasets import (
    KorczowskiBrainInvaders2014a,
    SchalkWolpawPhysionet2009,
)
from foundry.data.datasets.mixins import EEGDatasetMixin
from foundry.data.datamodules.neurosoft import (
    AddNeurosoftLogFrequencyTargets,
    AddNeurosoftSourceId,
    LOGFREQ_NORMALIZE_MEAN,
    LOGFREQ_NORMALIZE_STD,
    NeurosoftDataModule,
    NeurosoftMinipigs2026DataModule,
    NeurosoftMonkeys2026DataModule,
    NeurosoftMinipigsMonkeys2026,
    filter_acoustic_stim_tone_intervals,
)

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


class TestNeurosoftLogFrequencyRegression:
    def test_single_species_datamodules_add_source_id_transform(self):
        minipigs = NeurosoftMinipigs2026DataModule(
            root="/tmp/data",
            task_type="acoustic_stim",
            split_type="intrasession-causal",
        )
        monkeys = NeurosoftMonkeys2026DataModule(
            root="/tmp/data",
            task_type="acoustic_stim",
            split_type="intrasession-causal",
        )

        assert isinstance(minipigs.transform[0], AddNeurosoftSourceId)
        assert minipigs.transform[0].source_id == "minipigs"
        assert isinstance(monkeys.transform[0], AddNeurosoftSourceId)
        assert monkeys.transform[0].source_id == "monkeys"

    def test_source_id_transform_attaches_source_id_to_data(self):
        data = Data(domain=Interval(0.0, 1.0))

        transformed = AddNeurosoftSourceId("minipigs")(data)

        assert transformed.source_id == "minipigs"

    def test_logfreq_modality_is_registered(self):
        spec = MODALITY_REGISTRY["neurosoft_acoustic_stim_logfreq"]

        assert spec.dim == 1
        assert spec.type == DataType.CONTINUOUS
        assert spec.timestamp_key == "acoustic_stim_trials.timestamps"
        assert spec.value_key == "acoustic_stim_trials.log_frequency_values"
        assert isinstance(spec.loss_fn, MSELoss)

    def test_logfreq_task_maps_to_regression_readout(self):
        assert NeurosoftDataModule.get_readout_specs_for_task(
            "acoustic_stim_logfreq"
        ) == ["neurosoft_acoustic_stim_logfreq"]
        assert NeurosoftDataModule.get_class_names_for_task(
            "acoustic_stim_logfreq"
        ) == {}

    def test_logfreq_datamodule_uses_acoustic_stim_dataset_task(self):
        datamodule = NeurosoftMinipigs2026DataModule(
            root="/tmp/data",
            task_type="acoustic_stim_logfreq",
            split_type="intrasession-causal",
        )

        assert datamodule.task_type == "acoustic_stim_logfreq"
        assert datamodule.dataset_kwargs["task_type"] == "acoustic_stim"
        assert isinstance(
            datamodule.transform[0], AddNeurosoftSourceId
        )
        assert isinstance(
            datamodule.transform[1], AddNeurosoftLogFrequencyTargets
        )

    def test_logfreq_transform_filters_white_noise_and_adds_targets(self):
        data = Data(domain=Interval(0.0, 3.0))
        data.acoustic_stim_trials = Interval(
            start=np.array([0.0, 1.0, 2.0]),
            end=np.array([0.5, 1.5, 2.5]),
            timestamps=np.array([0.25, 1.25, 2.25]),
            behavior_labels=np.array(
                ["stim_100Hz", "stim_wn", "stim_12000Hz"], dtype=object
            ),
            behavior_ids=np.array([0, 25, 8]),
            timekeys=["start", "end", "timestamps"],
        )

        transformed = AddNeurosoftLogFrequencyTargets()(data)

        np.testing.assert_array_equal(
            transformed.acoustic_stim_trials.behavior_labels,
            np.array(["stim_100Hz", "stim_12000Hz"], dtype=object),
        )
        np.testing.assert_allclose(
            transformed.acoustic_stim_trials.log_frequency_values,
            np.log(np.array([[100.0], [12000.0]], dtype=np.float32)),
        )
        assert (
            transformed.acoustic_stim_trials.log_frequency_values.dtype
            == np.float32
        )
        readout_config = transformed.config["multitask_readout"][0]
        assert readout_config == {
            "readout_id": "neurosoft_acoustic_stim_logfreq",
            "normalize_mean": LOGFREQ_NORMALIZE_MEAN,
            "normalize_std": LOGFREQ_NORMALIZE_STD,
        }
        assert readout_config["normalize_std"] > 0

    def test_logfreq_readout_uses_z_scored_targets(self):
        data = Data(domain=Interval(0.0, 3.0))
        data.acoustic_stim_trials = Interval(
            start=np.array([0.0, 1.0]),
            end=np.array([0.5, 1.5]),
            timestamps=np.array([0.25, 1.25]),
            behavior_labels=np.array(
                ["stim_100Hz", "stim_12000Hz"], dtype=object
            ),
            behavior_ids=np.array([0, 8]),
            timekeys=["start", "end", "timestamps"],
        )

        transformed = AddNeurosoftLogFrequencyTargets()(data)
        _, values, _, _, _ = prepare_for_multitask_readout(
            transformed,
            {"neurosoft_acoustic_stim_logfreq": MODALITY_REGISTRY[
                "neurosoft_acoustic_stim_logfreq"
            ]},
        )

        raw_log_frequencies = np.log(
            np.array([[100.0], [12000.0]], dtype=np.float32)
        )
        expected = (
            raw_log_frequencies - LOGFREQ_NORMALIZE_MEAN
        ) / LOGFREQ_NORMALIZE_STD
        np.testing.assert_allclose(
            values["neurosoft_acoustic_stim_logfreq"], expected
        )

    def test_filter_acoustic_stim_tone_intervals_excludes_white_noise(self):
        intervals = {
            "r1": Interval(
                start=np.array([0.0, 1.0, 2.0]),
                end=np.array([0.5, 1.5, 2.5]),
                timestamps=np.array([0.25, 1.25, 2.25]),
                behavior_labels=np.array(
                    ["stim_100Hz", "stim_wn", "stim_400Hz"],
                    dtype=object,
                ),
                behavior_ids=np.array([0, 25, 3]),
                timekeys=["start", "end", "timestamps"],
            )
        }

        filtered = filter_acoustic_stim_tone_intervals(intervals)

        np.testing.assert_array_equal(
            filtered["r1"].behavior_labels,
            np.array(["stim_100Hz", "stim_400Hz"], dtype=object),
        )

    def test_filter_uses_recording_trials_for_unlabeled_split_intervals(self):
        class Dataset:
            def get_recording(self, recording_id):
                data = Data(domain=Interval(0.0, 5.0))
                data.acoustic_stim_trials = Interval(
                    start=np.array([0.25, 1.0, 4.0]),
                    end=np.array([0.75, 1.5, 4.5]),
                    timestamps=np.array([0.5, 1.25, 4.25]),
                    behavior_labels=np.array(
                        ["stim_100Hz", "stim_wn", "stim_400Hz"],
                        dtype=object,
                    ),
                    behavior_ids=np.array([0, 25, 3]),
                    timekeys=["start", "end", "timestamps"],
                )
                return data

        intervals = {
            "r1": Interval(
                start=np.array([0.0]),
                end=np.array([3.0]),
                timekeys=["start", "end"],
            )
        }

        filtered = filter_acoustic_stim_tone_intervals(
            intervals, dataset=Dataset()
        )

        np.testing.assert_array_equal(
            filtered["r1"].behavior_labels,
            np.array(["stim_100Hz"], dtype=object),
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
