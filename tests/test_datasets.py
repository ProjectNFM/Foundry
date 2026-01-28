import pytest

from foundry.data.datasets import (
    KempSleepEDF2013,
    KlinzingSleepDS0055552024,
    KorczowskiBrainInvaders2014a,
    SchalkWolpawPhysionet2009,
    ShiraziHbnr1DS0055052024,
)

from .conftest import skip_if_missing_dataset


class TestKempSleepEDF2013:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset("kemp_sleep_edf_2013", data_root)
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_dataset_initialization(self, data_root):
        dataset = KempSleepEDF2013(root=str(data_root))
        assert dataset is not None
        assert hasattr(dataset, "recording_ids")

    def test_get_channel_ids(self, data_root):
        dataset = KempSleepEDF2013(root=str(data_root))
        channel_ids = dataset.get_channel_ids()
        assert isinstance(channel_ids, list)

    def test_get_sampling_intervals_train(self, data_root):
        dataset = KempSleepEDF2013(root=str(data_root), fold_number=0)
        intervals = dataset.get_sampling_intervals(split="train")
        assert isinstance(intervals, dict)

    def test_get_sampling_intervals_valid(self, data_root):
        dataset = KempSleepEDF2013(root=str(data_root), fold_number=0)
        intervals = dataset.get_sampling_intervals(split="valid")
        assert isinstance(intervals, dict)

    def test_get_sampling_intervals_test(self, data_root):
        dataset = KempSleepEDF2013(root=str(data_root), fold_number=0)
        intervals = dataset.get_sampling_intervals(split="test")
        assert isinstance(intervals, dict)


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


class TestKlinzingSleepDS0055552024:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset(
            "klinzing_sleep_ds005555_2024", data_root
        )
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_dataset_initialization(self, data_root):
        dataset = KlinzingSleepDS0055552024(root=str(data_root))
        assert dataset is not None
        assert hasattr(dataset, "recording_ids")

    def test_get_channel_ids(self, data_root):
        dataset = KlinzingSleepDS0055552024(root=str(data_root))
        channel_ids = dataset.get_channel_ids()
        assert isinstance(channel_ids, list)

    def test_uniquify_channel_ids_enabled(self, data_root):
        dataset = KlinzingSleepDS0055552024(
            root=str(data_root), uniquify_channel_ids=True
        )
        assert dataset.eeg_dataset_mixin_uniquify_channel_ids is True

    def test_uniquify_channel_ids_disabled(self, data_root):
        dataset = KlinzingSleepDS0055552024(
            root=str(data_root), uniquify_channel_ids=False
        )
        assert dataset.eeg_dataset_mixin_uniquify_channel_ids is False


class TestShiraziHbnr1DS0055052024:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset(
            "shirazi_hbnr1_ds005505_2024", data_root
        )
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_dataset_initialization(self, data_root):
        dataset = ShiraziHbnr1DS0055052024(root=str(data_root))
        assert dataset is not None
        assert hasattr(dataset, "recording_ids")

    def test_get_channel_ids(self, data_root):
        dataset = ShiraziHbnr1DS0055052024(root=str(data_root))
        channel_ids = dataset.get_channel_ids()
        assert isinstance(channel_ids, list)

    def test_uniquify_channel_ids_enabled(self, data_root):
        dataset = ShiraziHbnr1DS0055052024(
            root=str(data_root), uniquify_channel_ids=True
        )
        assert dataset.eeg_dataset_mixin_uniquify_channel_ids is True

    def test_uniquify_channel_ids_disabled(self, data_root):
        dataset = ShiraziHbnr1DS0055052024(
            root=str(data_root), uniquify_channel_ids=False
        )
        assert dataset.eeg_dataset_mixin_uniquify_channel_ids is False
