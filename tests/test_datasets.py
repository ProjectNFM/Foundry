import pytest

from foundry.data.datasets import (
    KorczowskiBrainInvaders2014a,
    SchalkWolpawPhysionet2009,
)

from .conftest import skip_if_missing_dataset


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
