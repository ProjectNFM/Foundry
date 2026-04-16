import pytest

from foundry.data.datamodules import NeuroprobeDataModule

from .conftest import skip_if_missing_dataset


def test_neuroprobe_task_readout_mapping():
    readouts = NeuroprobeDataModule.get_readout_specs_for_task("speech_binary")
    assert readouts == ["neuroprobe_speech_binary"]


def test_neuroprobe_invalid_task_readout_mapping():
    with pytest.raises(ValueError, match="Unknown task_type"):
        NeuroprobeDataModule.get_readout_specs_for_task("invalid_task")


class TestNeuroprobeDataModule:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root):
        skip_marker = skip_if_missing_dataset("neuroprobe_2025", data_root)
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    @staticmethod
    def _build_datamodule(data_root):
        return NeuroprobeDataModule(
            root=str(data_root),
            batch_size=2,
            num_workers=0,
            sequence_length=0.5,
            task_type="speech_binary",
            subset_tier="full",
            test_subject=1,
            test_session=0,
            label_mode="binary",
            task="speech",
            regime="SS-SM",
            fold=0,
        )

    def test_setup_creates_train_and_valid_datasets(self, data_root):
        datamodule = self._build_datamodule(data_root)
        datamodule.setup("fit")

        assert "train" in datamodule.datasets
        assert "valid" in datamodule.datasets
        assert datamodule.dataset is datamodule.datasets["train"]

    def test_train_and_val_dataloaders_build(self, data_root):
        datamodule = self._build_datamodule(data_root)
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader.dataset is datamodule.datasets["train"]
        assert val_loader.dataset is datamodule.datasets["valid"]

    def test_test_dataloader_builds(self, data_root):
        datamodule = self._build_datamodule(data_root)
        datamodule.setup("test")

        test_loader = datamodule.test_dataloader()

        assert test_loader.dataset is datamodule.datasets["test"]
