import pytest

from foundry.data.datamodules.openneuro import OpenNeuroDataModule

from .conftest import skip_if_missing_dataset


class TestOpenNeuroDataModule:
    @pytest.fixture
    def klinzing_dir(self):
        return "klinzing_sleep_ds005555"

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root, klinzing_dir):
        skip_marker = skip_if_missing_dataset(klinzing_dir, data_root)
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_rejects_empty_dataset_dirs(self, data_root):
        with pytest.raises(ValueError, match="at least one"):
            OpenNeuroDataModule(
                root=str(data_root),
                dataset_dirs=[],
                sequence_length=1.0,
            )

    def test_val_dataloader_uses_runtime_splits(self, data_root, klinzing_dir):
        dm = OpenNeuroDataModule(
            root=str(data_root),
            dataset_dirs=[klinzing_dir],
            sequence_length=60.0,
            keep_files_open=False,
        )
        dm.setup()
        loader = dm.val_dataloader()
        assert loader is not None
        assert len(loader) > 0

    def test_train_and_test_splits_are_non_empty(self, data_root, klinzing_dir):
        dm = OpenNeuroDataModule(
            root=str(data_root),
            dataset_dirs=[klinzing_dir],
            keep_files_open=False,
        )
        dm.setup()
        for split in ("train", "val", "test"):
            intervals = dm.dataset.get_sampling_intervals(split=split)
            assert len(intervals) > 0

    def test_namespaced_recording_ids_require_dataset_prefix(self, data_root):
        with pytest.raises(ValueError, match="namespaced"):
            dm = OpenNeuroDataModule(
                root=str(data_root),
                dataset_dirs=[
                    "klinzing_sleep_ds005555",
                    "kochi_visualnaming_ds006914",
                ],
                recording_ids=["sub-100_task-Sleep_acq-headband"],
                sequence_length=1.0,
            )
            dm.setup()

    def test_namespaced_recording_ids_subset(self, data_root, klinzing_dir):
        dm_all = OpenNeuroDataModule(
            root=str(data_root),
            dataset_dirs=[klinzing_dir, "kochi_visualnaming_ds006914"],
            keep_files_open=False,
        )
        dm_all.setup()
        subset = [
            rid
            for rid in dm_all.get_recording_ids()
            if rid.startswith(f"{klinzing_dir}/")
        ][:2]
        dm = OpenNeuroDataModule(
            root=str(data_root),
            dataset_dirs=[klinzing_dir, "kochi_visualnaming_ds006914"],
            recording_ids=subset,
            keep_files_open=False,
        )
        dm.setup()
        assert set(dm.get_recording_ids()) == set(subset)

    def test_unknown_dataset_prefix_in_recording_ids(self, data_root):
        with pytest.raises(ValueError, match="Unknown dataset prefix"):
            OpenNeuroDataModule(
                root=str(data_root),
                dataset_dirs=[
                    "klinzing_sleep_ds005555",
                    "kochi_visualnaming_ds006914",
                ],
                recording_ids=["unknown_ds/foo"],
                sequence_length=1.0,
            ).setup()
