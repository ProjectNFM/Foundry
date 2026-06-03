import pytest
from brainsets.datasets import OpenNeuroDataset

from foundry.data.utils import get_max_channels, get_sampling_rate

from .conftest import skip_if_missing_dataset


class TestDataUtilsOpenNeuro:
    @pytest.fixture
    def kochi_dir(self):
        return "kochi_visualnaming_ds006914"

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self, data_root, kochi_dir):
        skip_marker = skip_if_missing_dataset(kochi_dir, data_root)
        if skip_marker.args[0]:
            pytest.skip(skip_marker.kwargs["reason"])

    def test_get_sampling_rate_from_ieeg_recording(self, data_root, kochi_dir):
        dataset = OpenNeuroDataset(
            root=str(data_root),
            dataset_dir=kochi_dir,
            split_type="intrasession",
            keep_files_open=False,
        )
        rate = get_sampling_rate(dataset)
        assert rate > 0

    def test_get_max_channels_counts_ieeg_channels(self, data_root, kochi_dir):
        dataset = OpenNeuroDataset(
            root=str(data_root),
            dataset_dir=kochi_dir,
            split_type="intrasession",
            keep_files_open=False,
        )
        assert get_max_channels(dataset) > 0
