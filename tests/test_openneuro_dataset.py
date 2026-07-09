"""Tests for the OpenNeuroMultiBrainset dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import pytest
from omegaconf import OmegaConf
from torch_brain.datasets import (
    KlinzingSleepDS005555,
    KochiVisualNamingDS006914,
    NestedDataset,
    ShiraziHBNR1DS005505,
)

from foundry.data.datasets import OpenNeuroMultiBrainset
from foundry.data.datasets.openneuro import OPENNEURO_BRAINSET_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"


def _create_stub_h5(root: Path, brainset: str, recording_id: str) -> None:
    """Create a minimal .h5 stub so torch_brain's file-existence check passes."""
    brainset_dir = root / brainset
    brainset_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(brainset_dir / f"{recording_id}.h5", "w") as f:
        f.attrs["absolute_start"] = 0.0


# ---------------------------------------------------------------------------
# QA 1a: Registry validation
# ---------------------------------------------------------------------------
class TestRegistryValidation:
    def test_valid_brainset_names_no_error(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
        )
        assert isinstance(ds, NestedDataset)

    def test_invalid_brainset_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown brainset"):
            OpenNeuroMultiBrainset(
                root=str(tmp_path),
                brainsets=["nonexistent_brainset"],
                split_type="intrasession",
            )

    def test_empty_brainset_list_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            OpenNeuroMultiBrainset(
                root=str(tmp_path),
                brainsets=[],
                split_type="intrasession",
            )


# ---------------------------------------------------------------------------
# QA 1b: Constructor forwarding
# ---------------------------------------------------------------------------
class TestConstructorForwarding:
    def test_single_brainset_child_type(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert isinstance(child, KlinzingSleepDS005555)

    def test_multiple_brainsets_keys(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555", "shirazi_hbnr1_ds005505"],
            split_type="intrasession",
            recording_ids={
                "klinzing_sleep_ds005555": [],
                "shirazi_hbnr1_ds005505": [],
            },
        )
        assert set(ds.datasets.keys()) == {
            "klinzing_sleep_ds005555",
            "shirazi_hbnr1_ds005505",
        }
        assert isinstance(
            ds.datasets["shirazi_hbnr1_ds005505"], ShiraziHBNR1DS005505
        )


# ---------------------------------------------------------------------------
# QA 1c: split_type forwarding
# ---------------------------------------------------------------------------
class TestSplitTypeForwarding:
    @pytest.mark.parametrize(
        "split_type", ["intrasession", "intersubject", "intersession"]
    )
    def test_split_type_propagated_to_children(self, tmp_path, split_type):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type=split_type,
            recording_ids={"klinzing_sleep_ds005555": []},
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert child.split_type == split_type


# ---------------------------------------------------------------------------
# QA 1d: recording_ids per-brainset filtering
# ---------------------------------------------------------------------------
class TestRecordingIdsFiltering:
    def test_per_brainset_recording_ids_forwarded(self, tmp_path):
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec1")
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": ["rec1"]},
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert child.recording_ids == ["rec1"]

    def test_none_recording_ids_discovers_from_disk(self, tmp_path):
        """When recording_ids is None, child datasets discover recordings from disk."""
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec_a")
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec_b")
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids=None,
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert sorted(child.recording_ids) == ["rec_a", "rec_b"]


# ---------------------------------------------------------------------------
# QA 1e: split_ratios forwarding
# ---------------------------------------------------------------------------
class TestSplitRatiosForwarding:
    def test_custom_split_ratios_forwarded(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
            split_ratios=(0.7, 0.15, 0.15),
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert child.split_ratios == (0.7, 0.15, 0.15)

    def test_default_split_ratios(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
        )
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert child.split_ratios == (0.8, 0.1, 0.1)


# ---------------------------------------------------------------------------
# QA 1f: task_type consumption
# ---------------------------------------------------------------------------
class TestTaskTypeConsumption:
    def test_task_type_does_not_raise(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
            task_type="some_task",
        )
        assert isinstance(ds, OpenNeuroMultiBrainset)


# ---------------------------------------------------------------------------
# QA 1g: Split name mapping ("valid" -> "val")
# ---------------------------------------------------------------------------
class TestSplitNameMapping:
    def _make_ds(self, tmp_path):
        return OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
        )

    def test_valid_mapped_to_val(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch.object(
            NestedDataset, "get_sampling_intervals", return_value={}
        ) as mock:
            ds.get_sampling_intervals(split="valid")
            mock.assert_called_once_with(split="val")

    def test_train_passes_through(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch.object(
            NestedDataset, "get_sampling_intervals", return_value={}
        ) as mock:
            ds.get_sampling_intervals(split="train")
            mock.assert_called_once_with(split="train")

    def test_test_passes_through(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch.object(
            NestedDataset, "get_sampling_intervals", return_value={}
        ) as mock:
            ds.get_sampling_intervals(split="test")
            mock.assert_called_once_with(split="test")

    def test_none_passes_through(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch.object(
            NestedDataset, "get_sampling_intervals", return_value={}
        ) as mock:
            ds.get_sampling_intervals(split=None)
            mock.assert_called_once_with(split=None)


# ---------------------------------------------------------------------------
# QA 1h: get_channel_ids() aggregation
# ---------------------------------------------------------------------------
class TestGetChannelIds:
    def test_empty_recordings_returns_empty(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555", "shirazi_hbnr1_ds005505"],
            split_type="intrasession",
            recording_ids={
                "klinzing_sleep_ds005555": [],
                "shirazi_hbnr1_ds005505": [],
            },
        )
        assert ds.get_channel_ids() == []

    def test_aggregates_across_datasets(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555", "shirazi_hbnr1_ds005505"],
            split_type="intrasession",
            recording_ids={
                "klinzing_sleep_ds005555": [],
                "shirazi_hbnr1_ds005505": [],
            },
        )
        with (
            patch.object(
                KlinzingSleepDS005555,
                "get_channel_ids",
                return_value=["ch_a", "ch_b"],
            ),
            patch.object(
                ShiraziHBNR1DS005505,
                "get_channel_ids",
                return_value=["ch_b", "ch_c"],
            ),
            patch.object(
                KlinzingSleepDS005555,
                "recording_ids",
                new_callable=lambda: property(lambda self: ["fake"]),
            ),
            patch.object(
                ShiraziHBNR1DS005505,
                "recording_ids",
                new_callable=lambda: property(lambda self: ["fake"]),
            ),
        ):
            result = ds.get_channel_ids()
            assert result == [
                "klinzing_sleep_ds005555/ch_a",
                "klinzing_sleep_ds005555/ch_b",
                "shirazi_hbnr1_ds005505/ch_b",
                "shirazi_hbnr1_ds005505/ch_c",
            ]


# ---------------------------------------------------------------------------
# QA 1i: recording_ids namespacing
# ---------------------------------------------------------------------------
class TestRecordingIdsNamespacing:
    def test_recording_ids_have_prefix(self, tmp_path):
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec1")
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec2")
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": ["rec1", "rec2"]},
        )
        for rid in ds.recording_ids:
            assert rid.startswith("klinzing_sleep_ds005555/")

    def test_multi_brainset_namespacing(self, tmp_path):
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec1")
        _create_stub_h5(tmp_path, "shirazi_hbnr1_ds005505", "recA")
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555", "shirazi_hbnr1_ds005505"],
            split_type="intrasession",
            recording_ids={
                "klinzing_sleep_ds005555": ["rec1"],
                "shirazi_hbnr1_ds005505": ["recA"],
            },
        )
        ids = ds.recording_ids
        assert any(r.startswith("klinzing_sleep_ds005555/") for r in ids)
        assert any(r.startswith("shirazi_hbnr1_ds005505/") for r in ids)


# ---------------------------------------------------------------------------
# QA 1j: transform passthrough
# ---------------------------------------------------------------------------
class TestTransformPassthrough:
    def test_transform_stored_on_nested(self, tmp_path):
        dummy_transform = lambda x: x  # noqa: E731
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=["klinzing_sleep_ds005555"],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
            transform=dummy_transform,
        )
        assert ds.transform is dummy_transform
        child = ds.datasets["klinzing_sleep_ds005555"]
        assert child.transform is None


# ---------------------------------------------------------------------------
# QA 2: Hydra Config
# ---------------------------------------------------------------------------
class TestHydraConfig:
    def test_config_loads(self):
        cfg = OmegaConf.load(
            CONFIGS_ROOT / "data" / "openneuro" / "multi_brainset.yaml"
        )
        assert cfg._target_ == "foundry.data.datamodules.NeuralDataModule"
        assert (
            cfg.dataset_class == "foundry.data.datasets.OpenNeuroMultiBrainset"
        )

    def test_dataset_kwargs_has_brainsets(self):
        cfg = OmegaConf.load(
            CONFIGS_ROOT / "data" / "openneuro" / "multi_brainset.yaml"
        )
        assert "brainsets" in cfg.dataset_kwargs
        brainsets = list(cfg.dataset_kwargs.brainsets)
        assert "klinzing_sleep_ds005555" in brainsets


# ---------------------------------------------------------------------------
# QA 3: Integration with NeuralDataModule
# ---------------------------------------------------------------------------
class TestNeuralDataModuleIntegration:
    def test_setup_creates_correct_dataset(self, tmp_path):
        from foundry.data.datamodules import NeuralDataModule

        dm = NeuralDataModule(
            dataset_class=OpenNeuroMultiBrainset,
            root=str(tmp_path),
            batch_size=2,
            sequence_length=2.0,
            dataset_kwargs={
                "brainsets": ["klinzing_sleep_ds005555"],
                "split_type": "intrasession",
                "recording_ids": {"klinzing_sleep_ds005555": []},
            },
        )
        dm.setup()
        assert isinstance(dm.dataset, OpenNeuroMultiBrainset)

    def test_get_recording_ids_namespaced(self, tmp_path):
        from foundry.data.datamodules import NeuralDataModule

        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec1")
        _create_stub_h5(tmp_path, "klinzing_sleep_ds005555", "rec2")
        dm = NeuralDataModule(
            dataset_class=OpenNeuroMultiBrainset,
            root=str(tmp_path),
            batch_size=2,
            sequence_length=2.0,
            dataset_kwargs={
                "brainsets": ["klinzing_sleep_ds005555"],
                "split_type": "intrasession",
                "recording_ids": {"klinzing_sleep_ds005555": ["rec1", "rec2"]},
            },
        )
        dm.setup()
        ids = dm.get_recording_ids()
        assert all("/" in rid for rid in ids)

    def test_get_channel_ids_no_raise(self, tmp_path):
        from foundry.data.datamodules import NeuralDataModule

        dm = NeuralDataModule(
            dataset_class=OpenNeuroMultiBrainset,
            root=str(tmp_path),
            batch_size=2,
            sequence_length=2.0,
            dataset_kwargs={
                "brainsets": ["klinzing_sleep_ds005555"],
                "split_type": "intrasession",
                "recording_ids": {"klinzing_sleep_ds005555": []},
            },
        )
        dm.setup()
        result = dm.get_channel_ids()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# QA 4: Import and Export
# ---------------------------------------------------------------------------
class TestImportExport:
    def test_import_from_datasets(self):
        from foundry.data.datasets import OpenNeuroMultiBrainset as _cls

        assert _cls is OpenNeuroMultiBrainset

    def test_import_registry(self):
        from foundry.data.datasets.openneuro import (
            OPENNEURO_BRAINSET_REGISTRY as reg,
        )

        assert "klinzing_sleep_ds005555" in reg
        assert "shirazi_hbnr1_ds005505" in reg
        assert "kochi_visualnaming_ds006914" in reg

    def test_all_includes_class(self):
        import foundry.data.datasets as mod

        assert "OpenNeuroMultiBrainset" in mod.__all__


# ---------------------------------------------------------------------------
# QA 5: Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_duplicate_brainset_names(self, tmp_path):
        ds = OpenNeuroMultiBrainset(
            root=str(tmp_path),
            brainsets=[
                "klinzing_sleep_ds005555",
                "klinzing_sleep_ds005555",
            ],
            split_type="intrasession",
            recording_ids={"klinzing_sleep_ds005555": []},
        )
        # Duplicate key in dict → only one child dataset
        assert len(ds.datasets) == 1

    def test_registry_has_expected_entries(self):
        assert set(OPENNEURO_BRAINSET_REGISTRY.keys()) == {
            "klinzing_sleep_ds005555",
            "shirazi_hbnr1_ds005505",
            "kochi_visualnaming_ds006914",
        }

    def test_registry_maps_to_correct_classes(self):
        assert (
            OPENNEURO_BRAINSET_REGISTRY["klinzing_sleep_ds005555"]
            is KlinzingSleepDS005555
        )
        assert (
            OPENNEURO_BRAINSET_REGISTRY["shirazi_hbnr1_ds005505"]
            is ShiraziHBNR1DS005505
        )
        assert (
            OPENNEURO_BRAINSET_REGISTRY["kochi_visualnaming_ds006914"]
            is KochiVisualNamingDS006914
        )
