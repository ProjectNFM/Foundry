"""Tests for datamodule-driven effective readout specs (Option B).

Verifies that effective readout specs correctly adapt to datamodule config
(e.g., data.classes frequency filtering in NeuroSoft).
"""

import numpy as np
import pytest
from temporaldata import Interval
from torch_brain.registry import MODALITY_REGISTRY

from foundry.data.datamodules.neurosoft import (
    NeurosoftDataModule,
    build_acoustic_stim_label_mapping,
)
from foundry.data.datasets.modalities import MappedCrossEntropyLoss
from foundry.data.readout_specs import clone_readout_spec


class TestBuildAcousticStimLabelMapping:
    """Unit tests for build_acoustic_stim_label_mapping helper."""

    def test_none_classes_returns_no_mapping(self):
        """With classes=None, no band collapse mapping should be created."""
        mapping, bands = build_acoustic_stim_label_mapping(None)
        assert mapping == {}
        assert bands == []

    def test_all_frequencies_3_bands(self):
        """With 6 freqs (2 per band), mapping should collapse to 3 bands."""
        classes = [
            "stim_500Hz",
            "stim_800Hz",  # low (raw IDs 4,6)
            "stim_1000Hz",
            "stim_2000Hz",  # medium (raw IDs 7,11)
            "stim_5000Hz",
            "stim_8000Hz",  # high (raw IDs 14,16)
        ]
        from foundry.data.datamodules.neurosoft import FREQ_GROUPINGS
        mapping, bands = build_acoustic_stim_label_mapping(
            classes, freq_grouping=FREQ_GROUPINGS["3band"]
        )

        # Verify band names in LABEL_TO_ID order
        assert bands == ["low", "medium", "high"]

        # Verify all raw IDs are mapped
        assert 4 in mapping and mapping[4] == 0  # stim_500Hz -> low
        assert 6 in mapping and mapping[6] == 0  # stim_800Hz -> low
        assert 7 in mapping and mapping[7] == 1  # stim_1000Hz -> medium
        assert 11 in mapping and mapping[11] == 1  # stim_2000Hz -> medium
        assert 14 in mapping and mapping[14] == 2  # stim_5000Hz -> high
        assert 16 in mapping and mapping[16] == 2  # stim_8000Hz -> high

    def test_subset_frequencies_partial_bands(self):
        """With subset of frequencies, only used bands appear in output."""
        classes = ["stim_500Hz", "stim_1000Hz", "stim_5000Hz"]  # low, medium, high
        from foundry.data.datamodules.neurosoft import FREQ_GROUPINGS
        mapping, bands = build_acoustic_stim_label_mapping(
            classes, freq_grouping=FREQ_GROUPINGS["3band"]
        )

        assert bands == ["low", "medium", "high"]
        assert 4 in mapping and mapping[4] == 0  # stim_500Hz (id=4) -> low
        assert 7 in mapping and mapping[7] == 1  # stim_1000Hz (id=7) -> medium
        assert 14 in mapping and mapping[14] == 2  # stim_5000Hz (id=14) -> high

    def test_single_frequency_single_band(self):
        """With single frequency, effective dim=1."""
        classes = ["stim_500Hz"]
        from foundry.data.datamodules.neurosoft import FREQ_GROUPINGS
        mapping, bands = build_acoustic_stim_label_mapping(
            classes, freq_grouping=FREQ_GROUPINGS["3band"]
        )

        assert bands == ["low"]
        # stim_500Hz is at index 4 in STIM_FREQUENCY_TO_ID
        assert 4 in mapping and mapping[4] == 0

    def test_two_bands_only(self):
        """With only low and high, medium is omitted."""
        classes = ["stim_500Hz", "stim_5000Hz"]  # low and high
        from foundry.data.datamodules.neurosoft import FREQ_GROUPINGS
        mapping, bands = build_acoustic_stim_label_mapping(
            classes, freq_grouping=FREQ_GROUPINGS["3band"]
        )

        assert bands == ["low", "high"]
        # stim_500Hz id=4 -> low (band_id=0)
        # stim_5000Hz id=14 -> high (band_id=1 since medium omitted)
        assert 4 in mapping and mapping[4] == 0  # low
        assert 14 in mapping and mapping[14] == 1  # high

    def test_invalid_frequency_raises(self):
        """Invalid frequency names should raise ValueError."""
        classes = ["invalid_freq"]
        with pytest.raises(ValueError, match="No valid frequencies found"):
            build_acoustic_stim_label_mapping(classes)


class TestReadoutSpecCloning:
    """Tests for clone_readout_spec helper."""

    def test_clone_preserves_base_fields(self):
        """Cloned spec should preserve routing fields."""
        base = MODALITY_REGISTRY["neurosoft_acoustic_stim"]
        cloned = clone_readout_spec(base, dim=3)

        # Routing fields unchanged
        assert cloned.id == base.id
        assert cloned.type == base.type
        assert cloned.timestamp_key == base.timestamp_key
        assert cloned.value_key == base.value_key

        # Dimension updated
        assert cloned.dim == 3
        assert base.dim == 26  # Original unchanged

    def test_clone_with_loss_fn(self):
        """Cloned spec can override loss_fn."""
        base = MODALITY_REGISTRY["neurosoft_acoustic_stim"]
        mapping = {0: 0, 1: 0, 2: 1}
        loss_fn = MappedCrossEntropyLoss(mapping)

        cloned = clone_readout_spec(base, dim=2, loss_fn=loss_fn)

        assert cloned.dim == 2
        assert isinstance(cloned.loss_fn, MappedCrossEntropyLoss)
        assert cloned.id == base.id


class TestNeurosoftEffectiveSpecs:
    """Tests for NeurosoftDataModule effective spec methods (integration)."""

    def test_no_classes_returns_registry_spec(self):
        """Without data.classes, should return registry spec unchanged."""
        dm = NeurosoftDataModule(
            dataset_class=None,  # Mock for this test
            root="/tmp",
            task_type="acoustic_stim",
            classes=None,
        )

        specs = dm.get_effective_readout_specs()

        assert "neurosoft_acoustic_stim" in specs
        spec = specs["neurosoft_acoustic_stim"]
        # Should match registry
        assert spec.dim == MODALITY_REGISTRY["neurosoft_acoustic_stim"].dim

    def test_with_classes_returns_mapped_spec(self):
        """With data.classes set, should return cloned spec with mapping."""
        dm = NeurosoftDataModule(
            dataset_class=None,  # Mock for this test
            root="/tmp",
            task_type="acoustic_stim",
            classes=[
                "stim_500Hz",
                "stim_800Hz",
                "stim_1000Hz",
                "stim_2000Hz",
                "stim_5000Hz",
                "stim_8000Hz",
            ],
            freq_grouping="3band",
        )

        specs = dm.get_effective_readout_specs()

        assert "neurosoft_acoustic_stim" in specs
        spec = specs["neurosoft_acoustic_stim"]
        # Should be modified: 3 bands
        assert spec.dim == 3
        assert isinstance(spec.loss_fn, MappedCrossEntropyLoss)

    def test_effective_class_names_no_filtering(self):
        """Without filtering, should return full 26 frequency names with stim_ stripped."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=None,
        )

        names = dm.get_effective_class_names_for_task("acoustic_stim")

        assert "neurosoft_acoustic_stim" in names
        # Should have all 26 frequencies with stim_ stripped
        assert len(names["neurosoft_acoustic_stim"]) == 26
        # Check a few examples have stim_ removed
        assert "100Hz" in names["neurosoft_acoustic_stim"]
        assert "500Hz" in names["neurosoft_acoustic_stim"]
        assert "wn" in names["neurosoft_acoustic_stim"]

    def test_effective_class_names_with_filtering(self):
        """With filtering and grouping, should return only band names (3)."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=[
                "stim_500Hz",
                "stim_800Hz",
                "stim_1000Hz",
                "stim_2000Hz",
                "stim_5000Hz",
                "stim_8000Hz",
            ],
            freq_grouping="3band",
        )

        names = dm.get_effective_class_names_for_task("acoustic_stim")

        assert "neurosoft_acoustic_stim" in names
        assert names["neurosoft_acoustic_stim"] == ["low", "medium", "high"]

    def test_on_vs_off_unaffected_by_classes(self):
        """on_vs_off task should not be affected by acoustic_stim classes."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="on_vs_off",
            classes=["stim_500Hz"],  # Should be ignored for on_vs_off
        )

        specs = dm.get_effective_readout_specs()

        # Should return registry spec unchanged
        assert "neurosoft_on_vs_off" in specs
        registry_spec = MODALITY_REGISTRY["neurosoft_on_vs_off"]
        assert specs["neurosoft_on_vs_off"].dim == registry_spec.dim

    def test_raw_mode_subset_frequencies(self):
        """With classes but no freq_grouping, should return dense 0..N-1 mapping."""
        classes = ["stim_500Hz", "stim_1000Hz", "stim_5000Hz"]
        mapping, names = build_acoustic_stim_label_mapping(classes, freq_grouping=None)

        # Should have dense remap: raw_id -> 0..N-1
        # stim_500Hz=4, stim_1000Hz=7, stim_5000Hz=14
        assert mapping == {4: 0, 7: 1, 14: 2}
        assert names == ["stim_500Hz", "stim_1000Hz", "stim_5000Hz"]

    def test_raw_mode_effective_specs(self):
        """Raw mode (freq_grouping=None) should create cloned spec with dim=N."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=["stim_500Hz", "stim_1000Hz", "stim_5000Hz"],
            freq_grouping=None,
        )

        specs = dm.get_effective_readout_specs()

        assert "neurosoft_acoustic_stim" in specs
        spec = specs["neurosoft_acoustic_stim"]
        assert spec.dim == 3  # 3 frequencies selected
        assert isinstance(spec.loss_fn, MappedCrossEntropyLoss)

    def test_raw_mode_display_names_strip_prefix(self):
        """Confusion matrix names should strip stim_ prefix."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=["stim_500Hz", "stim_1000Hz", "stim_5000Hz"],
            freq_grouping=None,
        )

        names = dm.get_effective_class_names_for_task("acoustic_stim")

        assert "neurosoft_acoustic_stim" in names
        # Should have stim_ prefix removed for display
        assert names["neurosoft_acoustic_stim"] == ["500Hz", "1000Hz", "5000Hz"]

    def test_grouped_mode_band_names_unchanged(self):
        """Band names should not have stim_ prefix to remove."""
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=["stim_500Hz", "stim_1000Hz", "stim_5000Hz"],
            freq_grouping="3band",
        )

        names = dm.get_effective_class_names_for_task("acoustic_stim")

        assert "neurosoft_acoustic_stim" in names
        # Band names unchanged
        assert names["neurosoft_acoustic_stim"] == ["low", "medium", "high"]


class _MockNeurosoftDataset:
    def __init__(self, recording_ids, intervals_by_rid):
        self.recording_ids = recording_ids
        self._intervals_by_rid = intervals_by_rid

    def get_sampling_intervals(self, split):
        del split
        return self._intervals_by_rid


class TestValidateBinaryClassCoverage:
    """Tests for early binary-class frequency coverage validation."""

    def _make_dm(self, classes, intervals_by_rid):
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=classes,
            freq_grouping=None,
        )
        dm.dataset = _MockNeurosoftDataset(
            recording_ids=list(intervals_by_rid.keys()),
            intervals_by_rid=intervals_by_rid,
        )
        return dm

    def test_passes_when_both_frequencies_present(self):
        intervals = {
            "rec-a": Interval(
                start=np.array([0.0, 1.0]),
                end=np.array([0.5, 1.5]),
                behavior_labels=np.array(["stim_500Hz", "stim_800Hz"]),
            )
        }
        dm = self._make_dm(["stim_500Hz", "stim_800Hz"], intervals)
        dm.validate_binary_class_coverage()

    def test_raises_when_one_frequency_missing(self):
        intervals = {
            "rec-a": Interval(
                start=np.array([0.0]),
                end=np.array([0.5]),
                behavior_labels=np.array(["stim_500Hz"]),
            )
        }
        dm = self._make_dm(["stim_500Hz", "stim_800Hz"], intervals)
        with pytest.raises(ValueError, match="missing \\['stim_800Hz'\\]"):
            dm.validate_binary_class_coverage()

    def test_raises_when_no_trials_for_either_class(self):
        intervals = {
            "rec-a": Interval(
                start=np.array([]),
                end=np.array([]),
                behavior_labels=np.array([], dtype="U"),
            )
        }
        dm = self._make_dm(["stim_500Hz", "stim_800Hz"], intervals)
        with pytest.raises(ValueError, match="missing"):
            dm.validate_binary_class_coverage()

    def test_skips_non_binary_runs(self):
        intervals = {
            "rec-a": Interval(
                start=np.array([0.0]),
                end=np.array([0.5]),
                behavior_labels=np.array(["stim_500Hz"]),
            )
        }
        dm = self._make_dm(["stim_500Hz", "stim_800Hz", "stim_1000Hz"], intervals)
        dm.validate_binary_class_coverage()

    def test_skips_when_classes_not_set(self):
        dm = NeurosoftDataModule(
            dataset_class=None,
            root="/tmp",
            task_type="acoustic_stim",
            classes=None,
        )
        dm.dataset = _MockNeurosoftDataset(recording_ids=[], intervals_by_rid={})
        dm.validate_binary_class_coverage()
