"""Tests for ClassificationMapping schema and validation."""

from __future__ import annotations

import numpy as np
import pytest
from torch_brain.data import Interval

from foundry.tasks.classification_mapping import (
    ClassificationMapping,
    filter_intervals_by_mapping,
)
from foundry.tasks.config import TaskConfig


class TestClassificationMappingValidation:
    """Validates strict mapping invariants on construction."""

    def test_valid_mapping_constructs_successfully(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1, 2: None})
        assert mapping.raw_to_mapped == {0: 0, 1: 1, 2: None}

    def test_mapped_ids_must_be_contiguous_from_zero(self):
        with pytest.raises(ValueError, match="contiguous"):
            ClassificationMapping(raw_to_mapped={0: 0, 1: 2})

    def test_many_to_one_mapping_is_valid(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 0, 2: None})
        assert mapping.num_classes == 1

    def test_empty_raw_to_mapped_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ClassificationMapping(raw_to_mapped={})

    def test_all_null_mapping_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ClassificationMapping(raw_to_mapped={0: None, 1: None})

    def test_negative_mapped_id_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ClassificationMapping(raw_to_mapped={0: -1, 1: 0})


class TestClassificationMappingDerivedProperties:
    """Verify derived properties computed from mapping."""

    def test_num_classes_counts_distinct_non_null_mapped_ids(self):
        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1, 2: 1, 3: None}
        )
        assert mapping.num_classes == 2

    def test_class_names_from_explicit_names(self):
        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1},
            names={0: "Wake", 1: "Sleep"},
        )
        assert mapping.class_names == ["Wake", "Sleep"]

    def test_class_names_default_when_not_provided(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1, 2: 2})
        assert mapping.class_names == ["class_0", "class_1", "class_2"]

    def test_class_names_partial_fills_defaults(self):
        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1, 2: 2},
            names={0: "Wake", 2: "REM"},
        )
        assert mapping.class_names == ["Wake", "class_1", "REM"]

    def test_kept_raw_ids_returns_non_null_sources(self):
        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1, 2: None, 3: 1}
        )
        assert mapping.kept_raw_ids == {0, 1, 3}

    def test_removed_raw_ids_returns_null_sources(self):
        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1, 2: None, 3: None}
        )
        assert mapping.removed_raw_ids == {2, 3}


class TestClassificationMappingApply:
    """Verify label application behavior."""

    def test_apply_remaps_values_correctly(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 0, 2: 1, 3: 2})
        raw = np.array([0, 1, 2, 3, 0, 2], dtype=np.int64)
        result = mapping.apply(raw)
        expected = np.array([0, 0, 1, 2, 0, 1], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_apply_with_removal_returns_negative_one_for_removed(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1, 2: None})
        raw = np.array([0, 1, 2, 0], dtype=np.int64)
        result = mapping.apply(raw)
        expected = np.array([0, 1, -1, 0], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_apply_raises_on_undeclared_raw_id(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1})
        raw = np.array([0, 1, 5], dtype=np.int64)
        with pytest.raises(ValueError, match="undeclared raw"):
            mapping.apply(raw)

    def test_apply_preserves_dtype(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1})
        raw = np.array([0, 1, 0], dtype=np.int32)
        result = mapping.apply(raw)
        assert result.dtype == np.int32


class TestTaskConfigWithMapping:
    """TaskConfig derives output_dim, class_names, num_classes from mapping."""

    def test_output_dim_derived_from_mapping(self):
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=ClassificationMapping(
                raw_to_mapped={0: 0, 1: 1, 2: 2, 3: None}
            ),
        )
        assert cfg.output_dim == 3

    def test_output_dim_falls_back_to_head_without_mapping(self):
        cfg = TaskConfig(
            name="test",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 5,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        assert cfg.output_dim == 5

    def test_class_names_derived_from_mapping(self):
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=ClassificationMapping(
                raw_to_mapped={0: 0, 1: 1},
                names={0: "Wake", 1: "Sleep"},
            ),
        )
        assert cfg.get_class_names() == ["Wake", "Sleep"]

    def test_class_names_falls_back_to_field_without_mapping(self):
        cfg = TaskConfig(
            name="test",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            class_names=["A", "B"],
        )
        assert cfg.get_class_names() == ["A", "B"]

    def test_from_dict_parses_classification_mapping(self):
        data = {
            "name": "mapped_task",
            "head": {"_target_": "foundry.tasks.heads.ReadoutHead"},
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "stages.timestamps",
                "value_key": "stages.values",
            },
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            "classification_mapping": {
                "raw_to_mapped": {0: 0, 1: 1, 2: 1, 3: 2, 4: None},
                "names": {0: "Wake", 1: "Light", 2: "Deep"},
            },
        }
        cfg = TaskConfig.from_dict(data)
        assert cfg.classification_mapping is not None
        assert cfg.output_dim == 3
        assert cfg.get_class_names() == ["Wake", "Light", "Deep"]

    def test_from_dict_without_mapping_leaves_none(self):
        data = {
            "name": "legacy",
            "head": {
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 5,
            },
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
            },
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        }
        cfg = TaskConfig.from_dict(data)
        assert cfg.classification_mapping is None
        assert cfg.output_dim == 5

    def test_mapping_validation_errors_propagate_on_load(self):
        data = {
            "name": "bad",
            "head": {"_target_": "foundry.tasks.heads.ReadoutHead"},
            "target_extractor": {"timestamp_key": "t", "value_key": "v"},
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            "classification_mapping": {
                "raw_to_mapped": {0: 0, 1: 3},
            },
        }
        with pytest.raises(ValueError, match="contiguous"):
            TaskConfig.from_dict(data)

    def test_label_map_and_mapping_coexist_raises(self):
        data = {
            "name": "conflict",
            "head": {"_target_": "foundry.tasks.heads.ReadoutHead"},
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
                "label_map": {4: 3},
            },
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            "classification_mapping": {
                "raw_to_mapped": {0: 0, 1: 1, 2: 2, 3: 2, 4: None},
            },
        }
        with pytest.raises(ValueError, match="label_map.*deprecated"):
            TaskConfig.from_dict(data)

    def test_head_output_dim_with_mapping_warns_if_mismatch(self):
        """When mapping is set, head output_dim should not also be specified."""
        data = {
            "name": "dup",
            "head": {
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 99,
            },
            "target_extractor": {"timestamp_key": "t", "value_key": "v"},
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            "classification_mapping": {
                "raw_to_mapped": {0: 0, 1: 1},
            },
        }
        cfg = TaskConfig.from_dict(data)
        # Mapping overrides head output_dim
        assert cfg.output_dim == 2


class TestFilterIntervalsByMapping:
    """Split intervals filtered to only keep intervals with retained labels."""

    def test_removes_intervals_with_null_mapped_labels(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1, 2: None})
        intervals = Interval(
            start=np.array([0.0, 1.0, 2.0, 3.0]),
            end=np.array([1.0, 2.0, 3.0, 4.0]),
            values=np.array([0, 2, 1, 2]),
        )
        result = filter_intervals_by_mapping(
            intervals, mapping, value_field="values"
        )
        assert len(result.start) == 2
        np.testing.assert_array_equal(result.start, [0.0, 2.0])
        np.testing.assert_array_equal(result.end, [1.0, 3.0])

    def test_keeps_all_when_no_removals(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1, 2: 2})
        intervals = Interval(
            start=np.array([0.0, 1.0, 2.0]),
            end=np.array([1.0, 2.0, 3.0]),
            values=np.array([0, 1, 2]),
        )
        result = filter_intervals_by_mapping(
            intervals, mapping, value_field="values"
        )
        assert len(result.start) == 3

    def test_returns_empty_when_all_removed(self):
        mapping = ClassificationMapping(raw_to_mapped={0: None, 1: None, 2: 0})
        intervals = Interval(
            start=np.array([0.0, 1.0]),
            end=np.array([1.0, 2.0]),
            values=np.array([0, 1]),
        )
        result = filter_intervals_by_mapping(
            intervals, mapping, value_field="values"
        )
        assert len(result.start) == 0

    def test_passes_through_intervals_without_value_field(self):
        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: None})
        intervals = Interval(
            start=np.array([0.0, 1.0]),
            end=np.array([1.0, 2.0]),
        )
        result = filter_intervals_by_mapping(
            intervals, mapping, value_field="values"
        )
        assert len(result.start) == 2


class TestTargetExtractorWithMapping:
    """TargetExtractor applies classification_mapping when provided."""

    def test_applies_mapping_to_extracted_values(self):
        from dataclasses import dataclass

        from torch_brain.data import Data

        from foundry.tasks.targets import TargetExtractor

        @dataclass
        class _Stages:
            timestamps: np.ndarray
            values: np.ndarray

        data = Data(
            stages=_Stages(
                timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
                values=np.array([0, 1, 4, 5], dtype=np.int64),
            )
        )

        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
        )
        extractor = TargetExtractor(
            timestamp_key="stages.timestamps",
            value_key="stages.values",
            classification_mapping=mapping,
        )
        result = extractor(data)
        expected = np.array([0, 1, 3, 4], dtype=np.int64)
        assert np.array_equal(result["values"], expected)

    def test_raises_on_undeclared_raw_id_in_extraction(self):
        from dataclasses import dataclass

        from torch_brain.data import Data

        from foundry.tasks.targets import TargetExtractor

        @dataclass
        class _Stages:
            timestamps: np.ndarray
            values: np.ndarray

        data = Data(
            stages=_Stages(
                timestamps=np.array([0.0, 1.0]),
                values=np.array([0, 99], dtype=np.int64),
            )
        )

        mapping = ClassificationMapping(raw_to_mapped={0: 0, 1: 1})
        extractor = TargetExtractor(
            timestamp_key="stages.timestamps",
            value_key="stages.values",
            classification_mapping=mapping,
        )
        with pytest.raises(ValueError, match="undeclared raw"):
            extractor(data)


class TestClassWeightsWithMapping:
    """Class weight computation uses classification_mapping when present."""

    def test_weights_computed_on_mapped_labels(self):
        """Mapping {0:0, 1:0, 2:1, 3:None} means mapped-0 gets 2 units,
        mapped-1 gets 1 unit, and raw-3 is excluded."""
        from foundry.tasks.class_weights import compute_class_weights_for_tasks

        mapping = ClassificationMapping(
            raw_to_mapped={0: 0, 1: 0, 2: 1, 3: None}
        )
        cfg = TaskConfig(
            name="test_clf",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "trials.timestamps",
                "value_key": "trials.values",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=mapping,
        )

        class _MockDataset:
            def get_sampling_intervals(self, split):
                # Each interval is 1 second. Raw labels: 0, 1, 2, 3.
                # After mapping: 0→0(1s), 1→0(1s), 2→1(1s), 3→removed
                return {
                    "rec1": Interval(
                        start=np.array([0.0, 1.0, 2.0, 3.0]),
                        end=np.array([1.0, 2.0, 3.0, 4.0]),
                        values=np.array([0, 1, 2, 3]),
                    )
                }

        weights = compute_class_weights_for_tasks(
            {"test_clf": cfg}, _MockDataset(), split="train"
        )
        assert "test_clf" in weights
        assert len(weights["test_clf"]) == 2
        # mapped-0 has 2 units of time (raw 0 + raw 1), mapped-1 has 1 unit
        # inverse frequency: total=3, weights proportional to 3/(2*count)
        # class 0: 3/(2*2) = 0.75, class 1: 3/(2*1) = 1.5
        assert weights["test_clf"][0] < weights["test_clf"][1]
