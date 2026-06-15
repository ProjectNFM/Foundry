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

    def test_valid_int_mapping_constructs(self):
        m = ClassificationMapping(mapping={0: "Wake", 1: "N1", 2: "N2"})
        assert m.num_classes == 3

    def test_valid_str_mapping_constructs(self):
        m = ClassificationMapping(mapping={"a": "ClassA", "b": "ClassB"})
        assert m.num_classes == 2

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ClassificationMapping(mapping={})

    def test_mixed_key_types_raises(self):
        with pytest.raises(ValueError, match="same type"):
            ClassificationMapping(mapping={0: "A", "b": "B"})

    def test_order_must_contain_all_unique_values(self):
        with pytest.raises(ValueError, match="order"):
            ClassificationMapping(
                mapping={0: "Wake", 1: "Sleep"}, order=["Wake"]
            )

    def test_order_must_not_have_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            ClassificationMapping(
                mapping={0: "Wake", 1: "Sleep"},
                order=["Wake", "Sleep", "Wake"],
            )

    def test_order_must_not_have_extra_names(self):
        with pytest.raises(ValueError, match="order"):
            ClassificationMapping(
                mapping={0: "Wake", 1: "Sleep"},
                order=["Wake", "Sleep", "Extra"],
            )

    def test_many_to_one_mapping_is_valid(self):
        m = ClassificationMapping(mapping={0: "A", 1: "A", 2: "B"})
        assert m.num_classes == 2


class TestClassificationMappingDerivedProperties:
    """Verify derived properties computed from mapping."""

    def test_num_classes_counts_unique_output_names(self):
        m = ClassificationMapping(
            mapping={0: "Wake", 1: "N1", 2: "N1", 3: "N2"}
        )
        assert m.num_classes == 3

    def test_class_names_first_appearance_order(self):
        m = ClassificationMapping(
            mapping={0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}
        )
        assert m.class_names == ["Wake", "N1", "N2", "N3", "REM"]

    def test_class_names_with_explicit_order(self):
        m = ClassificationMapping(
            mapping={0: "Wake", 1: "Sleep"},
            order=["Sleep", "Wake"],
        )
        assert m.class_names == ["Sleep", "Wake"]

    def test_kept_inputs_returns_all_mapping_keys(self):
        m = ClassificationMapping(
            mapping={4: "low", 6: "low", 14: "high", 16: "high"}
        )
        assert m.kept_inputs == frozenset({4, 6, 14, 16})

    def test_kept_inputs_str_keys(self):
        m = ClassificationMapping(mapping={"a": "X", "b": "Y"})
        assert m.kept_inputs == frozenset({"a", "b"})


class TestClassificationMappingApply:
    """Verify label application behavior."""

    def test_apply_int_labels(self):
        m = ClassificationMapping(
            mapping={0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}
        )
        raw = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        result = m.apply(raw)
        expected = np.array([0, 1, 2, 3, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_apply_unlisted_labels_get_negative_one(self):
        m = ClassificationMapping(
            mapping={4: "low", 6: "low", 14: "high", 16: "high"}
        )
        raw = np.array([0, 4, 6, 10, 14, 16, 25], dtype=np.int64)
        result = m.apply(raw)
        expected = np.array([-1, 0, 0, -1, 1, 1, -1], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_apply_str_labels(self):
        m = ClassificationMapping(
            mapping={"cat": "animal", "dog": "animal", "rose": "plant"}
        )
        raw = np.array(["cat", "dog", "rose", "cat"])
        result = m.apply(raw)
        expected = np.array([0, 0, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_apply_str_unlisted_get_negative_one(self):
        m = ClassificationMapping(mapping={"cat": "animal", "dog": "animal"})
        raw = np.array(["cat", "fish", "dog"])
        result = m.apply(raw)
        expected = np.array([0, -1, 0], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_apply_with_explicit_order(self):
        m = ClassificationMapping(
            mapping={4: "low", 14: "high"},
            order=["high", "low"],
        )
        raw = np.array([4, 14], dtype=np.int64)
        result = m.apply(raw)
        expected = np.array([1, 0], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)


class TestClassificationMappingKeptMask:
    """Verify kept_mask behavior."""

    def test_kept_mask_int(self):
        m = ClassificationMapping(mapping={4: "low", 6: "low", 14: "high"})
        raw = np.array([0, 4, 6, 10, 14, 25], dtype=np.int64)
        mask = m.kept_mask(raw)
        expected = np.array([False, True, True, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_kept_mask_str(self):
        m = ClassificationMapping(mapping={"cat": "animal", "dog": "animal"})
        raw = np.array(["cat", "fish", "dog", "bird"])
        mask = m.kept_mask(raw)
        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask, expected)


class TestFilterAndRemap:
    """Tests for the atomic filter_and_remap() method."""

    def test_basic_filter_and_remap(self):
        m = ClassificationMapping(mapping={0: "A", 1: "B", 3: "B"})
        values = np.array([0, 1, 2, 3, 2, 0], dtype=np.int64)
        mapped, keep = m.filter_and_remap(values)
        np.testing.assert_array_equal(
            keep, [True, True, False, True, False, True]
        )
        np.testing.assert_array_equal(mapped, [0, 1, 1, 0])

    def test_all_kept(self):
        m = ClassificationMapping(mapping={0: "A", 1: "B", 2: "C"})
        values = np.array([0, 1, 2], dtype=np.int64)
        mapped, keep = m.filter_and_remap(values)
        assert keep.all()
        np.testing.assert_array_equal(mapped, [0, 1, 2])

    def test_all_filtered(self):
        m = ClassificationMapping(mapping={5: "X"})
        values = np.array([0, 1, 2], dtype=np.int64)
        mapped, keep = m.filter_and_remap(values)
        assert not keep.any()
        assert len(mapped) == 0


class TestFilterIntervalsByMapping:
    """Intervals filtered to only keep intervals with retained labels."""

    def test_removes_unlisted_labels(self):
        m = ClassificationMapping(mapping={0: "A", 1: "B"})
        intervals = Interval(
            start=np.array([0.0, 1.0, 2.0, 3.0]),
            end=np.array([1.0, 2.0, 3.0, 4.0]),
            values=np.array([0, 2, 1, 2]),
        )
        result = filter_intervals_by_mapping(intervals, m, value_field="values")
        assert len(result.start) == 2
        np.testing.assert_array_equal(result.start, [0.0, 2.0])
        np.testing.assert_array_equal(result.end, [1.0, 3.0])

    def test_keeps_all_when_all_in_mapping(self):
        m = ClassificationMapping(mapping={0: "A", 1: "B", 2: "C"})
        intervals = Interval(
            start=np.array([0.0, 1.0, 2.0]),
            end=np.array([1.0, 2.0, 3.0]),
            values=np.array([0, 1, 2]),
        )
        result = filter_intervals_by_mapping(intervals, m, value_field="values")
        assert len(result.start) == 3

    def test_passes_through_intervals_without_value_field(self):
        m = ClassificationMapping(mapping={0: "A", 1: "B"})
        intervals = Interval(
            start=np.array([0.0, 1.0]),
            end=np.array([1.0, 2.0]),
        )
        result = filter_intervals_by_mapping(intervals, m, value_field="values")
        assert len(result.start) == 2


class TestFromDict:
    """Test from_dict parsing of new YAML format."""

    def test_basic_int_mapping(self):
        data = {"mapping": {0: "Wake", 1: "N1", 2: "N2"}}
        m = ClassificationMapping.from_dict(data)
        assert m.num_classes == 3
        assert m.class_names == ["Wake", "N1", "N2"]

    def test_with_string_keys_from_yaml(self):
        """YAML often produces string keys for int values."""
        data = {"mapping": {"0": "Wake", "1": "N1", "2": "N2"}}
        m = ClassificationMapping.from_dict(data)
        assert m.kept_inputs == frozenset({0, 1, 2})

    def test_with_explicit_order(self):
        data = {
            "mapping": {4: "low", 14: "high"},
            "order": ["high", "low"],
        }
        m = ClassificationMapping.from_dict(data)
        assert m.class_names == ["high", "low"]

    def test_str_mapping_non_numeric_keys(self):
        data = {"mapping": {"cat": "animal", "dog": "animal", "rose": "plant"}}
        m = ClassificationMapping.from_dict(data)
        assert m.kept_inputs == frozenset({"cat", "dog", "rose"})
        assert m.num_classes == 2


class TestTaskConfigWithMapping:
    """TaskConfig derives output_dim, class_names from new mapping."""

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
                mapping={0: "Wake", 1: "N1", 2: "N2"}
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
                mapping={0: "Wake", 1: "Sleep"}
            ),
        )
        assert cfg.get_class_names() == ["Wake", "Sleep"]

    def test_from_dict_parses_new_classification_mapping(self):
        data = {
            "name": "mapped_task",
            "head": {"_target_": "foundry.tasks.heads.ReadoutHead"},
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "stages.start",
                "value_key": "stages.id",
            },
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            "classification_mapping": {
                "mapping": {0: "Wake", 1: "Light", 2: "Light", 3: "Deep"},
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
            "target_extractor": {"timestamp_key": "t", "value_key": "v"},
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        }
        cfg = TaskConfig.from_dict(data)
        assert cfg.classification_mapping is None
        assert cfg.output_dim == 5


class TestTargetExtractorWithMapping:
    """TargetExtractor applies new classification_mapping when provided."""

    def test_applies_mapping_to_extracted_values(self):
        from dataclasses import dataclass

        from torch_brain.data import Data

        from foundry.tasks.targets import TargetExtractor

        @dataclass
        class _Stages:
            start: np.ndarray
            id: np.ndarray

        data = Data(
            stages=_Stages(
                start=np.array([0.0, 1.0, 2.0, 3.0]),
                id=np.array([0, 1, 4, 5], dtype=np.int64),
            )
        )

        m = ClassificationMapping(
            mapping={0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}
        )
        extractor = TargetExtractor(
            timestamp_key="stages.start",
            value_key="stages.id",
            classification_mapping=m,
        )
        result = extractor(data)
        expected = np.array([0, 1, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(result["values"], expected)


class TestExtractorProperty:
    """TaskConfig.extractor caches a fully-wired TargetExtractor."""

    def test_extractor_returns_target_extractor_instance(self):
        from foundry.tasks.targets import TargetExtractor

        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        assert isinstance(cfg.extractor, TargetExtractor)

    def test_extractor_injects_classification_mapping(self):
        m = ClassificationMapping(mapping={0: "Wake", 1: "Sleep"})
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=m,
        )
        assert cfg.extractor.classification_mapping is m


class TestValidateTaskMappings:
    """Startup validation logs informational message for unlisted labels."""

    def test_passes_when_all_labels_in_mapping(self):
        from foundry.tasks.validation import validate_task_mappings

        m = ClassificationMapping(mapping={0: "A", 1: "B", 2: "C"})
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "stages.start",
                "value_key": "stages.id",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=m,
        )

        class _MockRec:
            def get_nested_attribute(self, key):
                return np.array([0, 1, 2, 0, 1])

        class _MockDataset:
            recording_ids = ["rec1"]

            def get_recording(self, rid):
                return _MockRec()

        validate_task_mappings({"test": cfg}, _MockDataset())

    def test_logs_info_for_unlisted_labels(self, caplog):
        """Unlisted labels produce informational log, not an error."""
        import logging

        from foundry.tasks.validation import validate_task_mappings

        m = ClassificationMapping(mapping={0: "A", 1: "B"})
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "stages.start",
                "value_key": "stages.id",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=m,
        )

        class _MockRec:
            def get_nested_attribute(self, key):
                return np.array([0, 1, 5])

        class _MockDataset:
            recording_ids = ["rec1"]

            def get_recording(self, rid):
                return _MockRec()

        with caplog.at_level(logging.INFO):
            validate_task_mappings({"test": cfg}, _MockDataset())

        assert any(
            "filtered" in r.message.lower() or "not in" in r.message.lower()
            for r in caplog.records
        )

    def test_raises_when_no_labels_match(self):
        """If zero data labels match the mapping, raise (likely misconfiguration)."""
        from foundry.tasks.validation import validate_task_mappings

        m = ClassificationMapping(mapping={10: "A", 11: "B"})
        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "stages.start",
                "value_key": "stages.id",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=m,
        )

        class _MockRec:
            def get_nested_attribute(self, key):
                return np.array([0, 1, 2, 3])

        class _MockDataset:
            recording_ids = ["rec1"]

            def get_recording(self, rid):
                return _MockRec()

        with pytest.raises(ValueError, match="no labels.*match|none.*match"):
            validate_task_mappings({"test": cfg}, _MockDataset())

    def test_skips_tasks_without_mapping(self):
        from foundry.tasks.validation import validate_task_mappings

        cfg = TaskConfig(
            name="test",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.MSETaskLoss"},
        )

        class _MockDataset:
            recording_ids = ["rec1"]

        validate_task_mappings({"test": cfg}, _MockDataset())


class TestIgnoreIndex:
    """CrossEntropyTaskLoss handles ignore_index correctly."""

    def test_ignore_index_excludes_negative_targets(self):
        import torch

        from foundry.tasks.losses import CrossEntropyTaskLoss

        loss_fn = CrossEntropyTaskLoss(ignore_index=-1)
        predictions = torch.randn(5, 3)
        targets = torch.tensor([0, 1, -1, 2, -1])

        loss = loss_fn(predictions, targets)
        assert loss.isfinite()
        assert loss > 0

    def test_ignore_index_all_ignored_returns_zero(self):
        import torch

        from foundry.tasks.losses import CrossEntropyTaskLoss

        loss_fn = CrossEntropyTaskLoss(ignore_index=-1)
        predictions = torch.randn(3, 3)
        targets = torch.tensor([-1, -1, -1])

        loss = loss_fn(predictions, targets)
        assert loss.item() == 0.0


class TestConfusionMatrixBoundsCheck:
    """ConfusionMatrixTracker drops out-of-bounds targets."""

    def test_drops_negative_targets(self):
        import torch

        from foundry.training.confusion_matrix import ConfusionMatrixTracker

        tracker = ConfusionMatrixTracker(num_classes=3)
        preds = torch.tensor([0, 1, 2, 0])
        targets = torch.tensor([0, -1, 2, -1])

        tracker.update(preds, targets)
        counts, _ = tracker.compute()
        assert counts.sum() == 2

    def test_drops_targets_above_num_classes(self):
        import torch

        from foundry.training.confusion_matrix import ConfusionMatrixTracker

        tracker = ConfusionMatrixTracker(num_classes=3)
        preds = torch.tensor([0, 1, 2, 0])
        targets = torch.tensor([0, 1, 5, 3])

        tracker.update(preds, targets)
        counts, _ = tracker.compute()
        assert counts.sum() == 2


class TestClassWeightsWithMapping:
    """Class weight computation uses new classification_mapping."""

    def test_weights_computed_on_mapped_labels(self):
        from foundry.tasks.class_weights import compute_class_weights_for_tasks

        m = ClassificationMapping(mapping={0: "A", 1: "A", 2: "B"})
        cfg = TaskConfig(
            name="test_clf",
            head={"_target_": "foundry.tasks.heads.ReadoutHead"},
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "trials.timestamps",
                "value_key": "trials.values",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            classification_mapping=m,
        )

        class _MockDataset:
            def get_sampling_intervals(self, split):
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
        # mapped-A has 2 units (raw 0 + raw 1), mapped-B has 1 unit
        assert weights["test_clf"][0] < weights["test_clf"][1]
