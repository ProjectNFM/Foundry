"""Tests for task config adaptation and class filtering."""

import copy
import numpy as np
import pytest
from temporaldata import Interval

from foundry.tasks.adaptation import (
    TaskClassSchema,
    _build_label_mapping,
    _adapt_task_config,
    adapt_task_configs,
    filter_sampling_intervals,
)
from foundry.tasks.config import TaskConfig
from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.datasets.neurosoft import (
    NeurosoftMinipigs2026,
    FREQ_GROUPINGS,
    FREQ_GROUP_ORDER,
    format_acoustic_stim_display_names,
)
from foundry.data.datasets.peterson_brunton_pose_trajectory_2022 import (
    PetersonBruntonPoseTrajectory2022,
    ACTIVE_BEHAVIOR_TO_ID,
    ACTIVE_VS_INACTIVE_TO_ID,
)


# ============================================================================
# Unit tests for _build_label_mapping
# ============================================================================


class TestBuildLabelMapping:
    """Tests for _build_label_mapping function."""

    def test_none_classes_returns_empty(self):
        """No adaptation when classes is None."""
        label_map, names = _build_label_mapping(None, {"a": 0, "b": 1})
        assert label_map == {}
        assert names == []

    def test_subset_mode_dense_remap(self):
        """Raw subset mode: dense remap sorted by raw ID."""
        vocab = {"stim_500Hz": 4, "stim_1000Hz": 7, "stim_5000Hz": 14}
        label_map, names = _build_label_mapping(
            ["stim_500Hz", "stim_1000Hz", "stim_5000Hz"], vocab
        )
        # Sorted by raw ID: 4, 7, 14 → remapped to 0, 1, 2
        assert label_map == {4: 0, 7: 1, 14: 2}
        assert names == ["stim_500Hz", "stim_1000Hz", "stim_5000Hz"]

    def test_subset_mode_with_custom_order(self):
        """Subset mode respects raw ID ordering."""
        vocab = {"a": 10, "b": 5, "c": 15}
        label_map, names = _build_label_mapping(["a", "b", "c"], vocab)
        # Should be sorted by raw ID: 5, 10, 15 → "b", "a", "c"
        assert label_map == {5: 0, 10: 1, 15: 2}
        assert names == ["b", "a", "c"]

    def test_grouping_mode_collapses_classes(self):
        """Grouping mode: collapse selected classes into groups."""
        vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
        grouping = {"a": "group1", "b": "group1", "c": "group2"}
        label_map, names = _build_label_mapping(
            ["a", "b", "c"],
            vocab,
            grouping,
            group_order={"group1": 0, "group2": 1},
        )
        # raw 1 -> group1 -> idx 0, raw 2 -> group1 -> idx 0, raw 3 -> group2 -> idx 1
        assert label_map == {1: 0, 2: 0, 3: 1}
        assert names == ["group1", "group2"]

    def test_grouping_uses_group_order(self):
        """Grouping respects group_order when provided."""
        vocab = {"low_freq": 1, "med_freq": 5, "high_freq": 10}
        grouping = {
            "low_freq": "high",
            "med_freq": "low",
            "high_freq": "high",
        }
        group_order = {"high": 0, "low": 1}
        label_map, names = _build_label_mapping(
            ["low_freq", "med_freq", "high_freq"],
            vocab,
            grouping,
            group_order=group_order,
        )
        assert names == ["high", "low"]
        # Raw 1 (low_freq) -> high -> 0, Raw 5 (med_freq) -> low -> 1, Raw 10 (high_freq) -> high -> 0
        assert label_map == {1: 0, 5: 1, 10: 0}

    def test_grouping_alphabetical_order_when_no_group_order(self):
        """Without group_order, groups sorted alphabetically."""
        vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
        grouping = {"a": "zebra", "b": "apple", "c": "zebra"}
        label_map, names = _build_label_mapping(
            ["a", "b", "c"], vocab, grouping
        )
        # Sorted alphabetically: apple, zebra
        assert names == ["apple", "zebra"]
        assert label_map == {
            1: 1,
            2: 0,
            3: 1,
        }  # a->zebra->1, b->apple->0, c->zebra->1

    def test_preset_resolution(self):
        """String preset resolves via grouping_presets."""
        vocab = {"a": 1, "b": 2, "c": 3}
        presets = {"my_preset": {"a": "group1", "b": "group1", "c": "group2"}}
        label_map, names = _build_label_mapping(
            ["a", "b", "c"],
            vocab,
            class_grouping="my_preset",
            grouping_presets=presets,
        )
        assert "group1" in names and "group2" in names

    def test_invalid_class_name_raises(self):
        """Invalid class name in classes raises clear error."""
        vocab = {"valid_a": 0, "valid_b": 1}
        with pytest.raises(ValueError) as exc_info:
            _build_label_mapping(["valid_a", "invalid_c"], vocab)
        assert "Invalid class names" in str(exc_info.value)
        assert "invalid_c" in str(exc_info.value)
        assert "Valid options" in str(exc_info.value)

    def test_invalid_preset_raises(self):
        """Unknown preset name raises clear error."""
        vocab = {"a": 0}
        with pytest.raises(ValueError) as exc_info:
            _build_label_mapping(
                ["a"],
                vocab,
                class_grouping="nonexistent_preset",
                grouping_presets={"other": {}},
            )
        assert "Preset" in str(exc_info.value)
        assert "nonexistent_preset" in str(exc_info.value)

    def test_empty_mapping_after_grouping_raises(self):
        """Grouping that produces no mappings raises error."""
        vocab = {"a": 1, "b": 2}
        grouping = {"c": "group1"}  # c not in vocab
        with pytest.raises(ValueError) as exc_info:
            _build_label_mapping(["a", "b"], vocab, grouping)
        assert "No valid class mappings" in str(exc_info.value)


# ============================================================================
# Unit tests for _adapt_task_config
# ============================================================================


class TestAdaptTaskConfig:
    """Tests for _adapt_task_config function."""

    @pytest.fixture
    def base_task_config(self):
        """Create a simple base task config."""
        return TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 5,
            },
            target_extractor={
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "ts",
                "value_key": "values",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 5,
            },
            class_names=["a", "b", "c", "d", "e"],
        )

    def test_updates_output_dim(self, base_task_config):
        """_adapt_task_config updates head output_dim."""
        adapted = _adapt_task_config(
            base_task_config, label_map={}, effective_names=["x", "y", "z"]
        )
        assert adapted.head["output_dim"] == 3

    def test_updates_metrics_num_classes(self, base_task_config):
        """_adapt_task_config updates metrics.num_classes."""
        adapted = _adapt_task_config(
            base_task_config, label_map={}, effective_names=["x", "y"]
        )
        assert adapted.metrics["num_classes"] == 2

    def test_updates_label_map(self, base_task_config):
        """_adapt_task_config sets target_extractor.label_map."""
        label_map = {1: 0, 2: 1, 3: 2}
        adapted = _adapt_task_config(
            base_task_config,
            label_map=label_map,
            effective_names=["x", "y", "z"],
        )
        assert adapted.target_extractor["label_map"] == label_map

    def test_updates_class_names(self, base_task_config):
        """_adapt_task_config sets class_names."""
        adapted = _adapt_task_config(
            base_task_config, label_map={}, effective_names=["new_a", "new_b"]
        )
        assert adapted.class_names == ["new_a", "new_b"]

    def test_applies_display_formatter(self, base_task_config):
        """_adapt_task_config applies display_formatter if provided."""
        formatter = lambda names: [n.upper() for n in names]
        adapted = _adapt_task_config(
            base_task_config,
            label_map={},
            effective_names=["foo", "bar"],
            display_formatter=formatter,
        )
        assert adapted.class_names == ["FOO", "BAR"]

    def test_skips_metrics_when_none(self):
        """_adapt_task_config handles configs with no metrics."""
        cfg = TaskConfig(
            name="regression",
            head={"_target_": "test.head", "output_dim": 18},
            target_extractor={"_target_": "test.extractor"},
            loss={"_target_": "test.loss"},
            metrics=None,
        )
        adapted = _adapt_task_config(cfg, label_map={}, effective_names=["x"])
        assert adapted.metrics is None
        assert adapted.head["output_dim"] == 1

    def test_does_not_mutate_base_config(self, base_task_config):
        """_adapt_task_config deep-copies to avoid mutating base."""
        original_dim = base_task_config.head["output_dim"]
        _adapt_task_config(
            base_task_config, label_map={}, effective_names=["x"]
        )
        assert base_task_config.head["output_dim"] == original_dim


# ============================================================================
# Unit tests for adapt_task_configs
# ============================================================================


class TestAdaptTaskConfigs:
    """Tests for adapt_task_configs (plural) function."""

    def test_skips_tasks_without_schema(self):
        """Tasks without schema returned unchanged."""
        task_configs = {
            "task1": TaskConfig(
                name="task1",
                head={"output_dim": 5},
                target_extractor={},
                loss={},
            ),
        }
        schemas = {}  # No schema for task1
        result = adapt_task_configs(task_configs, schemas, ["a"], None)
        assert result["task1"] is task_configs["task1"]

    def test_skips_continuous_tasks(self):
        """Continuous/regression tasks not adapted."""
        task_configs = {
            "regression": TaskConfig(
                name="regression",
                head={"output_dim": 18},
                target_extractor={},
                loss={"_target_": "mse_loss"},  # Not CrossEntropy
            ),
        }
        schema = TaskClassSchema(
            vocabulary={"a": 0},
            interval_filter_field="f",
            interval_filter_mode="ids",
        )
        schemas = {"regression": schema}
        result = adapt_task_configs(task_configs, schemas, ["a"], None)
        assert result["regression"].head["output_dim"] == 18


# ============================================================================
# Unit tests for filter_sampling_intervals
# ============================================================================


class TestFilterSamplingIntervals:
    """Tests for filter_sampling_intervals function."""

    def test_names_mode_filters_by_string(self):
        """Names mode filters by string values in field."""
        # Create interval with behavior_labels field
        behavior_labels = np.array(["stim_500Hz", "stim_1000Hz", "stim_500Hz"])
        start = np.array([0.0, 1.0, 2.0])
        end = np.array([0.5, 1.5, 2.5])
        ivl = Interval(
            start=start,
            end=end,
            behavior_labels=behavior_labels,
        )

        schema = TaskClassSchema(
            vocabulary={"stim_500Hz": 1, "stim_1000Hz": 2},
            interval_filter_field="behavior_labels",
            interval_filter_mode="names",
        )

        filtered = filter_sampling_intervals(ivl, schema, ["stim_500Hz"])
        # Should keep indices 0, 2 (both stim_500Hz)
        assert len(filtered) == 2
        assert np.array_equal(filtered.start, np.array([0.0, 2.0]))

    def test_ids_mode_converts_names_to_ids(self):
        """IDs mode converts class names to raw IDs before filtering."""
        behavior_ids = np.array([1, 2, 1, 3, 2])
        start = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        end = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        ivl = Interval(
            start=start,
            end=end,
            behavior_id=behavior_ids,
        )

        schema = TaskClassSchema(
            vocabulary={"Eat": 1, "Talk": 2, "Other": 3},
            interval_filter_field="behavior_id",
            interval_filter_mode="ids",
        )

        # Filter for Eat (id=1) and Talk (id=2)
        filtered = filter_sampling_intervals(ivl, schema, ["Eat", "Talk"])
        # Indices 0, 1, 2, 4 have IDs 1, 2, 1, 2
        assert len(filtered) == 4

    def test_empty_result_returns_empty_interval(self):
        """Filtering with no matches returns empty interval."""
        behavior_labels = np.array(["a", "a"])
        start = np.array([0.0, 1.0])
        end = np.array([0.5, 1.5])
        ivl = Interval(
            start=start,
            end=end,
            behavior_labels=behavior_labels,
        )

        schema = TaskClassSchema(
            vocabulary={"a": 1, "b": 2},
            interval_filter_field="behavior_labels",
            interval_filter_mode="names",
        )

        filtered = filter_sampling_intervals(ivl, schema, ["b"])
        assert len(filtered) == 0


# ============================================================================
# Integration tests
# ============================================================================


class TestNeuroSoftEffectiveConfigs:
    """Integration tests for NeuroSoft class filtering."""

    def test_acoustic_stim_subset_mode(self):
        """NeuroSoft acoustic_stim subset: 2 freqs -> dim 2."""
        task_config = NeurosoftMinipigs2026.AVAILABLE_TASKS[
            "neurosoft_acoustic_stim"
        ]
        schema = NeurosoftMinipigs2026.TASK_CLASS_SCHEMAS[
            "neurosoft_acoustic_stim"
        ]

        # Select 2 frequencies
        adapted = _adapt_task_config(
            task_config,
            label_map={4: 0, 14: 1},  # Arbitrary mapping for 2 classes
            effective_names=["stim_500Hz", "stim_5000Hz"],
        )

        assert adapted.head["output_dim"] == 2
        assert adapted.metrics["num_classes"] == 2
        assert len(adapted.class_names) == 2

    def test_acoustic_stim_grouping_mode(self):
        """NeuroSoft acoustic_stim 3-band grouping."""
        task_config = NeurosoftMinipigs2026.AVAILABLE_TASKS[
            "neurosoft_acoustic_stim"
        ]

        # Select 6 frequencies for 3-band grouping
        frequencies = [
            "stim_500Hz",
            "stim_800Hz",
            "stim_1000Hz",
            "stim_2000Hz",
            "stim_5000Hz",
            "stim_8000Hz",
        ]
        label_map, effective_names = _build_label_mapping(
            frequencies,
            NeurosoftMinipigs2026.TASK_CLASS_SCHEMAS[
                "neurosoft_acoustic_stim"
            ].vocabulary,
            class_grouping="3band",
            grouping_presets=FREQ_GROUPINGS,
            group_order=FREQ_GROUP_ORDER,
        )

        assert len(effective_names) == 3
        assert set(effective_names) == {"low", "medium", "high"}

    def test_display_name_formatter_strips_stim_prefix(self):
        """format_acoustic_stim_display_names strips stim_ prefix."""
        names = ["stim_500Hz", "stim_1000Hz", "low", "medium"]
        formatted = format_acoustic_stim_display_names(names)
        assert formatted == ["500Hz", "1000Hz", "low", "medium"]


class TestAjileEffectiveConfigs:
    """Integration tests for Ajile class filtering."""

    def test_active_behavior_subset(self):
        """Ajile active_behavior subset: 2 behaviors -> dim 2."""
        task_config = PetersonBruntonPoseTrajectory2022.AVAILABLE_TASKS[
            "ajile_active_behavior"
        ]

        # Select 2 behaviors
        label_map, effective_names = _build_label_mapping(
            ["Eat", "Talk"],
            ACTIVE_BEHAVIOR_TO_ID,
        )

        assert len(effective_names) == 2
        assert "Eat" in effective_names
        assert "Talk" in effective_names

    def test_inactive_active_subset(self):
        """Ajile inactive_active with class filtering."""
        label_map, effective_names = _build_label_mapping(
            ["Active"],
            ACTIVE_VS_INACTIVE_TO_ID,
        )

        assert len(effective_names) == 1
        assert effective_names[0] == "Active"

    def test_ajile_custom_grouping(self):
        """Ajile custom grouping: collapse multiple behaviors to one group."""
        # E.g., group Eat, Talk as "Social"
        custom_grouping = {
            "Eat": "Social",
            "Talk": "Social",
            "TV": "Passive",
            "Computer/Phone": "Passive",
            "Other Activity": "Active",
        }
        label_map, effective_names = _build_label_mapping(
            ["Eat", "Talk", "TV", "Computer/Phone", "Other Activity"],
            ACTIVE_BEHAVIOR_TO_ID,
            class_grouping=custom_grouping,
        )

        # 3 groups: Social, Passive, Active
        assert len(effective_names) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
