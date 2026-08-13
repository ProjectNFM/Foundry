from __future__ import annotations

import numpy as np
import pytest
from torch_brain.data import Interval

from foundry.data.datamodules.neurosoft import (
    NeurosoftMultispeciesDataModule,
    filter_source_class_intervals,
    sample_uniform_source_class_intervals,
)
from foundry.tasks.config import TaskConfig


@pytest.fixture
def frequency_task():
    return TaskConfig.from_yaml(
        "configs/tasks/neurosoft_acoustic_stim_8band.yaml"
    )


def _interval(labels: list[str]) -> Interval:
    starts = np.arange(len(labels), dtype=float)
    return Interval(
        starts,
        starts + 0.5,
        behavior_labels=np.asarray(labels),
    )


def test_filter_source_class_intervals_only_changes_selected_source(
    frequency_task,
):
    mapping = frequency_task.class_mapping
    intervals = {
        "minipigs/pig": _interval(["stim_100Hz", "stim_12000Hz"]),
        "monkeys/monkey": _interval(["stim_100Hz", "stim_12000Hz"]),
    }

    filtered = filter_source_class_intervals(
        intervals,
        mapping,
        "behavior_labels",
        {"minipigs": ["high_treble"]},
    )

    assert filtered["minipigs/pig"].behavior_labels.tolist() == ["stim_12000Hz"]
    assert filtered["monkeys/monkey"].behavior_labels.tolist() == [
        "stim_100Hz",
        "stim_12000Hz",
    ]


def test_uniform_source_sampling_is_exact_balanced_and_deterministic(
    frequency_task,
):
    mapping = frequency_task.class_mapping
    labels_by_class = {}
    for raw_label, class_name in mapping.mapping.items():
        labels_by_class.setdefault(class_name, raw_label)
    labels = [
        labels_by_class[class_name]
        for class_name in mapping.class_names
        for _ in range(4)
    ]
    intervals = {
        "minipigs/pig": _interval(labels),
        "monkeys/monkey": _interval(["stim_100Hz", "stim_12000Hz"]),
    }

    sampled = sample_uniform_source_class_intervals(
        intervals,
        mapping,
        "behavior_labels",
        {"minipigs": 17},
        seed=7,
    )
    repeated = sample_uniform_source_class_intervals(
        intervals,
        mapping,
        "behavior_labels",
        {"minipigs": 17},
        seed=7,
    )

    sampled_labels = sampled["minipigs/pig"].behavior_labels
    sampled_ids = mapping.map_to_class_ids(sampled_labels)
    counts = np.bincount(sampled_ids, minlength=mapping.num_classes)
    assert len(sampled_labels) == 17
    assert counts.max() - counts.min() <= 1
    assert (
        sampled["minipigs/pig"].start.tolist()
        == repeated["minipigs/pig"].start.tolist()
    )
    assert sampled["monkeys/monkey"].behavior_labels.tolist() == [
        "stim_100Hz",
        "stim_12000Hz",
    ]


def test_sampling_controls_only_apply_to_training_split(frequency_task):
    module = NeurosoftMultispeciesDataModule(
        dataset_class=object,
        root="unused",
        train_band_ids_by_source={"minipigs": ["high_treble"]},
    )
    module._task_configs = {frequency_task.name: frequency_task}
    intervals = {
        "minipigs/pig": _interval(["stim_100Hz", "stim_12000Hz"]),
        "monkeys/monkey": _interval(["stim_100Hz", "stim_12000Hz"]),
    }

    train = module._filter_intervals(intervals, split="train")
    valid = module._filter_intervals(intervals, split="valid")

    assert train["minipigs/pig"].behavior_labels.tolist() == ["stim_12000Hz"]
    assert valid["minipigs/pig"].behavior_labels.tolist() == [
        "stim_100Hz",
        "stim_12000Hz",
    ]


def test_sampling_controls_reject_unknown_band_name(frequency_task):
    with pytest.raises(ValueError, match="Unknown class 'ultrasound'"):
        filter_source_class_intervals(
            {"minipigs/pig": _interval(["stim_100Hz"])},
            frequency_task.class_mapping,
            "behavior_labels",
            {"minipigs": ["ultrasound"]},
        )


def test_sampling_controls_for_same_source_are_mutually_exclusive():
    with pytest.raises(ValueError, match="both band filtering and uniform"):
        NeurosoftMultispeciesDataModule(
            dataset_class=object,
            root="unused",
            train_band_ids_by_source={"minipigs": ["high_treble"]},
            train_uniform_band_total_count_by_source={"minipigs": 100},
        )
