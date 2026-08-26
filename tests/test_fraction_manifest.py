from __future__ import annotations

import numpy as np
import pytest

from foundry.data.fraction_manifest import FractionManifestBuilder


class FakeIntervals:
    def __init__(self, labels: list[str]) -> None:
        self.behavior_labels = np.asarray(labels)
        self.start = np.arange(len(labels), dtype=float)
        self.end = self.start + 0.5

    def __len__(self) -> int:
        return len(self.start)


class FakeMapping:
    class_names = ["a", "b"]

    @staticmethod
    def filter_and_remap(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.isin(values, ["raw-a", "raw-b"])
        mapped = np.where(values[keep] == "raw-a", 0, 1)
        return mapped, keep


class FakeMappingWithAbsentClass(FakeMapping):
    class_names = ["a", "b", "c"]


def make_builder(
    recording_id: str = "sub-01_ses-01",
    seed: int = 42,
    min_class_support: int = 1,
) -> FractionManifestBuilder:
    intervals = FakeIntervals(["raw-a"] * 60 + ["ignored"] * 5 + ["raw-b"] * 40)
    return FractionManifestBuilder(
        recording_id=recording_id,
        train_intervals=intervals,
        class_mapping=FakeMapping(),
        seed=seed,
        min_class_support=min_class_support,
    )


def test_manifests_are_nested_deterministic_and_auditable() -> None:
    first = make_builder().build_all_fractions()
    second = make_builder().build_all_fractions()

    assert [manifest.to_dict() for manifest in first] == [
        manifest.to_dict() for manifest in second
    ]
    assert all(
        set(smaller.selected_interval_ids).issubset(
            larger.selected_interval_ids
        )
        for smaller, larger in zip(first, first[1:])
    )
    assert first[-1].total_intervals == 100
    assert len(first[-1].selected_indices) == 100
    assert len(set(first[-1].selected_interval_ids)) == 100
    assert first[-1].per_class_total_counts == {"a": 60, "b": 40}
    assert first[-1].realized_fraction == 1.0


def test_recording_id_changes_selection_and_interval_identity() -> None:
    first = make_builder("sub-01_ses-01").build_fraction(0.25)
    second = make_builder("sub-02_ses-01").build_fraction(0.25)

    assert first.selected_indices != second.selected_indices
    assert first.selected_interval_ids != second.selected_interval_ids
    assert first.manifest_hash != second.manifest_hash


def test_fraction_support_failure_is_explicit() -> None:
    manifest = make_builder(min_class_support=3).build_fraction(0.05)

    assert not manifest.available
    assert manifest.failure_reason == "b: 2 < 3"
    assert manifest.per_class_counts == {"a": 3, "b": 2}


def test_absent_class_is_allowed_when_present_class_threshold_is_met() -> None:
    intervals = FakeIntervals(["raw-a"] * 60 + ["raw-b"] * 60)
    manifest = FractionManifestBuilder(
        recording_id="sub-01_ses-01",
        train_intervals=intervals,
        class_mapping=FakeMappingWithAbsentClass(),
        seed=42,
        min_class_support=3,
        min_present_classes=2,
    ).build_fraction(0.05)

    assert manifest.available
    assert manifest.present_classes == ["a", "b"]
    assert manifest.absent_classes == ["c"]
    assert manifest.per_class_counts == {"a": 3, "b": 3, "c": 0}


def test_too_few_present_classes_fails_loudly() -> None:
    intervals = FakeIntervals(["raw-a"] * 60 + ["raw-b"] * 60)
    manifest = FractionManifestBuilder(
        recording_id="sub-01_ses-01",
        train_intervals=intervals,
        class_mapping=FakeMappingWithAbsentClass(),
        seed=42,
        min_class_support=3,
        min_present_classes=3,
    ).build_fraction(0.05)

    assert not manifest.available
    assert manifest.failure_reason == "present classes: 2 < 3"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recording_id": ""}, "recording_id"),
        ({"min_class_support": 0}, "min_class_support"),
        ({"min_present_classes": 3}, "min_present_classes"),
        ({"fractions": []}, "fractions"),
        ({"fractions": [0.5, 0.25]}, "strictly increasing"),
        ({"fractions": [0.0, 1.0]}, "interval"),
    ],
)
def test_invalid_configuration_fails_loudly(
    kwargs: dict[str, object], message: str
) -> None:
    base = {
        "recording_id": "sub-01_ses-01",
        "train_intervals": FakeIntervals(["raw-a", "raw-b"]),
        "class_mapping": FakeMapping(),
        "seed": 42,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        FractionManifestBuilder(**base)
