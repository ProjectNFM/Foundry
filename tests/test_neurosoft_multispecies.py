from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from torch_brain.datasets import DatasetIndex

from foundry.data.datasets.neurosoft import NeurosoftMinipigsMonkeys2026


@dataclass
class _Session:
    id: str


@dataclass
class _Channels:
    id: np.ndarray
    type: np.ndarray


@dataclass
class _Recording:
    session: _Session
    channels: _Channels


class _FakeDataset:
    source = ""
    channel_counts: dict[str, int] = {}

    def __init__(
        self,
        *,
        recording_ids=None,
        transform=None,
        **kwargs,
    ):
        del kwargs
        self.transform = transform
        self.recording_ids = list(recording_ids or sorted(self.channel_counts))

    def get_recording(self, recording_id, _namespace=""):
        prefix = f"{_namespace}/" if _namespace else ""
        count = self.channel_counts[recording_id]
        return _Recording(
            session=_Session(f"{prefix}{recording_id}"),
            channels=_Channels(
                id=np.asarray(
                    [f"{prefix}ch-{index}" for index in range(count)]
                ),
                type=np.asarray(["ECOG"] * count),
            ),
        )

    def __getitem__(self, index: DatasetIndex):
        sample = self.get_recording(index.recording_id, index._namespace)
        sample.source_id = self.source
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(self, split=None):
        return {
            recording_id: f"{self.source}:{split}:{recording_id}"
            for recording_id in self.recording_ids
        }

    def get_channel_ids(self):
        return sorted(
            {
                f"ch-{index}"
                for count in self.channel_counts.values()
                for index in range(count)
            }
        )


class _FakeMinipigs(_FakeDataset):
    source = "minipigs"
    channel_counts = {"shared": 4, "pig-good": 8}


class _FakeMonkeys(_FakeDataset):
    source = "monkeys"
    channel_counts = {"shared": 3, "monkey-good": 9}


@pytest.fixture
def fake_sources(monkeypatch):
    monkeypatch.setattr(
        NeurosoftMinipigsMonkeys2026,
        "SOURCES",
        {"minipigs": _FakeMinipigs, "monkeys": _FakeMonkeys},
    )


def test_combined_dataset_namespaces_recordings_channels_and_sessions(
    fake_sources,
):
    dataset = NeurosoftMinipigsMonkeys2026(root="unused")

    assert dataset.recording_ids == [
        "minipigs/pig-good",
        "minipigs/shared",
        "monkeys/monkey-good",
        "monkeys/shared",
    ]
    assert "minipigs/ch-0" in dataset.get_channel_ids()
    assert "monkeys/ch-0" in dataset.get_channel_ids()

    sample = dataset[DatasetIndex("monkeys/shared", start=0.0, end=1.0)]
    assert sample.session.id == "monkeys/shared"
    assert sample.channels.id.tolist() == [
        "monkeys/ch-0",
        "monkeys/ch-1",
        "monkeys/ch-2",
    ]
    assert sample.source_id == "monkeys"


def test_combined_dataset_preserves_source_after_dictionary_transform(
    fake_sources,
):
    dataset = NeurosoftMinipigsMonkeys2026(
        root="unused",
        transform=lambda sample: {"session_id": sample.session.id},
    )

    sample = dataset[DatasetIndex("minipigs/shared", start=0.0, end=1.0)]

    assert sample == {
        "session_id": "minipigs/shared",
        "source_id": "minipigs",
    }


def test_combined_dataset_routes_namespaced_sampling_intervals(fake_sources):
    dataset = NeurosoftMinipigsMonkeys2026(root="unused")

    assert dataset.get_sampling_intervals("train") == {
        "minipigs/pig-good": "minipigs:train:pig-good",
        "minipigs/shared": "minipigs:train:shared",
        "monkeys/monkey-good": "monkeys:train:monkey-good",
        "monkeys/shared": "monkeys:train:shared",
    }


def test_min_channels_filters_each_recording_independently(
    fake_sources, caplog
):
    dataset = NeurosoftMinipigsMonkeys2026(root="unused", min_channels=8)

    assert dataset.recording_ids == [
        "minipigs/pig-good",
        "monkeys/monkey-good",
    ]
    assert "Filtered 1 minipigs recording(s) with fewer than 8 channels" in (
        caplog.text
    )


def test_min_channels_raises_if_a_source_has_no_usable_recordings(fake_sources):
    with pytest.raises(
        ValueError,
        match="min_channels=10 filtered out all minipigs recordings",
    ):
        NeurosoftMinipigsMonkeys2026(root="unused", min_channels=10)


def test_sources_selects_one_species_without_building_the_other(fake_sources):
    dataset = NeurosoftMinipigsMonkeys2026(
        root="unused",
        sources=["minipigs"],
        min_channels=8,
    )

    assert dataset.recording_ids == ["minipigs/pig-good"]
    assert set(dataset.datasets) == {"minipigs"}


def test_sources_rejects_unknown_species(fake_sources):
    with pytest.raises(ValueError, match="Unknown Neurosoft source"):
        NeurosoftMinipigsMonkeys2026(root="unused", sources=["humans"])
