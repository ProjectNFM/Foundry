"""Unit tests for the dependency-free NeuralSet adapter boundary."""

from __future__ import annotations

import numpy as np
import pytest

from foundry.data.neuralbench.adapter import NeuralSetAdapter


class _Dataset:
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _sample(*, target: tuple[int, int] = (1, 0), subject: int = 7) -> dict:
    return {
        "neuro": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        "target": np.array([target]),
        "subject_id": np.array([[subject]]),
    }


def test_preserves_signal_label_timing_and_identity():
    adapter = NeuralSetAdapter(
        _Dataset([_sample(target=(0, 1))]),
        channel_names=["Cz", "Pz"],
        sampling_rate=2.0,
        split="train",
    )

    data = adapter[0]

    np.testing.assert_array_equal(
        data.eeg.signal,
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
    )
    assert data.eeg.signal.dtype == np.float32
    np.testing.assert_array_equal(data.domain.start, np.array([0.0]))
    np.testing.assert_array_equal(data.domain.end, np.array([1.5]))
    assert data.p300_trials.targets[0] == "Target"
    assert data.session.id == "nb/p3/sub-7"
    assert data.subject.id == "nb/p3/sub-7"
    assert list(data.channels.id) == ["nb/p3/sub-7/Cz", "nb/p3/sub-7/Pz"]


def test_applies_tokenizer_after_constructing_data():
    def tokenizer(data):
        data.tokenized = True
        return data

    adapter = NeuralSetAdapter(
        _Dataset([_sample()]),
        channel_names=["Cz", "Pz"],
        sampling_rate=2.0,
        split="train",
        transform=tokenizer,
    )

    assert adapter[0].tokenized


@pytest.mark.parametrize("channel_names", [[], ["Cz", "Cz"]])
def test_rejects_empty_or_duplicate_channel_names(channel_names):
    with pytest.raises(ValueError, match="channel_names|Duplicate channel"):
        NeuralSetAdapter(
            _Dataset([_sample()]),
            channel_names=channel_names,
            sampling_rate=2.0,
            split="train",
        )


def test_rejects_unmapped_label():
    adapter = NeuralSetAdapter(
        _Dataset([_sample()]),
        channel_names=["Cz", "Pz"],
        sampling_rate=2.0,
        split="train",
        label_map={1: "Target"},
    )

    with pytest.raises(ValueError, match="not in label_map"):
        adapter[0]
