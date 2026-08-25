"""Unit tests for the dependency-free NeuralSet adapter boundary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foundry.data.neuralbench.adapter import NeuralSetAdapter
from foundry.models.baselines import EEGNetEncoder
from foundry.tasks.config import TaskConfig


LABEL_MAP = {0: "NonTarget", 1: "Target"}


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
        label_map=LABEL_MAP,
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
        label_map=LABEL_MAP,
        transform=tokenizer,
    )

    assert adapter[0].tokenized


def test_preserves_normalized_head_positions_and_marks_missing_channels():
    sample = _sample()
    sample["channel_positions"] = np.array(
        [[[0.2, 0.4, 0.8], [-0.1, -0.1, -0.1]]], dtype=np.float32
    )
    adapter = NeuralSetAdapter(
        _Dataset([sample]),
        channel_names=["Cz", "unknown"],
        sampling_rate=2.0,
        split="train",
        label_map=LABEL_MAP,
    )

    data = adapter[0]

    np.testing.assert_array_equal(
        data.channels.position, sample["channel_positions"][0]
    )
    np.testing.assert_array_equal(
        data.channels.position_valid, np.array([True, False])
    )
    assert set(data.channels.position_frame) == {"head"}
    assert set(data.channels.position_units) == {"normalized"}


@pytest.mark.parametrize("channel_names", [[], ["Cz", "Cz"]])
def test_rejects_empty_or_duplicate_channel_names(channel_names):
    with pytest.raises(ValueError, match="channel_names|Duplicate channel"):
        NeuralSetAdapter(
            _Dataset([_sample()]),
            channel_names=channel_names,
            sampling_rate=2.0,
            split="train",
            label_map=LABEL_MAP,
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


def test_keeps_subject_and_recording_identities_distinct():
    adapter = NeuralSetAdapter(
        _Dataset([_sample()]),
        channel_names=["Cz", "Pz"],
        sampling_rate=2.0,
        split="train",
        label_map=LABEL_MAP,
        identity_fn=lambda *_: ("study/sub-7", "study/sub-7/run-04"),
    )

    data = adapter[0]

    assert data.subject.id == "nb/p3/study/sub-7"
    assert data.session.id == "nb/p3/study/sub-7/run-04"
    assert list(data.channels.id) == [
        "nb/p3/study/sub-7/run-04/Cz",
        "nb/p3/study/sub-7/run-04/Pz",
    ]


def test_four_second_mi_signal_is_identical_at_eegnet_input():
    """The adapter may transpose layout, but must not alter any MI value."""
    channels, samples = 64, 480  # NeuralBench MI: 4 seconds at 120 Hz
    raw_signal = np.arange(channels * samples, dtype=np.float32).reshape(
        1, channels, samples
    )
    adapter = NeuralSetAdapter(
        _Dataset(
            [
                {
                    "neuro": raw_signal,
                    "target": np.array([[1, 0, 0, 0]]),
                    "subject_id": np.array([[7]]),
                }
            ]
        ),
        channel_names=[f"EEG{i}" for i in range(channels)],
        sampling_rate=120.0,
        split="train",
        label_map={
            0: "imagery_bilateral_feet",
            1: "imagery_bilateral_fist",
            2: "imagery_left_fist",
            3: "imagery_right_fist",
        },
        interval_name="motor_imagery_trials",
    )
    data = adapter[0]
    np.testing.assert_array_equal(data.eeg.signal, raw_signal[0].T)
    assert data.domain.end[0] == 4.0

    task = TaskConfig.from_yaml("configs/tasks/neuralbench/motor_imagery.yaml")
    model = EEGNetEncoder(
        task_configs={task.name: task},
        num_channels=channels,
        num_samples=samples,
    )
    tokenized = model.tokenize(data)
    np.testing.assert_array_equal(
        tokenized["input_values"].obj.numpy(), raw_signal[0].T
    )
    normalized = model._check_input_shape_conv2d(
        tokenized["input_values"].obj.unsqueeze(0)
    )
    torch.testing.assert_close(
        normalized, torch.from_numpy(raw_signal).unsqueeze(0), rtol=0, atol=0
    )
