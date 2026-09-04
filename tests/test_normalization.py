"""Tests for train-split input normalization.

Covers:
- RecordingChannelStats validation
- Interval merging
- Chunked train-stat fitting
- RecordingChannelStandardize transform
- DataModule normalization lifecycle
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import h5py
import numpy as np
import pytest
from torch_brain.data import Data, Interval, RegularTimeSeries
from torch_brain.data.arraydict import ArrayDict

from foundry.data.fraction_manifest import _canonical_hash
from foundry.data.normalization import (
    RecordingChannelStats,
    _sampling_rate_from_signal_source,
    fit_recording_global_stats,
    fit_recording_stats,
    load_normalization_stats,
    merge_time_intervals,
    save_normalization_stats,
)
from foundry.data.transforms import RecordingChannelStandardize
from foundry.data.utils import resolve_neural_signal


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_recording(
    session_id: str,
    n_channels: int,
    n_samples: int,
    sampling_rate: float = 2000.0,
    signal: np.ndarray | None = None,
    modality: str = "ecog",
    channel_types: list[str] | None = None,
) -> Data:
    """Build a synthetic recording matching NeuroSoft conventions."""
    if signal is None:
        signal = np.random.randn(n_samples, n_channels).astype(np.float32)
    ts = RegularTimeSeries(
        signal=signal,
        sampling_rate=sampling_rate,
        domain_start=0.0,
    )
    data = Data(domain=Interval(0.0, n_samples / sampling_rate))
    setattr(data, modality, ts)

    if channel_types is None:
        channel_types = [modality.upper()] * n_channels
    data.channels = ArrayDict(
        id=np.array([f"ch{i}" for i in range(n_channels)]),
        type=np.array(channel_types),
    )

    class _Session:
        id = session_id

    data.session = _Session()
    data._absolute_start = 0.0
    return data


class _MockDataset:
    """Minimal dataset supporting get_recording and get_sampling_intervals."""

    def __init__(self, recordings: dict[str, Data], splits: dict):
        self._recordings = recordings
        self._splits = splits

    @property
    def recording_ids(self):
        return list(self._recordings.keys())

    def get_recording(self, rid: str) -> Data:
        return self._recordings[rid]

    def get_sampling_intervals(
        self, split: Optional[Literal["train", "valid", "test"]] = None
    ):
        if split is None:
            return {
                rid: self._recordings[rid].domain for rid in self._recordings
            }
        return self._splits.get(split, {})


def _make_intervals(starts, ends) -> Interval:
    """Create an Interval from start/end arrays."""
    return Interval(
        np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    )


class TestSamplingRateResolution:
    def test_derives_rate_from_timestamp_only_source(self):
        source = SimpleNamespace(
            timestamps=np.arange(20, dtype=np.float64) / 2000
        )

        assert _sampling_rate_from_signal_source(source) == 2000.0


# ─── RecordingChannelStats ────────────────────────────────────────────────────


class TestRecordingChannelStats:
    def test_valid_construction(self):
        stats = RecordingChannelStats(
            recording_id="rec1",
            signal_field="ecog",
            channel_names=("ch0", "ch1"),
            mean=np.array([0.0, 1.0], dtype=np.float32),
            scale=np.array([1.0, 2.0], dtype=np.float32),
            sample_count=100,
            floored_channels=(),
            sampling_rate=2000.0,
        )
        assert stats.recording_id == "rec1"
        assert stats.mean.dtype == np.float32
        assert stats.scale.dtype == np.float32
        assert len(stats.channel_names) == 2

    def test_auto_casts_to_float32(self):
        stats = RecordingChannelStats(
            recording_id="rec1",
            signal_field="ecog",
            channel_names=("ch0",),
            mean=np.array([0.5], dtype=np.float64),
            scale=np.array([1.5], dtype=np.float64),
            sample_count=10,
            floored_channels=(),
            sampling_rate=2000.0,
        )
        assert stats.mean.dtype == np.float32
        assert stats.scale.dtype == np.float32

    def test_rejects_mismatched_mean_length(self):
        with pytest.raises(ValueError, match="mean length"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0", "ch1"),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([1.0, 1.0], dtype=np.float32),
                sample_count=10,
                floored_channels=(),
                sampling_rate=2000.0,
            )

    def test_rejects_mismatched_scale_length(self):
        with pytest.raises(ValueError, match="scale length"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([1.0, 2.0], dtype=np.float32),
                sample_count=10,
                floored_channels=(),
                sampling_rate=2000.0,
            )

    def test_rejects_non_finite_mean(self):
        with pytest.raises(ValueError, match="Non-finite mean"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([np.nan], dtype=np.float32),
                scale=np.array([1.0], dtype=np.float32),
                sample_count=10,
                floored_channels=(),
                sampling_rate=2000.0,
            )

    def test_rejects_non_finite_scale(self):
        with pytest.raises(ValueError, match="Non-finite scale"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([np.inf], dtype=np.float32),
                sample_count=10,
                floored_channels=(),
                sampling_rate=2000.0,
            )

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="Non-positive scale"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([0.0], dtype=np.float32),
                sample_count=10,
                floored_channels=(),
                sampling_rate=2000.0,
            )

    def test_rejects_zero_sample_count(self):
        with pytest.raises(ValueError, match="Empty train population"):
            RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([1.0], dtype=np.float32),
                sample_count=0,
                floored_channels=(),
                sampling_rate=2000.0,
            )


# ─── merge_time_intervals ────────────────────────────────────────────────────


class TestMergeTimeIntervals:
    def test_empty_intervals(self):
        starts, ends = merge_time_intervals(np.array([]), np.array([]))
        assert len(starts) == 0
        assert len(ends) == 0

    def test_single_interval(self):
        starts, ends = merge_time_intervals(np.array([1.0]), np.array([2.0]))
        np.testing.assert_array_equal(starts, [1.0])
        np.testing.assert_array_equal(ends, [2.0])

    def test_non_overlapping(self):
        starts, ends = merge_time_intervals(
            np.array([1.0, 3.0, 5.0]),
            np.array([2.0, 4.0, 6.0]),
        )
        np.testing.assert_array_equal(starts, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(ends, [2.0, 4.0, 6.0])

    def test_overlapping(self):
        starts, ends = merge_time_intervals(
            np.array([1.0, 1.5, 3.0]),
            np.array([2.0, 2.5, 4.0]),
        )
        np.testing.assert_array_equal(starts, [1.0, 3.0])
        np.testing.assert_array_equal(ends, [2.5, 4.0])

    def test_adjacent_intervals(self):
        starts, ends = merge_time_intervals(
            np.array([1.0, 2.0]),
            np.array([2.0, 3.0]),
        )
        np.testing.assert_array_equal(starts, [1.0])
        np.testing.assert_array_equal(ends, [3.0])

    def test_unsorted_input(self):
        starts, ends = merge_time_intervals(
            np.array([3.0, 1.0]),
            np.array([4.0, 2.0]),
        )
        np.testing.assert_array_equal(starts, [1.0, 3.0])
        np.testing.assert_array_equal(ends, [2.0, 4.0])

    def test_fully_contained(self):
        starts, ends = merge_time_intervals(
            np.array([1.0, 1.5]),
            np.array([4.0, 2.0]),
        )
        np.testing.assert_array_equal(starts, [1.0])
        np.testing.assert_array_equal(ends, [4.0])


# ─── fit_recording_stats ─────────────────────────────────────────────────────


class TestFitRecordingStats:
    def _make_dataset(
        self,
        signal: np.ndarray,
        train_starts: list[float],
        train_ends: list[float],
        session_id: str = "sess1",
        sampling_rate: float = 100.0,
    ) -> tuple[_MockDataset, Interval]:
        n_samples, n_channels = signal.shape
        recording = _make_recording(
            session_id=session_id,
            n_channels=n_channels,
            n_samples=n_samples,
            sampling_rate=sampling_rate,
            signal=signal,
        )
        train_iv = _make_intervals(train_starts, train_ends)
        dataset = _MockDataset(
            recordings={session_id: recording},
            splits={
                "train": {session_id: train_iv},
                "valid": {session_id: _make_intervals([0.0], [0.01])},
                "test": {session_id: _make_intervals([0.0], [0.01])},
            },
        )
        return dataset, train_iv

    def test_correct_mean_and_std(self):
        """Two channels with known mean/std over the train region."""
        rng = np.random.RandomState(42)
        n_samples = 10000
        sr = 100.0
        ch0 = rng.randn(n_samples).astype(np.float32) * 3.0 + 5.0
        ch1 = rng.randn(n_samples).astype(np.float32) * 0.5 - 2.0
        signal = np.stack([ch0, ch1], axis=1)

        duration = n_samples / sr
        dataset, train_iv = self._make_dataset(
            signal, [0.0], [duration], sampling_rate=sr
        )
        stats = fit_recording_stats(dataset, "sess1", train_iv)

        np.testing.assert_allclose(stats.mean[0], 5.0, atol=0.2)
        np.testing.assert_allclose(stats.mean[1], -2.0, atol=0.1)
        np.testing.assert_allclose(stats.scale[0], 3.0, atol=0.2)
        np.testing.assert_allclose(stats.scale[1], 0.5, atol=0.1)

    def test_global_stats_use_one_scalar_across_channels(self):
        """Global z-score preserves relative channel scale after fitting."""
        signal = np.array(
            [[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]], dtype=np.float32
        )
        dataset, train_iv = self._make_dataset(
            signal, [0.0], [0.03], sampling_rate=100.0
        )

        stats = fit_recording_global_stats(dataset, "sess1", train_iv)

        expected_mean = signal.mean()
        expected_scale = signal.std(ddof=0)
        np.testing.assert_allclose(stats.mean, expected_mean)
        np.testing.assert_allclose(stats.scale, expected_scale)
        assert stats.sample_count == signal.size

    def test_global_stats_allow_constant_individual_channel(self):
        """A globally varying recording need not vary in every channel."""
        signal = np.column_stack(
            [np.ones(100, dtype=np.float32), np.arange(100, dtype=np.float32)]
        )
        dataset, train_iv = self._make_dataset(
            signal, [0.0], [1.0], sampling_rate=100.0
        )

        stats = fit_recording_global_stats(dataset, "sess1", train_iv)

        assert stats.scale[0] > 0

    def test_disjoint_intervals(self):
        """Statistics computed over disjoint train intervals."""
        sr = 100.0
        n_samples = 1000
        signal = np.ones((n_samples, 2), dtype=np.float32)
        signal[:500, 0] = 2.0
        signal[500:, 0] = 4.0
        signal[:, 1] = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)

        dataset, train_iv = self._make_dataset(
            signal,
            [0.0, 5.0],
            [5.0, 10.0],
            sampling_rate=sr,
        )
        stats = fit_recording_stats(dataset, "sess1", train_iv)

        np.testing.assert_allclose(stats.mean[0], 3.0, atol=1e-5)
        assert stats.sample_count == 1000

    def test_overlapping_intervals_unique_weighting(self):
        """Overlapping intervals do not double-count samples."""
        sr = 100.0
        n_samples = 100
        signal = np.arange(n_samples, dtype=np.float32).reshape(-1, 1)

        dataset, train_iv = self._make_dataset(
            signal,
            [0.0, 0.0],
            [1.0, 1.0],
            sampling_rate=sr,
        )
        stats = fit_recording_stats(dataset, "sess1", train_iv)
        assert stats.sample_count == 100

    def test_rejects_channel_at_scale_floor(self):
        """Constant channels are invalid rather than silently normalized."""
        signal = np.ones((1000, 2), dtype=np.float32)
        signal[:, 0] = 5.0
        signal[:, 1] = 5.0

        dataset, train_iv = self._make_dataset(
            signal, [0.0], [10.0], sampling_rate=100.0
        )
        with pytest.raises(ValueError, match="at or below scale_floor"):
            fit_recording_stats(dataset, "sess1", train_iv, scale_floor=1e-6)

    def test_rejects_empty_train(self):
        """Error when no train samples exist within intervals."""
        signal = np.ones((100, 1), dtype=np.float32)
        dataset, _ = self._make_dataset(
            signal, [99.0], [100.0], sampling_rate=100.0
        )
        empty_iv = _make_intervals([], [])
        with pytest.raises(RuntimeError, match="no train samples"):
            fit_recording_stats(dataset, "sess1", empty_iv)

    def test_float64_accumulation(self):
        """Verify accumulator uses float64 by default."""
        signal = np.linspace(0.5e-6, 1.5e-6, 100, dtype=np.float32).reshape(
            -1, 1
        )
        dataset, train_iv = self._make_dataset(
            signal, [0.0], [1.0], sampling_rate=100.0
        )
        stats = fit_recording_stats(
            dataset, "sess1", train_iv, accumulator_dtype=np.float64
        )
        assert stats.mean.dtype == np.float32
        np.testing.assert_allclose(stats.mean[0], 1e-6, rtol=1e-4)

    def test_sampling_rate_recorded(self):
        stats_sr = 2000.0
        signal = np.linspace(-1.0, 1.0, 200, dtype=np.float32).reshape(-1, 1)
        dataset, train_iv = self._make_dataset(
            signal, [0.0], [0.1], sampling_rate=stats_sr
        )
        stats = fit_recording_stats(dataset, "sess1", train_iv)
        assert stats.sampling_rate == stats_sr

    def test_uses_timestamps_for_discontiguous_recording_domains(self):
        signal = np.arange(200, dtype=np.float32).reshape(-1, 1)
        dataset, train_iv = self._make_dataset(
            signal, [0.0, 1.0], [0.5, 1.5], sampling_rate=100.0
        )
        recording = dataset.get_recording("sess1")
        recording._domain = Interval(np.array([0.0, 1.0]), np.array([0.5, 1.5]))

        stats = fit_recording_stats(dataset, "sess1", train_iv)

        expected = np.concatenate([signal[:50], signal[100:150]]).mean()
        np.testing.assert_allclose(stats.mean, [expected])

    def test_hdf5_signal_reads_only_selected_slices(self, tmp_path):
        """Lazy HDF5 fitting must never request the complete signal array."""
        signal = np.arange(400, dtype=np.float32).reshape(200, 2)
        h5_path = tmp_path / "recording.h5"
        with h5py.File(h5_path, "w") as handle:
            signal_ds = handle.create_dataset("signal", data=signal)

            class _TrackedH5Dataset:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.reads = []
                    self.shape = dataset.shape
                    self.dtype = dataset.dtype
                    self.ndim = dataset.ndim
                    self.file = dataset.file

                def __getitem__(self, key):
                    self.reads.append(key)
                    if key == slice(None):
                        raise AssertionError("full signal read")
                    return self.dataset[key]

            tracked = _TrackedH5Dataset(signal_ds)

            class _RegularSource:
                sampling_rate = 100.0
                domain = Interval(0.0, 2.0)

                def __init__(self):
                    self.__dict__["signal"] = tracked

            recording = Data(domain=Interval(0.0, 2.0))
            recording.ecog = _RegularSource()
            recording.channels = ArrayDict(
                id=np.array(["ch0", "ch1"]),
                type=np.array(["ECOG", "ECOG"]),
            )
            train_iv = _make_intervals([0.10, 1.00], [0.20, 1.10])
            dataset = _MockDataset(
                {"sess1": recording}, {"train": {"sess1": train_iv}}
            )

            stats = fit_recording_stats(
                dataset, "sess1", train_iv, chunk_samples=4
            )

            selected = np.concatenate([signal[10:20], signal[100:110]])
            np.testing.assert_allclose(stats.mean, selected.mean(axis=0))
            np.testing.assert_allclose(stats.scale, selected.std(axis=0))
            assert tracked.reads
            assert all(read != slice(None) for read in tracked.reads)
            assert all(
                read.start >= 10 and read.stop <= 110 for read in tracked.reads
            )
            assert not any(
                read.start < 10 or 20 < read.stop <= 100
                for read in tracked.reads
            )

    def test_timestamped_hdf5_stream_matches_eager_and_rejects_nonfinite(
        self, tmp_path
    ):
        """Timestamp and signal datasets remain lazy while preserving results."""
        signal = np.arange(300, dtype=np.float32).reshape(150, 2)
        timestamps = np.concatenate(
            [np.arange(75) / 100.0, 2.0 + np.arange(75) / 100.0]
        )
        h5_path = tmp_path / "timestamped.h5"
        with h5py.File(h5_path, "w") as handle:
            signal_ds = handle.create_dataset("signal", data=signal)
            timestamp_ds = handle.create_dataset("timestamps", data=timestamps)

            class _TimestampedSource:
                def __init__(self):
                    self.__dict__["signal"] = signal_ds
                    self.__dict__["timestamps"] = timestamp_ds
                    self.domain = Interval([0.0, 2.0], [0.75, 2.75])

            data = Data(domain=Interval([0.0, 2.0], [0.75, 2.75]))
            data.ecog = _TimestampedSource()
            data.channels = ArrayDict(
                id=np.array(["ch0", "ch1"]),
                type=np.array(["ECOG", "ECOG"]),
            )
            dataset = _MockDataset(
                {"sess1": data},
                {"train": {"sess1": _make_intervals([0.1, 2.1], [0.2, 2.2])}},
            )
            intervals = dataset.get_sampling_intervals("train")["sess1"]
            stats = fit_recording_global_stats(
                dataset, "sess1", intervals, chunk_samples=7
            )

            selected = np.concatenate([signal[10:20], signal[85:95]])
            np.testing.assert_allclose(stats.mean, selected.mean(), rtol=1e-6)
            np.testing.assert_allclose(stats.scale, selected.std(), rtol=1e-6)

            timestamp_ds[90] = np.nan
            with pytest.raises(ValueError, match="timestamps must be finite"):
                fit_recording_stats(
                    dataset, "sess1", intervals, chunk_samples=7
                )

    def test_rejects_malformed_interval_before_signal_read(self):
        signal = np.arange(100, dtype=np.float32).reshape(50, 2)
        dataset, _ = self._make_dataset(
            signal, [0.0], [0.1], sampling_rate=100.0
        )
        with pytest.raises(ValueError, match="end must be greater"):
            fit_recording_stats(dataset, "sess1", _make_intervals([0.2], [0.1]))


# ─── save_normalization_stats ─────────────────────────────────────────────────


class TestSaveNormalizationStats:
    def test_saves_npz_and_json(self, tmp_path):
        stats = {
            "rec1": RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0", "ch1"),
                mean=np.array([0.0, 1.0], dtype=np.float32),
                scale=np.array([1.0, 2.0], dtype=np.float32),
                sample_count=100,
                floored_channels=(),
                sampling_rate=2000.0,
            )
        }
        cfg = {"mode": "recording_train_channel_zscore", "scale_floor": 1e-8}

        npz_path, json_path = save_normalization_stats(
            stats, tmp_path, cfg, provenance={"git_sha": "abc123"}
        )

        assert Path(npz_path).exists()
        assert Path(json_path).exists()

        with open(json_path) as f:
            manifest = json.load(f)
        assert manifest["mode"] == "recording_train_channel_zscore"
        assert "rec1" in manifest["recordings"]
        assert manifest["provenance"]["git_sha"] == "abc123"
        assert "stats_artifact_sha256" in manifest

        npz = np.load(npz_path)
        np.testing.assert_array_equal(npz["rec1/mean"], stats["rec1"].mean)
        np.testing.assert_array_equal(npz["rec1/scale"], stats["rec1"].scale)

        loaded = load_normalization_stats(npz_path, json_path)
        np.testing.assert_array_equal(loaded["rec1"].mean, stats["rec1"].mean)

    def test_rejects_hash_mismatched_artifact(self, tmp_path):
        stats = {
            "rec1": RecordingChannelStats(
                recording_id="rec1",
                signal_field="ecog",
                channel_names=("ch0",),
                mean=np.array([0.0], dtype=np.float32),
                scale=np.array([1.0], dtype=np.float32),
                sample_count=1,
                floored_channels=(),
                sampling_rate=1.0,
            )
        }
        npz_path, manifest_path = save_normalization_stats(
            stats, tmp_path, {"mode": "recording_train_channel_zscore"}
        )
        npz_path.write_bytes(b"not an npz")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_normalization_stats(npz_path, manifest_path)


# ─── resolve_neural_signal ────────────────────────────────────────────────────


class TestResolveNeuralSignal:
    def test_ecog_priority(self):
        data = _make_recording("s1", 3, 100, modality="ecog")
        field_name, source, keep, names = resolve_neural_signal(data)
        assert field_name == "ecog"
        assert keep.sum() == 3

    def test_eeg_priority_over_ecog(self):
        """EEG is checked first by modality priority."""
        data = _make_recording("s1", 2, 100, modality="eeg")
        data.ecog = RegularTimeSeries(
            signal=np.zeros((100, 1), dtype=np.float32),
            sampling_rate=100.0,
            domain_start=0.0,
        )
        field_name, _, _, _ = resolve_neural_signal(data)
        assert field_name == "eeg"

    def test_filters_unsupported_modalities(self):
        data = _make_recording(
            "s1",
            4,
            100,
            modality="ecog",
            channel_types=["ECOG", "ECOG", "EMG", "EOG"],
        )
        _, _, keep, names = resolve_neural_signal(data)
        assert keep.sum() == 2
        assert len(names) == 2

    def test_raises_when_no_signal_field(self):
        data = Data(domain=Interval(0.0, 1.0))
        data.channels = ArrayDict(id=np.array(["ch0"]))
        with pytest.raises(ValueError, match="must contain"):
            resolve_neural_signal(data)


# ─── RecordingChannelStandardize ──────────────────────────────────────────────


class TestRecordingChannelStandardize:
    def _make_stats(self, session_id="s1", n_channels=3, mean=None, scale=None):
        if mean is None:
            mean = np.zeros(n_channels, dtype=np.float32)
        if scale is None:
            scale = np.ones(n_channels, dtype=np.float32)
        return RecordingChannelStats(
            recording_id=session_id,
            signal_field="ecog",
            channel_names=tuple(f"ch{i}" for i in range(n_channels)),
            mean=mean,
            scale=scale,
            sample_count=1000,
            floored_channels=(),
            sampling_rate=2000.0,
        )

    def _make_window(self, session_id="s1", n_channels=3, n_samples=100):
        return _make_recording(
            session_id=session_id,
            n_channels=n_channels,
            n_samples=n_samples,
        )

    def test_basic_normalization(self):
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scale = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        stats = self._make_stats(mean=mean, scale=scale)
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )

        data = self._make_window()
        original_signal = data.ecog.signal.copy()
        result = transform(data)

        expected = (original_signal - mean) / scale
        np.testing.assert_allclose(result.ecog.signal, expected, rtol=1e-5)

    def test_zero_mean_unit_scale_is_identity(self):
        stats = self._make_stats()
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )

        signal = np.random.randn(100, 3).astype(np.float32)
        data = _make_recording("s1", 3, 100, signal=signal.copy())
        result = transform(data)

        np.testing.assert_allclose(result.ecog.signal, signal, rtol=1e-5)

    def test_near_zero_mean_unit_std_after_normalization(self):
        """When normalizing with the true mean/std, output has ~0 mean, ~1 std."""
        rng = np.random.RandomState(42)
        n = 10000
        signal = rng.randn(n, 2).astype(np.float32) * np.array([3.0, 0.1])
        signal += np.array([5.0, -2.0])

        mean = signal.mean(axis=0).astype(np.float32)
        std = signal.std(axis=0).astype(np.float32)
        stats = self._make_stats(n_channels=2, mean=mean, scale=std)
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )

        data = _make_recording("s1", 2, n, signal=signal.copy())
        result = transform(data)

        np.testing.assert_allclose(
            result.ecog.signal.mean(axis=0), 0.0, atol=0.05
        )
        np.testing.assert_allclose(
            result.ecog.signal.std(axis=0), 1.0, atol=0.05
        )

    def test_rejects_unknown_session(self):
        stats = self._make_stats(session_id="known")
        transform = RecordingChannelStandardize(
            stats_by_recording={"known": stats}
        )
        data = self._make_window(session_id="unknown")
        with pytest.raises(KeyError, match="unknown"):
            transform(data)

    def test_rejects_channel_count_mismatch(self):
        stats = self._make_stats(n_channels=3)
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )
        data = self._make_window(n_channels=5)
        with pytest.raises(ValueError, match="supported channels"):
            transform(data)

    def test_rejects_channel_order_mismatch(self):
        stats = RecordingChannelStats(
            recording_id="s1",
            signal_field="ecog",
            channel_names=("chA", "chB"),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            sample_count=10,
            floored_channels=(),
            sampling_rate=2000.0,
        )
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )
        data = self._make_window(n_channels=2)
        with pytest.raises(ValueError, match="Channel order mismatch"):
            transform(data)

    def test_rejects_signal_field_mismatch(self):
        stats = self._make_stats()
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )
        data = _make_recording("s1", 3, 10, modality="eeg")
        with pytest.raises(ValueError, match="Signal field mismatch"):
            transform(data)

    def test_preserves_unsupported_channels(self):
        """Channels not in supported modalities are left untouched."""
        n = 50
        signal = np.ones((n, 4), dtype=np.float32) * 42.0

        stats = RecordingChannelStats(
            recording_id="s1",
            signal_field="ecog",
            channel_names=("ch0", "ch1"),
            mean=np.array([10.0, 10.0], dtype=np.float32),
            scale=np.array([2.0, 2.0], dtype=np.float32),
            sample_count=100,
            floored_channels=(),
            sampling_rate=2000.0,
        )
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )

        data = _make_recording(
            "s1",
            4,
            n,
            signal=signal.copy(),
            channel_types=["ECOG", "ECOG", "EMG", "EOG"],
        )
        result = transform(data)

        np.testing.assert_allclose(result.ecog.signal[:, 2:], 42.0)
        expected_norm = (42.0 - 10.0) / 2.0
        np.testing.assert_allclose(
            result.ecog.signal[:, :2], expected_norm, rtol=1e-5
        )

    def test_does_not_mutate_original_signal_array(self):
        """The original numpy array from the signal source is not modified."""
        stats = self._make_stats(
            n_channels=2,
            mean=np.array([1.0, 2.0], dtype=np.float32),
            scale=np.array([0.5, 0.5], dtype=np.float32),
        )
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )

        original = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        original_copy = original.copy()
        data = _make_recording("s1", 2, 2, signal=original)
        transform(data)

        np.testing.assert_array_equal(original, original_copy)

    def test_preserves_metadata(self):
        stats = self._make_stats()
        transform = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}
        )
        data = self._make_window()
        original_sr = data.ecog.sampling_rate
        original_domain = (data.domain.start, data.domain.end)

        result = transform(data)

        assert result.ecog.sampling_rate == original_sr
        assert result.domain.start == original_domain[0]
        assert result.domain.end == original_domain[1]
        assert str(result.session.id) == "s1"

    def test_repr(self):
        stats = self._make_stats()
        t = RecordingChannelStandardize(
            stats_by_recording={"s1": stats}, scale_floor=1e-8
        )
        r = repr(t)
        assert "RecordingChannelStandardize" in r
        assert "recordings=1" in r

    def test_rejects_empty_stats(self):
        with pytest.raises(ValueError, match="at least one"):
            RecordingChannelStandardize(stats_by_recording={})


# ─── DataModule integration ───────────────────────────────────────────────────


class _SplitMockDataset:
    """Dataset with configurable per-split intervals for DataModule tests."""

    def __init__(
        self,
        recordings: dict[str, Data],
        splits: dict[str, dict[str, Interval]],
    ):
        self._recordings = recordings
        self._splits = splits
        self.transform = None

    @property
    def recording_ids(self):
        return list(self._recordings.keys())

    def get_recording(self, rid: str) -> Data:
        return self._recordings[rid]

    def get_sampling_intervals(self, split=None):
        if split is None:
            return {
                rid: self._recordings[rid].domain for rid in self._recordings
            }
        return self._splits.get(split, {})

    def get_channel_ids(self):
        ids = []
        for data in self._recordings.values():
            ids.extend(list(data.channels.id))
        return ids


def _build_split_mock_dataset(
    root: str = "./data",
    signal_mean: float = 100.0,
    signal_std: float = 0.5,
    n_samples: int = 2000,
    sr: float = 100.0,
    **kwargs,
) -> _SplitMockDataset:
    """Factory that matches the NeuralDataModule dataset_class interface."""
    rng = np.random.RandomState(0)
    signal = (
        rng.randn(n_samples, 3).astype(np.float32) * signal_std + signal_mean
    )
    recording = _make_recording(
        "rec1", 3, n_samples, sampling_rate=sr, signal=signal
    )

    duration = n_samples / sr
    train_end = duration * 0.6
    valid_end = duration * 0.8

    splits = {
        "train": {
            "rec1": _make_intervals([0.0], [train_end]),
        },
        "valid": {
            "rec1": _make_intervals([train_end], [valid_end]),
        },
        "test": {
            "rec1": _make_intervals([valid_end], [duration]),
        },
    }

    return _SplitMockDataset(recordings={"rec1": recording}, splits=splits)


class TestDataModuleNormalization:
    def _make_dm(self, normalization_cfg=None, transforms=None):
        from foundry.data.datamodules import NeuralDataModule

        return NeuralDataModule(
            dataset_class=_build_split_mock_dataset,
            root="./data",
            batch_size=4,
            sequence_length=1.0,
            transforms=transforms,
            input_normalization=normalization_cfg,
        )

    def test_disabled_by_default(self):
        dm = self._make_dm()
        dm.setup("fit")
        assert dm._standardizer is None
        assert dm.normalization_stats is None

    def test_disabled_mode_explicit(self):
        dm = self._make_dm(normalization_cfg={"mode": "disabled"})
        dm.setup("fit")
        assert dm._standardizer is None

    def test_enabled_fits_stats(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
                "scale_floor": 1e-8,
            }
        )
        dm.setup("fit")

        assert dm._standardizer is not None
        assert dm.normalization_stats is not None
        assert "rec1" in dm.normalization_stats

        stats = dm.normalization_stats["rec1"]
        assert stats.sample_count > 0
        assert len(stats.channel_names) == 3

    def test_metadata_only_setup_defers_normalization_until_final_setup(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
                "scale_floor": 1e-8,
            }
        )
        dm.setup("fit", fit_normalization=False)
        assert dm.dataset is not None
        assert dm.normalization_stats is None
        assert dm._normalization_fit_count == 0

        dm.setup("fit")
        dm.setup("fit")
        assert dm.normalization_stats is not None
        assert dm._normalization_fit_count == 1

    def test_global_mode_fits_broadcast_statistics(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_global_zscore",
                "scale_floor": 1e-8,
            }
        )
        dm.setup("fit")

        stats = dm.normalization_stats["rec1"]
        assert np.all(stats.mean == stats.mean[0])
        assert np.all(stats.scale == stats.scale[0])

    def test_standardizer_in_transform_pipeline(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
            }
        )
        dm.setup("fit")

        transform_list = list(dm.transform)
        assert any(
            isinstance(t, RecordingChannelStandardize) for t in transform_list
        )

    def test_standardizer_before_tokenizer(self):
        called = []

        def fake_tokenizer(data):
            called.append("tokenizer")
            return data

        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
            }
        )
        dm.set_tokenizer(fake_tokenizer)
        dm.setup("fit")

        transform_list = list(dm.transform)
        std_idx = next(
            i
            for i, t in enumerate(transform_list)
            if isinstance(t, RecordingChannelStandardize)
        )
        tok_idx = next(
            i for i, t in enumerate(transform_list) if t is fake_tokenizer
        )
        assert std_idx < tok_idx

    def test_set_tokenizer_preserves_standardizer(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
            }
        )
        dm.setup("fit")

        def new_tokenizer(data):
            return data

        dm.set_tokenizer(new_tokenizer)

        assert dm._standardizer is not None
        transform_list = list(dm.transform)
        assert any(
            isinstance(t, RecordingChannelStandardize) for t in transform_list
        )
        assert new_tokenizer in transform_list

    def test_user_transforms_before_standardizer(self):
        class _UserTransform:
            def __call__(self, data):
                return data

        user_t = _UserTransform()
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
            },
            transforms=[user_t],
        )
        dm.setup("fit")

        transform_list = list(dm.transform)
        user_idx = transform_list.index(user_t)
        std_idx = next(
            i
            for i, t in enumerate(transform_list)
            if isinstance(t, RecordingChannelStandardize)
        )
        assert user_idx < std_idx

    def test_rejects_unsupported_mode(self):
        dm = self._make_dm(normalization_cfg={"mode": "batch_norm_something"})
        with pytest.raises(ValueError, match="Unsupported"):
            dm.setup("fit")

    def test_idempotent_setup(self):
        dm = self._make_dm(
            normalization_cfg={
                "mode": "recording_train_channel_zscore",
            }
        )
        dm.setup("fit")
        first_stats = dm.normalization_stats
        dm.setup("fit")
        assert dm.normalization_stats == first_stats

    @staticmethod
    def _cached_cfg(cache_dir: Path, mode="recording_train_channel_zscore"):
        return {
            "mode": mode,
            "scale_floor": 1e-8,
            "accumulator_dtype": "float64",
            "cache": {"enabled": True, "directory": str(cache_dir)},
        }

    def test_cache_miss_then_hit_skips_fit(self, tmp_path):
        cfg = self._cached_cfg(tmp_path / "cache")
        first = self._make_dm(normalization_cfg=cfg)
        first.setup("fit")
        assert first._normalization_cache_status == "miss"
        assert first._normalization_fit_count == 1

        second = self._make_dm(normalization_cfg=cfg)
        second.setup("fit")
        assert second._normalization_cache_status == "hit"
        assert second._normalization_fit_count == 0
        for recording_id in first.normalization_stats:
            np.testing.assert_array_equal(
                first.normalization_stats[recording_id].mean,
                second.normalization_stats[recording_id].mean,
            )

    def test_cache_key_changes_with_normalization_config(self, tmp_path):
        channel_dm = self._make_dm(
            normalization_cfg=self._cached_cfg(tmp_path / "cache")
        )
        channel_dm.setup("fit")
        global_dm = self._make_dm(
            normalization_cfg=self._cached_cfg(
                tmp_path / "cache", mode="recording_train_global_zscore"
            )
        )
        global_dm.setup("fit")
        assert (
            channel_dm._normalization_cache_key
            != global_dm._normalization_cache_key
        )
        assert global_dm._normalization_cache_status == "miss"

    def test_cache_key_changes_with_intervals_and_source_manifest(
        self, tmp_path
    ):
        from foundry.data.datamodules import NeuralDataModule

        cfg = self._cached_cfg(tmp_path / "cache")
        first = self._make_dm(normalization_cfg=cfg)
        first.setup("fit")

        def changed_intervals_dataset(**kwargs):
            dataset = _build_split_mock_dataset(**kwargs)
            dataset._splits["train"]["rec1"] = _make_intervals([0.0], [5.0])
            return dataset

        changed = NeuralDataModule(
            dataset_class=changed_intervals_dataset,
            root="./data",
            batch_size=4,
            sequence_length=1.0,
            input_normalization=cfg,
        )
        changed.setup("fit")
        assert (
            first._normalization_cache_key != changed._normalization_cache_key
        )
        assert changed._normalization_cache_status == "miss"

        intervals = first._effective_sampling_intervals("train")
        identity_args = {
            "mode": "recording_train_channel_zscore",
            "supported_modalities": frozenset({"eeg", "ecog", "seeg", "ieeg"}),
            "scale_floor": 1e-8,
            "accumulator_dtype": "float64",
        }
        first._source_manifest = SimpleNamespace(manifest_hash="manifest-a")
        manifest_a = first._build_normalization_cache_identity(
            intervals, **identity_args
        )
        first._source_manifest = SimpleNamespace(manifest_hash="manifest-b")
        manifest_b = first._build_normalization_cache_identity(
            intervals, **identity_args
        )
        assert _canonical_hash(manifest_a) != _canonical_hash(manifest_b)

        first._source_manifest = None
        first._fraction_manifests = {
            "rec1": SimpleNamespace(manifest_hash="fraction-seed-1")
        }
        fraction_a = first._build_normalization_cache_identity(
            intervals, **identity_args
        )
        first._fraction_manifests = {
            "rec1": SimpleNamespace(manifest_hash="fraction-seed-2")
        }
        fraction_b = first._build_normalization_cache_identity(
            intervals, **identity_args
        )
        assert _canonical_hash(fraction_a) != _canonical_hash(fraction_b)

    def test_changed_in_memory_data_invalidates_cache(self, tmp_path):
        from foundry.data.datamodules import NeuralDataModule

        cfg = self._cached_cfg(tmp_path / "cache")

        def shifted_dataset(signal_mean=100.0, **kwargs):
            return _build_split_mock_dataset(signal_mean=signal_mean, **kwargs)

        first = NeuralDataModule(
            dataset_class=shifted_dataset,
            dataset_kwargs={"signal_mean": 100.0},
            root="./data",
            batch_size=4,
            sequence_length=1.0,
            input_normalization=cfg,
        )
        changed = NeuralDataModule(
            dataset_class=shifted_dataset,
            dataset_kwargs={"signal_mean": 101.0},
            root="./data",
            batch_size=4,
            sequence_length=1.0,
            input_normalization=cfg,
        )
        first.setup("fit")
        changed.setup("fit")

        assert (
            first._normalization_cache_key != changed._normalization_cache_key
        )
        assert changed._normalization_cache_status == "miss"

    @pytest.mark.parametrize("corrupt", ["npz", "manifest", "identity"])
    def test_corrupt_cache_is_rejected_and_recomputed(self, tmp_path, corrupt):
        cfg = self._cached_cfg(tmp_path / "cache")
        first = self._make_dm(normalization_cfg=cfg)
        first.setup("fit")
        entry = tmp_path / "cache" / first._normalization_cache_key
        npz_path = entry / "input_normalization_stats.npz"
        manifest_path = entry / "input_normalization_manifest.json"
        if corrupt == "npz":
            npz_path.write_bytes(b"interrupted")
        elif corrupt == "manifest":
            manifest_path.write_text("{not-json", encoding="utf-8")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["provenance"]["cache_identity"]["mode"] = "wrong"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        second = self._make_dm(normalization_cfg=cfg)
        second.setup("fit")
        assert second._normalization_cache_status == "miss"
        assert second._normalization_fit_count == 1
        load_normalization_stats(npz_path, manifest_path)

    def test_run_artifact_preserves_cache_and_interval_provenance(
        self, tmp_path
    ):
        cfg = self._cached_cfg(tmp_path / "cache")
        first = self._make_dm(normalization_cfg=cfg)
        first.setup("fit")
        hit = self._make_dm(normalization_cfg=cfg)
        hit.setup("fit")

        metadata = hit.write_normalization_artifacts(tmp_path / "run")
        manifest = json.loads(
            Path(metadata["manifest_path"]).read_text(encoding="utf-8")
        )
        provenance = manifest["provenance"]
        assert provenance["cache_status"] == "hit"
        assert provenance["cache_key"] == hit._normalization_cache_key
        assert (
            provenance["train_interval_hash"] == metadata["train_interval_hash"]
        )
        assert (
            provenance["cache_identity"]["train_interval_hash"]
            == metadata["train_interval_hash"]
        )
        assert not list((tmp_path / "run").glob(".*input_normalization*"))


# ─── End-to-end fitting and normalization ─────────────────────────────────────


class TestEndToEndNormalization:
    def test_train_only_fitting(self):
        """Only train-partition samples affect fitted statistics."""
        rng = np.random.RandomState(42)
        sr = 100.0
        n = 1000

        train_signal = rng.randn(600, 2).astype(np.float32) * 2.0 + 10.0
        valid_signal = rng.randn(200, 2).astype(np.float32) * 50.0 + 500.0
        test_signal = rng.randn(200, 2).astype(np.float32) * 100.0 - 1000.0
        signal = np.vstack([train_signal, valid_signal, test_signal])

        recording = _make_recording(
            "rec1", 2, n, sampling_rate=sr, signal=signal
        )
        splits = {
            "train": {"rec1": _make_intervals([0.0], [6.0])},
            "valid": {"rec1": _make_intervals([6.0], [8.0])},
            "test": {"rec1": _make_intervals([8.0], [10.0])},
        }
        dataset = _MockDataset(recordings={"rec1": recording}, splits=splits)

        train_iv = splits["train"]["rec1"]
        stats = fit_recording_stats(dataset, "rec1", train_iv)

        np.testing.assert_allclose(stats.mean, 10.0, atol=0.5)
        np.testing.assert_allclose(stats.scale, 2.0, atol=0.5)

    def test_fit_and_apply(self):
        """Fit stats, apply standardizer, verify near zero-mean unit-var."""
        rng = np.random.RandomState(42)
        sr = 100.0
        n = 2000
        true_mean = np.array([5.0, -3.0], dtype=np.float32)
        true_std = np.array([2.0, 0.5], dtype=np.float32)
        signal = rng.randn(n, 2).astype(np.float32) * true_std + true_mean

        recording = _make_recording(
            "rec1", 2, n, sampling_rate=sr, signal=signal.copy()
        )
        splits = {
            "train": {"rec1": _make_intervals([0.0], [n / sr])},
            "valid": {"rec1": _make_intervals([0.0], [1.0])},
            "test": {"rec1": _make_intervals([0.0], [1.0])},
        }
        dataset = _MockDataset(recordings={"rec1": recording}, splits=splits)

        stats = fit_recording_stats(dataset, "rec1", splits["train"]["rec1"])
        transform = RecordingChannelStandardize(
            stats_by_recording={"rec1": stats}
        )

        window = _make_recording(
            "rec1", 2, n, sampling_rate=sr, signal=signal.copy()
        )
        result = transform(window)

        np.testing.assert_allclose(
            result.ecog.signal.mean(axis=0), 0.0, atol=0.1
        )
        np.testing.assert_allclose(
            result.ecog.signal.std(axis=0), 1.0, atol=0.1
        )
