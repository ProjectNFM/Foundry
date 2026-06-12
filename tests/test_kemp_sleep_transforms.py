"""Tests for SelectEEGChannels and PrepareSleepStages transforms."""

from __future__ import annotations

import numpy as np
import pytest
from torch_brain.data import Data, Interval, RegularTimeSeries
from torch_brain.data.arraydict import ArrayDict

from foundry.data.transforms import SelectEEGChannels, PrepareSleepStages


# ─── Fixtures ────────────────────────────────────────────────────────────────


class _Stages:
    """Minimal stage annotation with the same interface as torch_brain LazyInterval."""

    def __init__(self, ids, starts, ends):
        self.id = np.array(ids, dtype=np.int64)
        self.start = np.array(starts, dtype=np.float64)
        self.end = np.array(ends, dtype=np.float64)


def _make_channel_data(channel_ids, channel_types, n_samples=50):
    n_channels = len(channel_ids)
    signal = np.arange(n_samples * n_channels, dtype=np.float32).reshape(
        n_samples, n_channels
    )
    data = Data(
        eeg=RegularTimeSeries(
            signal=signal, sampling_rate=100.0, domain_start=0.0
        ),
        domain=Interval(0.0, float(n_samples) / 100.0),
    )
    data.channels = ArrayDict(
        id=np.array(channel_ids),
        type=np.array(channel_types),
    )
    return data


def _make_stages_data(ids, starts, ends):
    data = Data(domain=Interval(0.0, 200.0))
    data.stages = _Stages(ids, starts, ends)
    return data


# ─── SelectEEGChannels ───────────────────────────────────────────────────────


class TestSelectEEGChannels:
    def test_cassette_recording_retains_two_eeg_channels(self):
        """Cassette (6 channels): keeps only Fpz-Cz and Pz-Oz."""
        data = _make_channel_data(
            channel_ids=[
                "SC4001E0-PSG/EEG Fpz-Cz",
                "SC4001E0-PSG/EEG Pz-Oz",
                "SC4001E0-PSG/EOG horizontal",
                "SC4001E0-PSG/Resp oro-nasal",
                "SC4001E0-PSG/EMG submental",
                "SC4001E0-PSG/Temp rectal",
            ],
            channel_types=["eeg", "eeg", "eog", "resp", "emg", "temp"],
        )

        result = SelectEEGChannels()(data)

        ch_ids = list(result.channels.id)
        assert len(ch_ids) == 2
        assert any("EEG Fpz-Cz" in cid for cid in ch_ids)
        assert any("EEG Pz-Oz" in cid for cid in ch_ids)

    def test_cassette_recording_filters_eeg_signal_columns(self):
        """Signal is reduced to two columns matching the retained channels."""
        data = _make_channel_data(
            channel_ids=[
                "SC4001E0-PSG/EEG Fpz-Cz",
                "SC4001E0-PSG/EEG Pz-Oz",
                "SC4001E0-PSG/EOG horizontal",
                "SC4001E0-PSG/Resp oro-nasal",
                "SC4001E0-PSG/EMG submental",
                "SC4001E0-PSG/Temp rectal",
            ],
            channel_types=["eeg", "eeg", "eog", "resp", "emg", "temp"],
        )

        result = SelectEEGChannels()(data)

        assert result.eeg.signal.shape[1] == 2

    def test_telemetry_recording_drops_marker_channel(self):
        """Telemetry (3 EEG-typed): EEG-typed Marker excluded by suffix filter."""
        data = _make_channel_data(
            channel_ids=[
                "ST7011J0-PSG/EEG Fpz-Cz",
                "ST7011J0-PSG/EEG Pz-Oz",
                "ST7011J0-PSG/EOG horizontal",
                "ST7011J0-PSG/EMG submental",
                "ST7011J0-PSG/Marker",
            ],
            channel_types=["eeg", "eeg", "eog", "emg", "eeg"],
        )

        result = SelectEEGChannels()(data)

        ch_ids = list(result.channels.id)
        assert len(ch_ids) == 2
        assert not any("Marker" in cid for cid in ch_ids)

    def test_selected_signal_columns_match_fpz_cz_and_pz_oz_positions(self):
        """Signal columns come from the Fpz-Cz and Pz-Oz positions (0 and 1)."""
        data = _make_channel_data(
            channel_ids=[
                "SC/EEG Fpz-Cz",
                "SC/EEG Pz-Oz",
                "SC/EOG horizontal",
            ],
            channel_types=["eeg", "eeg", "eog"],
        )
        original_signal = data.eeg.signal.copy()

        result = SelectEEGChannels()(data)

        np.testing.assert_array_equal(result.eeg.signal, original_signal[:, :2])

    def test_preserves_sampling_rate(self):
        data = _make_channel_data(
            channel_ids=["S/EEG Fpz-Cz", "S/EEG Pz-Oz", "S/EOG horizontal"],
            channel_types=["eeg", "eeg", "eog"],
        )

        result = SelectEEGChannels()(data)

        assert result.eeg.sampling_rate == 100.0

    def test_raises_when_no_matching_channels(self):
        data = _make_channel_data(
            channel_ids=["S/EOG horizontal", "S/EMG submental"],
            channel_types=["eog", "emg"],
        )

        with pytest.raises(ValueError, match="No EEG channels"):
            SelectEEGChannels()(data)


# ─── PrepareSleepStages ──────────────────────────────────────────────────────


class TestPrepareSleepStages:
    def test_materializes_midpoint_timestamps(self):
        """Midpoints = (start + end) / 2 for each stage interval."""
        data = _make_stages_data(
            ids=[0, 1, 2],
            starts=[0.0, 30.0, 60.0],
            ends=[30.0, 60.0, 90.0],
        )

        result = PrepareSleepStages()(data)

        np.testing.assert_allclose(result.stages.timestamps, [15.0, 45.0, 75.0])

    def test_materializes_stage_ids_as_values(self):
        """Stage IDs are preserved as values on the output."""
        data = _make_stages_data(
            ids=[0, 3, 5],
            starts=[0.0, 30.0, 60.0],
            ends=[30.0, 60.0, 90.0],
        )

        result = PrepareSleepStages()(data)

        np.testing.assert_array_equal(result.stages.values, [0, 3, 5])

    def test_excludes_unknown_stage_id_6(self):
        """Stage ID 6 (?) is excluded from both timestamps and values."""
        data = _make_stages_data(
            ids=[0, 6, 2],
            starts=[0.0, 30.0, 60.0],
            ends=[30.0, 60.0, 90.0],
        )

        result = PrepareSleepStages()(data)

        assert 6 not in result.stages.values
        assert len(result.stages.timestamps) == 2
        assert len(result.stages.values) == 2

    def test_excludes_multiple_unknown_stages(self):
        """All unknown-stage epochs are excluded."""
        data = _make_stages_data(
            ids=[0, 6, 1, 6, 2],
            starts=[0.0, 30.0, 60.0, 90.0, 120.0],
            ends=[30.0, 60.0, 90.0, 120.0, 150.0],
        )

        result = PrepareSleepStages()(data)

        assert 6 not in result.stages.values
        assert len(result.stages.timestamps) == 3

    def test_preserves_known_stages_around_unknown(self):
        """Timestamps and values for non-unknown stages are correct after filtering."""
        data = _make_stages_data(
            ids=[3, 6, 4, 5],
            starts=[0.0, 30.0, 60.0, 90.0],
            ends=[30.0, 60.0, 90.0, 120.0],
        )

        result = PrepareSleepStages()(data)

        np.testing.assert_array_equal(result.stages.values, [3, 4, 5])
        np.testing.assert_allclose(
            result.stages.timestamps, [15.0, 75.0, 105.0]
        )

    def test_handles_variable_length_intervals(self):
        """Stage intervals of different durations (coalesced epochs) are handled."""
        data = _make_stages_data(
            ids=[0, 1],
            starts=[0.0, 30630.0],
            ends=[30630.0, 30750.0],
        )

        result = PrepareSleepStages()(data)

        np.testing.assert_allclose(result.stages.timestamps, [15315.0, 30690.0])
        np.testing.assert_array_equal(result.stages.values, [0, 1])
