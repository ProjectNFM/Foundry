"""Tests for SelectEEGChannels transform."""

from __future__ import annotations

import numpy as np
import pytest
from torch_brain.data import Data, Interval, RegularTimeSeries
from torch_brain.data.arraydict import ArrayDict

from foundry.data.transforms import SelectEEGChannels


# ─── Fixtures ────────────────────────────────────────────────────────────────


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
