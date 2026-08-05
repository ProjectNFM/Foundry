"""Tests for LaBraM patch utilities and channel mapping.

Covers:
- to_labram_channel_name: channel ID mapping
- resolve_labram_channels: channel resolution and ordering
- labram_names_to_index_tensor / labram_index_tensor_to_names: encode/decode
- prepare_labram_continuous_signal: signal preparation pipeline
- extract_labram_patches: patch extraction shape and content
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from foundry.models.patch_utils import (
    extract_labram_patches,
    labram_index_tensor_to_names,
    labram_names_to_index_tensor,
    prepare_labram_continuous_signal,
    resolve_labram_channels,
    to_labram_channel_name,
)


class TestToLabramChannelName:
    """Test channel ID to LaBraM canonical name mapping."""

    def test_direct_match(self):
        """Exact channel name match."""
        assert to_labram_channel_name("Fpz") == "FPZ"
        assert to_labram_channel_name("Cz") == "CZ"

    def test_case_insensitive(self):
        """Case-insensitive matching."""
        assert to_labram_channel_name("fpz") == "FPZ"
        assert to_labram_channel_name("cZ") == "CZ"

    def test_eeg_prefix_stripped(self):
        """Strip 'EEG' prefix."""
        assert to_labram_channel_name("EEG Fpz") == "FPZ"
        assert to_labram_channel_name("EEG Fpz-Cz") == "FPZ"

    def test_bipolar_takes_first_valid(self):
        """Bipolar montage: take first electrode that's in canonical order."""
        assert to_labram_channel_name("Fpz-Cz") == "FPZ"
        assert to_labram_channel_name("Cz-Pz") == "CZ"

    def test_no_match_returns_none(self):
        """Unknown channel returns None."""
        assert to_labram_channel_name("UNKNOWN") is None
        assert to_labram_channel_name("XYZ") is None


class TestResolveLabramChannels:
    """Test channel filtering and ordering to LaBraM canonical order."""

    def test_exact_match_preserves_order(self):
        """Channels matching canonical order are kept and ordered correctly."""
        channel_ids = ["Fpz", "Fz", "Cz"]
        keep_indices, names = resolve_labram_channels(channel_ids)
        assert keep_indices == [0, 1, 2]
        assert names == ["FPZ", "FZ", "CZ"]

    def test_reorder_to_canonical(self):
        """Channels out of order are reordered to canonical."""
        channel_ids = ["Cz", "Fpz", "Fz"]  # Out of canonical order
        keep_indices, names = resolve_labram_channels(channel_ids)
        # Should be reordered to canonical: Fpz, Fz, Cz
        assert names == ["FPZ", "FZ", "CZ"]
        # keep_indices should point back to original positions
        assert keep_indices == [1, 2, 0]

    def test_unmapped_channels_dropped(self):
        """Unmapped channels are dropped."""
        channel_ids = ["Fpz", "UNKNOWN", "Cz"]
        keep_indices, names = resolve_labram_channels(channel_ids)
        assert "UNKNOWN" not in names
        assert len(names) == 2

    def test_min_3_channels_warning(self):
        """Warn if fewer than 3 channels match."""
        channel_ids = ["Fpz"]
        with pytest.warns(UserWarning, match="Only 1 channels matched"):
            resolve_labram_channels(channel_ids)

    def test_no_channels_raises(self):
        """No matching channels raises ValueError."""
        channel_ids = ["UNKNOWN", "INVALID"]
        with pytest.raises(ValueError, match="No channels could be mapped"):
            resolve_labram_channels(channel_ids)


class TestLabramIndexTensor:
    """Test encoding/decoding channel names as indices."""

    def test_names_to_index(self):
        """Encode channel names to indices."""
        names = ["FPZ", "FZ", "CZ"]
        indices = labram_names_to_index_tensor(names)
        assert indices.dtype == torch.long
        assert indices.shape == (3,)
        # Indices should be distinct and valid
        assert len(set(indices.tolist())) == 3

    def test_index_to_names(self):
        """Decode indices to channel names."""
        names = ["FPZ", "FZ", "CZ"]
        indices = labram_names_to_index_tensor(names)
        decoded = labram_index_tensor_to_names(indices)
        assert decoded == names

    def test_roundtrip_consistency(self):
        """Roundtrip encode/decode preserves names."""
        original = ["FPZ", "CZ", "FZ"]
        indices = labram_names_to_index_tensor(original)
        decoded = labram_index_tensor_to_names(indices)
        assert decoded == original

    def test_batched_2d_index_tensor(self):
        """Decode 2D (batched) index tensor."""
        names = ["FPZ", "FZ", "CZ"]
        indices = labram_names_to_index_tensor(names)
        # Repeat for batch
        batch_indices = indices.unsqueeze(0).repeat(4, 1)  # (4, 3)
        # Decode using first sample (assumed homogeneous)
        decoded = labram_index_tensor_to_names(batch_indices)
        assert decoded == names


class TestPrepareLaBramContinuousSignal:
    """Test shared signal preparation for LaBraM models."""

    @pytest.fixture
    def mock_data(self):
        """Create minimal mock torch_brain Data."""
        from unittest.mock import MagicMock
        data = MagicMock()

        # Mock EEG signal (256 samples, 4 channels)
        eeg = MagicMock()
        eeg.signal = np.random.randn(256, 4).astype(np.float32)
        eeg.sampling_rate = 256.0
        eeg.timestamps = np.arange(256) / 256.0

        data.eeg = eeg
        data.ecog = None
        data.seeg = None

        # Mock channels
        data.channels = MagicMock()
        data.channels.id = np.array(["Fpz", "Fz", "Cz", "Pz"])
        data.channels.type = np.array(["EEG", "EEG", "EEG", "EEG"])

        return data

    def test_shape_after_resample_and_length_norm(self, mock_data):
        """Output is (T_norm, C) at target sampling rate, length-normalized."""
        signal, names = prepare_labram_continuous_signal(
            mock_data, num_channels=4, num_samples=1600, target_sampling_rate=200
        )
        # num_samples = 1600, sr = 200 -> sequence_length = 8.0 s
        # At 200 Hz: 8.0 * 200 = 1600 samples
        assert signal.shape == (1600, 4)
        assert signal.dtype == np.float32
        assert names == ["FPZ", "FZ", "CZ", "PZ"]

    def test_resample_to_target_rate(self, mock_data):
        """Input signal at different rate is resampled."""
        mock_data.eeg.sampling_rate = 512.0  # 2x target
        signal, _ = prepare_labram_continuous_signal(
            mock_data, num_channels=4, num_samples=1600, target_sampling_rate=200
        )
        # After resample + length norm, should be at target rate
        assert signal.shape[0] == 1600

    def test_channel_filtering_and_order(self, mock_data):
        """Channels are filtered to canonical order."""
        # Add non-EEG channel
        mock_data.channels.type = np.array(["EEG", "EEG", "EMG", "EEG"])
        signal, names = prepare_labram_continuous_signal(
            mock_data, num_channels=3, num_samples=1600, target_sampling_rate=200
        )
        # EMG channel dropped, EEG channels reordered to canonical
        assert len(names) == 3
        assert "EMG" not in names

    def test_sanitize_non_finite(self, mock_data):
        """Non-finite values are replaced with zero."""
        mock_data.eeg.signal[10, 0] = np.nan
        mock_data.eeg.signal[20, 1] = np.inf
        signal, _ = prepare_labram_continuous_signal(
            mock_data, num_channels=4, num_samples=1600, target_sampling_rate=200
        )
        assert not np.isnan(signal).any()
        assert not np.isinf(signal).any()


class TestExtractLabramPatches:
    """Test patch extraction for LaBraM pre-training."""

    @pytest.fixture
    def mock_data_200hz(self):
        """Create mock data at 200 Hz (LaBraM standard)."""
        from unittest.mock import MagicMock
        data = MagicMock()

        # 1600 samples at 200 Hz = 8 seconds
        # With 1s non-overlapping patches -> 8 patches, each 200 samples
        eeg = MagicMock()
        eeg.signal = np.random.randn(1600, 4).astype(np.float32)
        eeg.sampling_rate = 200.0
        eeg.timestamps = np.arange(1600) / 200.0

        data.eeg = eeg
        data.ecog = None
        data.seeg = None

        data.channels = MagicMock()
        data.channels.id = np.array(["Fpz", "Fz", "Cz", "Pz"])
        data.channels.type = np.array(["EEG", "EEG", "EEG", "EEG"])

        return data

    def test_patch_shape_cxnxp(self, mock_data_200hz):
        """Patches have shape (C, N_patches, 200)."""
        patches, names = extract_labram_patches(
            mock_data_200hz, num_channels=4, num_samples=1600
        )
        # 1600 samples / 200 samples per patch = 8 patches
        assert patches.shape == (4, 8, 200)
        assert names == ["FPZ", "FZ", "CZ", "PZ"]

    def test_patch_count_from_num_samples(self, mock_data_200hz):
        """Number of patches matches num_samples / patch_size."""
        patches, _ = extract_labram_patches(
            mock_data_200hz, num_channels=4, num_samples=1600
        )
        # 1600 / 200 = 8 patches
        assert patches.shape[1] == 8

    def test_patches_are_tensor(self, mock_data_200hz):
        """Patches are torch tensors."""
        patches, _ = extract_labram_patches(
            mock_data_200hz, num_channels=4, num_samples=1600
        )
        assert isinstance(patches, torch.Tensor)

    def test_channel_mismatch_warning(self, mock_data_200hz):
        """Warn if actual channels differ from expected."""
        # Only 3 channels actually match (one unmapped)
        mock_data_200hz.channels.id = np.array(["Fpz", "Fz", "Cz", "UNKNOWN"])
        with pytest.warns(UserWarning, match="Expected 4 channels"):
            extract_labram_patches(
                mock_data_200hz, num_channels=4, num_samples=1600
            )
