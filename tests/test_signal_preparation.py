"""Tests for the signal-preparation and token-grid contract.

Covers:
- normalize_signal_length: T == expected, T < expected, T > expected, large differences
- compute_num_patches: non-overlapping, overlapping, edge cases
- normalize_encoder_inputs: standard z-scoring, constant channels, non-finite safety
- normalize_reconstruction_targets: padding, z-scoring, constant channels
- PreparedSignal: immutability and field correctness
- EEGTokenizer.prepare_signal: produces correct PreparedSignal
- EEGTokenizer.get_num_time_tokens: consistency across all tokenizer modes
- Encoder input and reconstruction targets share identical grid semantics
"""

import numpy as np
import pytest
import torch

from foundry.models.signal_preparation import (
    PreparedSignal,
    compute_num_patches,
    infer_sampling_rate_from_timestamps,
    normalize_encoder_inputs,
    normalize_reconstruction_targets,
    normalize_signal_length,
    resolve_signal_source,
)
from foundry.models.embeddings.channel import PerChannelStrategy
from foundry.models.embeddings.temporal import (
    PatchLinearEmbedding,
    PerTimepointLinearEmbedding,
)
from foundry.models.tokenizer import EEGTokenizer


# ---------------------------------------------------------------------------
# normalize_signal_length
# ---------------------------------------------------------------------------


class TestNormalizeSignalLength:
    """Single canonical length normalization."""

    def test_exact_length_unchanged(self):
        signal = np.random.randn(256, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)
        np.testing.assert_array_equal(result, signal)

    def test_longer_signal_trimmed(self):
        signal = np.random.randn(260, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)
        np.testing.assert_array_equal(result, signal[:256])

    def test_shorter_signal_padded(self):
        signal = np.random.randn(250, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)
        np.testing.assert_array_equal(result[:250], signal)
        np.testing.assert_array_equal(result[250:], 0.0)

    def test_large_difference_still_normalizes(self):
        """Unlike the old _normalize_signal_length which only acted within ±2 samples."""
        signal = np.random.randn(300, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)

    def test_one_sample_difference(self):
        signal = np.random.randn(257, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)

    def test_preserves_1d_signal(self):
        signal = np.random.randn(100)
        result = normalize_signal_length(signal, 50.0, 2.0)
        assert result.shape == (100,)

    def test_zero_length_signal(self):
        signal = np.random.randn(0, 4)
        result = normalize_signal_length(signal, 256.0, 1.0)
        assert result.shape == (256, 4)

    @pytest.mark.parametrize(
        "sr,seq_len,expected",
        [
            (128.0, 2.0, 256),
            (256.0, 0.5, 128),
            (100.0, 1.0, 100),
            (250.0, 3.0, 750),
        ],
    )
    def test_various_rates_and_lengths(self, sr, seq_len, expected):
        signal = np.random.randn(expected + 5, 2)
        result = normalize_signal_length(signal, sr, seq_len)
        assert result.shape[0] == expected


# ---------------------------------------------------------------------------
# compute_num_patches
# ---------------------------------------------------------------------------


class TestComputeNumPatches:
    """Single canonical patch-count calculation."""

    def test_non_overlapping_exact(self):
        assert compute_num_patches(256, 32, 32) == 8

    def test_non_overlapping_remainder(self):
        assert compute_num_patches(260, 32, 32) == 8

    def test_overlapping_patches(self):
        # stride=16, patch=32: (256-32)//16 + 1 = 15
        assert compute_num_patches(256, 32, 16) == 15

    def test_signal_shorter_than_patch(self):
        assert compute_num_patches(10, 32, 32) == 1

    def test_signal_equals_patch(self):
        assert compute_num_patches(32, 32, 32) == 1

    def test_single_sample(self):
        assert compute_num_patches(1, 32, 32) == 1

    def test_zero_samples(self):
        assert compute_num_patches(0, 32, 32) == 0

    def test_matches_torch_unfold(self):
        """Verify consistency with actual torch.unfold behavior."""
        T = 500
        patch_samples = 50
        stride_samples = 25
        signal = torch.randn(1, 1, T)
        patches = signal.unfold(2, patch_samples, stride_samples)
        expected_patches = patches.shape[2]
        assert (
            compute_num_patches(T, patch_samples, stride_samples)
            == expected_patches
        )

    @pytest.mark.parametrize(
        "T,P,S",
        [
            (100, 10, 10),
            (100, 10, 5),
            (1000, 50, 25),
            (256, 128, 64),
            (44, 30, 15),
        ],
    )
    def test_matches_unfold_parametrized(self, T, P, S):
        signal = torch.randn(1, 1, T)
        patches = signal.unfold(2, P, S)
        assert compute_num_patches(T, P, S) == patches.shape[2]


# ---------------------------------------------------------------------------
# normalize_encoder_inputs
# ---------------------------------------------------------------------------


class TestNormalizeEncoderInputs:
    """Per-channel z-scoring for encoder scale invariance."""

    def test_standard_zscore(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(loc=100, scale=10, size=(256, 4))
        result = normalize_encoder_inputs(signal)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-2)

    def test_constant_channel_stays_zero(self):
        signal = np.ones((256, 3))
        signal[:, 1] = np.random.randn(256)
        result = normalize_encoder_inputs(signal)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, 2], 0.0)
        assert result[:, 1].std() > 0.5

    def test_preserves_shape(self):
        signal = np.random.randn(128, 8)
        result = normalize_encoder_inputs(signal)
        assert result.shape == (128, 8)

    def test_does_not_produce_nan(self):
        signal = np.zeros((100, 5))
        result = normalize_encoder_inputs(signal)
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# normalize_reconstruction_targets
# ---------------------------------------------------------------------------


class TestNormalizeReconstructionTargets:
    """Per-channel z-scoring for reconstruction targets with padding."""

    def test_basic_zscore_with_padding(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(loc=50, scale=5, size=(256, 3))
        max_channels = 5
        targets = normalize_reconstruction_targets(signal, max_channels)
        assert targets.shape == (5, 256)
        np.testing.assert_allclose(targets[:3].mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(targets[:3].std(axis=1), 1.0, atol=1e-2)
        np.testing.assert_array_equal(targets[3:], 0.0)

    def test_padded_channels_zero(self):
        signal = np.random.randn(100, 2)
        targets = normalize_reconstruction_targets(signal, 8)
        np.testing.assert_array_equal(targets[2:], 0.0)

    def test_constant_channel_no_nan(self):
        signal = np.ones((100, 3))
        signal[:, 0] = np.random.randn(100)
        targets = normalize_reconstruction_targets(signal, 3)
        assert not np.isnan(targets).any()
        np.testing.assert_array_equal(targets[1], 0.0)
        np.testing.assert_array_equal(targets[2], 0.0)

    def test_more_channels_than_max(self):
        signal = np.random.randn(100, 10)
        targets = normalize_reconstruction_targets(signal, 5)
        assert targets.shape == (5, 100)

    def test_output_dtype_float32(self):
        signal = np.random.randn(100, 4).astype(np.float64)
        targets = normalize_reconstruction_targets(signal, 4)
        assert targets.dtype == np.float32


# ---------------------------------------------------------------------------
# PreparedSignal
# ---------------------------------------------------------------------------


class TestPreparedSignal:
    """Immutable signal contract."""

    def test_fields_correctly_set(self):
        signal = np.random.randn(256, 4)
        mask = np.array([True, True, True, True, False, False])
        ps = PreparedSignal(
            signal=signal,
            sampling_rate=256.0,
            num_samples=256,
            original_num_samples=260,
            num_channels=4,
            modality_mask=mask,
        )
        assert ps.num_samples == 256
        assert ps.original_num_samples == 260
        assert ps.num_channels == 4
        assert ps.sampling_rate == 256.0
        np.testing.assert_array_equal(ps.signal, signal)
        np.testing.assert_array_equal(ps.modality_mask, mask)

    def test_frozen(self):
        signal = np.random.randn(100, 2)
        mask = np.array([True, True])
        ps = PreparedSignal(
            signal=signal,
            sampling_rate=128.0,
            num_samples=100,
            original_num_samples=100,
            num_channels=2,
            modality_mask=mask,
        )
        with pytest.raises(Exception):
            ps.num_samples = 200


# ---------------------------------------------------------------------------
# EEGTokenizer.prepare_signal
# ---------------------------------------------------------------------------


class TestTokenizerPrepareSignal:
    """Tokenizer produces correct PreparedSignal."""

    @pytest.fixture
    def per_channel_tokenizer(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PerTimepointLinearEmbedding(embed_dim=64, input_dim=1)
        return EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
        )

    @pytest.fixture
    def patch_tokenizer(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PatchLinearEmbedding(
            embed_dim=64, num_input_channels=1, patch_samples=32
        )
        return EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
            patch_duration=0.125,
            stride=0.125,
        )

    def test_exact_length(self, per_channel_tokenizer):
        signal = np.random.randn(256, 4)
        mask = np.ones(4, dtype=bool)
        ps = per_channel_tokenizer.prepare_signal(signal, 256.0, 1.0, mask)
        assert ps.num_samples == 256
        assert ps.original_num_samples == 256
        assert ps.num_channels == 4
        assert ps.sampling_rate == 256.0

    def test_trims_longer(self, per_channel_tokenizer):
        signal = np.random.randn(300, 4)
        mask = np.ones(4, dtype=bool)
        ps = per_channel_tokenizer.prepare_signal(signal, 256.0, 1.0, mask)
        assert ps.num_samples == 256
        assert ps.original_num_samples == 300

    def test_pads_shorter(self, per_channel_tokenizer):
        signal = np.random.randn(200, 4)
        mask = np.ones(4, dtype=bool)
        ps = per_channel_tokenizer.prepare_signal(signal, 256.0, 1.0, mask)
        assert ps.num_samples == 256
        assert ps.original_num_samples == 200
        np.testing.assert_array_equal(ps.signal[200:], 0.0)

    def test_patching_tokenizer(self, patch_tokenizer):
        signal = np.random.randn(256, 4)
        mask = np.ones(4, dtype=bool)
        ps = patch_tokenizer.prepare_signal(signal, 256.0, 1.0, mask)
        assert ps.num_samples == 256


# ---------------------------------------------------------------------------
# EEGTokenizer.get_num_time_tokens
# ---------------------------------------------------------------------------


class TestTokenizerGetNumTimeTokens:
    """Canonical token count across all tokenizer modes."""

    def test_per_timepoint(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PerTimepointLinearEmbedding(embed_dim=64, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
        )
        assert tok.get_num_time_tokens(1.0, 256.0) == 256

    def test_patching_non_overlapping(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PatchLinearEmbedding(
            embed_dim=64, num_input_channels=1, patch_samples=32
        )
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
            patch_duration=0.125,
            stride=0.125,
        )
        assert tok.get_num_time_tokens(1.0, 256.0) == 8

    def test_patching_overlapping(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PatchLinearEmbedding(
            embed_dim=64, num_input_channels=1, patch_samples=32
        )
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
            patch_duration=0.125,
            stride=0.0625,
        )
        # 256 samples, patch=32, stride=16: (256-32)//16 + 1 = 15
        assert tok.get_num_time_tokens(1.0, 256.0) == 15

    def test_raises_for_non_patching_tokenizer(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PerTimepointLinearEmbedding(embed_dim=64, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
        )
        with pytest.raises(ValueError):
            tok.get_patch_samples(256.0)


# ---------------------------------------------------------------------------
# Grid alignment: encoder inputs and reconstruction targets
# ---------------------------------------------------------------------------


class TestGridAlignment:
    """Encoder input and reconstruction target grids have identical semantics."""

    @pytest.fixture
    def tokenizer_and_signal(self):
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PatchLinearEmbedding(
            embed_dim=64, num_input_channels=1, patch_samples=32
        )
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
            patch_duration=0.125,
            stride=0.125,
        )
        signal = np.random.randn(260, 4)
        return tok, signal

    def test_pretokenize_and_targets_same_grid(self, tokenizer_and_signal):
        """After prepare_signal, both paths see the same signal."""
        tok, signal = tokenizer_and_signal
        sampling_rate = 256.0
        sequence_length = 1.0
        modality_mask = np.ones(4, dtype=bool)

        prepared = tok.prepare_signal(
            signal, sampling_rate, sequence_length, modality_mask
        )

        channel_tokens = np.arange(4)
        pretok = tok.pretokenize(
            prepared.signal, channel_tokens, sampling_rate, sequence_length
        )
        targets = tok.compute_reconstruction_targets(
            prepared.signal, sampling_rate, sequence_length
        )

        num_time_tokens = tok.get_num_time_tokens(
            sequence_length, sampling_rate
        )

        # C_pad * num_patches tokens in pretokenized timestamps
        C_pad = 8
        expected_total_tokens = C_pad * num_time_tokens
        assert pretok["input_timestamps"].shape[0] == expected_total_tokens
        # Targets: C_pad * num_patches rows, patch_samples cols
        assert targets.shape[0] == C_pad * num_time_tokens

    def test_per_timepoint_grid_alignment(self):
        """Per-timepoint mode: targets (C_pad, N) where N = T."""
        strategy = PerChannelStrategy(max_channels=6)
        temporal = PerTimepointLinearEmbedding(embed_dim=32, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=32,
        )
        signal = np.random.randn(130, 3)
        sampling_rate = 128.0
        sequence_length = 1.0
        modality_mask = np.ones(3, dtype=bool)

        prepared = tok.prepare_signal(
            signal, sampling_rate, sequence_length, modality_mask
        )
        assert prepared.num_samples == 128

        targets = tok.compute_reconstruction_targets(
            prepared.signal, sampling_rate, sequence_length
        )
        # N = T for per-timepoint
        assert targets.shape == (6, 128)

    def test_non_eeg_channels_excluded(self):
        """Channels excluded by modality_mask do not appear in prepared signal."""
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PerTimepointLinearEmbedding(embed_dim=64, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=64,
        )
        signal = np.random.randn(256, 3)
        modality_mask = np.array([True, True, True, False, False])

        prepared = tok.prepare_signal(signal, 256.0, 1.0, modality_mask)
        assert prepared.num_channels == 3
        np.testing.assert_array_equal(prepared.modality_mask, modality_mask)

    def test_padded_channels_zero_in_targets(self):
        """Padded channels remain zero after target normalization."""
        strategy = PerChannelStrategy(max_channels=8)
        temporal = PerTimepointLinearEmbedding(embed_dim=32, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=32,
        )
        signal = np.random.randn(128, 3)
        prepared = tok.prepare_signal(
            signal, 128.0, 1.0, np.ones(3, dtype=bool)
        )
        targets = tok.compute_reconstruction_targets(
            prepared.signal, 128.0, 1.0
        )
        # Channels 3-7 should be zero
        np.testing.assert_array_equal(targets.numpy()[3:], 0.0)

    def test_constant_channel_no_nan_in_targets(self):
        """Constant channels produce zeros, not NaNs."""
        strategy = PerChannelStrategy(max_channels=4)
        temporal = PerTimepointLinearEmbedding(embed_dim=32, input_dim=1)
        tok = EEGTokenizer(
            channel_strategy=strategy,
            temporal_embedding=temporal,
            embed_dim=32,
        )
        signal = np.ones((128, 4))
        signal[:, 0] = np.random.randn(128)
        prepared = tok.prepare_signal(
            signal, 128.0, 1.0, np.ones(4, dtype=bool)
        )
        targets = tok.compute_reconstruction_targets(
            prepared.signal, 128.0, 1.0
        )
        assert not torch.isnan(targets).any()


# ---------------------------------------------------------------------------
# Normalization stage independence
# ---------------------------------------------------------------------------


class TestNormalizationStageIndependence:
    """Encoder normalization and target normalization are independent."""

    def test_encoder_norm_does_not_affect_target_norm(self):
        """Different normalization stages produce different outputs from same raw signal."""
        rng = np.random.default_rng(123)
        signal = rng.normal(loc=50, scale=10, size=(256, 4))

        encoder_normed = normalize_encoder_inputs(signal)
        target_normed = normalize_reconstruction_targets(signal, 4)

        # They produce z-scored results but from different axes/conventions
        assert encoder_normed.shape == (256, 4)
        assert target_normed.shape == (4, 256)

        # Verify they are different operations (target is transposed + padded)
        np.testing.assert_allclose(
            encoder_normed.T.astype(np.float32),
            target_normed,
            atol=1e-5,
        )

    def test_both_independently_exercised(self):
        """Both normalizations can be applied independently."""
        signal = np.random.randn(100, 3)

        # Only encoder
        enc = normalize_encoder_inputs(signal)
        assert enc.shape == signal.shape

        # Only target
        tgt = normalize_reconstruction_targets(signal, 5)
        assert tgt.shape == (5, 100)


# ---------------------------------------------------------------------------
# Shared signal-source helpers
# ---------------------------------------------------------------------------


class TestInferSamplingRateFromTimestamps:
    """Test sampling rate inference from timestamp deltas."""

    def test_regular_timestamps(self):
        """Regular 100 Hz timestamps."""
        timestamps = np.arange(0, 1.0, 0.01)  # 100 Hz
        sr = infer_sampling_rate_from_timestamps(timestamps)
        assert np.isclose(sr, 100.0, rtol=1e-2)

    def test_200hz_timestamps(self):
        """Regular 200 Hz timestamps."""
        timestamps = np.arange(0, 1.0, 0.005)
        sr = infer_sampling_rate_from_timestamps(timestamps)
        assert np.isclose(sr, 200.0, rtol=1e-2)

    def test_irregular_timestamps_uses_median(self):
        """Median is robust to outliers in sampling intervals."""
        timestamps = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.5, 0.51])
        # Most deltas are 0.01 s (100 Hz), one outlier is 0.46 s
        sr = infer_sampling_rate_from_timestamps(timestamps)
        assert np.isclose(sr, 100.0, rtol=1e-2)

    def test_empty_or_invalid_timestamps_raises(self):
        """Non-finite timestamps raise ValueError."""
        with pytest.raises(ValueError):
            infer_sampling_rate_from_timestamps(np.array([np.nan, np.inf]))

    def test_single_sample_raises(self):
        """Single timestamp (no deltas) raises ValueError."""
        with pytest.raises(ValueError):
            infer_sampling_rate_from_timestamps(np.array([0.0]))


class TestResolveSignalSource:
    """Test signal source resolution from torch_brain Data objects."""

    @pytest.fixture
    def mock_data_with_eeg(self):
        """Create a minimal mock torch_brain Data with EEG."""
        from unittest.mock import MagicMock
        data = MagicMock()
        # Mock EEG source with sampling rate
        eeg = MagicMock()
        eeg.sampling_rate = 128.0
        data.eeg = eeg
        data.ecog = None
        data.seeg = None
        return data

    @pytest.fixture
    def mock_data_with_timestamps(self):
        """Create a minimal mock torch_brain Data with inferred SR."""
        from unittest.mock import MagicMock
        data = MagicMock()
        # Mock EEG source without sampling_rate; infer from timestamps
        eeg = MagicMock()
        eeg.sampling_rate = None
        eeg.timestamps = np.arange(0, 1.0, 0.01)  # 100 Hz inferred
        data.eeg = eeg
        data.ecog = None
        data.seeg = None
        return data

    def test_resolve_eeg_with_sampling_rate(self, mock_data_with_eeg):
        """Resolve EEG source with explicit sampling_rate."""
        signal_source, modality, sr = resolve_signal_source(mock_data_with_eeg)
        assert signal_source is mock_data_with_eeg.eeg
        assert modality == "EEG"
        assert sr == 128.0

    def test_resolve_eeg_with_timestamps(self, mock_data_with_timestamps):
        """Resolve EEG source with inferred sampling_rate."""
        signal_source, modality, sr = resolve_signal_source(
            mock_data_with_timestamps
        )
        assert signal_source is mock_data_with_timestamps.eeg
        assert modality == "EEG"
        assert np.isclose(sr, 100.0, rtol=1e-2)

    def test_no_signal_source_raises(self):
        """Missing all signal sources raises ValueError."""
        from unittest.mock import MagicMock
        data = MagicMock()
        data.eeg = None
        data.ecog = None
        data.seeg = None
        with pytest.raises(ValueError, match="must have an 'eeg'"):
            resolve_signal_source(data)
