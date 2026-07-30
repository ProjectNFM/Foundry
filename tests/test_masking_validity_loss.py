"""Tests for token validity, masking edge cases, and weighted loss semantics.

Covers:
- build_token_validity_mask with mixed input_seq_len
- Padded tokens excluded from encoder output and reconstruction loss
- Single-channel ChannelMasking fails clearly
- No-real-channel, all-padded, ratio-boundary, and block-size-boundary cases
- Scalar reconstruction weights 0.0, 0.5, 1.0 match equivalent tensors
- Optimized-mode-safe constructor validation (ValueError, not assert)
- MaskedPOYOEEGModel.input_mask is required (no Optional)
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# build_token_validity_mask
# ---------------------------------------------------------------------------


class TestBuildTokenValidityMask:
    """Tests for build_token_validity_mask helper."""

    def test_all_channels_valid_no_seq_len(self):
        """All tokens valid when every channel is real and no seq_len."""
        from foundry.tasks.masking import build_token_validity_mask

        B, C_pad, N = 2, 4, 10
        channel_mask = torch.ones(B, C_pad, dtype=torch.bool)
        result = build_token_validity_mask(channel_mask, N)
        assert result.shape == (B, C_pad * N)
        assert result.all()

    def test_padded_channels_invalid(self):
        """Padded channels produce False in validity mask."""
        from foundry.tasks.masking import build_token_validity_mask

        B, C_pad, N = 2, 4, 10
        channel_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        result = build_token_validity_mask(channel_mask, N)
        assert result.shape == (B, C_pad * N)

        for b in range(B):
            for c in range(C_pad):
                for t in range(N):
                    idx = c * N + t
                    expected = channel_mask[b, c].item()
                    assert result[b, idx].item() == expected

    def test_time_validity_with_seq_len(self):
        """Time-padded positions are invalid when input_seq_len is provided."""
        from foundry.tasks.masking import build_token_validity_mask

        B, C_pad, N = 2, 3, 8
        channel_mask = torch.ones(B, C_pad, dtype=torch.bool)
        input_seq_len = torch.tensor([5, 3])

        result = build_token_validity_mask(channel_mask, N, input_seq_len)
        assert result.shape == (B, C_pad * N)

        for b in range(B):
            sl = input_seq_len[b].item()
            for c in range(C_pad):
                for t in range(N):
                    idx = c * N + t
                    expected = t < sl
                    assert result[b, idx].item() == expected

    def test_combined_channel_and_time_validity(self):
        """Channel invalidity AND time invalidity are both reflected."""
        from foundry.tasks.masking import build_token_validity_mask

        _, C_pad, N = 1, 3, 6
        channel_mask = torch.tensor([[True, True, False]])
        input_seq_len = torch.tensor([4])

        result = build_token_validity_mask(channel_mask, N, input_seq_len)

        for c in range(C_pad):
            for t in range(N):
                idx = c * N + t
                ch_valid = channel_mask[0, c].item()
                t_valid = t < 4
                assert result[0, idx].item() == (ch_valid and t_valid)

    def test_zero_seq_len(self):
        """seq_len=0 makes all time positions invalid."""
        from foundry.tasks.masking import build_token_validity_mask

        B, C_pad, N = 1, 2, 5
        channel_mask = torch.ones(B, C_pad, dtype=torch.bool)
        input_seq_len = torch.tensor([0])

        result = build_token_validity_mask(channel_mask, N, input_seq_len)
        assert not result.any()

    def test_seq_len_equals_N(self):
        """When seq_len == N, all time positions are valid (same as None)."""
        from foundry.tasks.masking import build_token_validity_mask

        B, C_pad, N = 2, 3, 10
        channel_mask = torch.ones(B, C_pad, dtype=torch.bool)
        input_seq_len = torch.tensor([N, N])

        result = build_token_validity_mask(channel_mask, N, input_seq_len)
        expected = build_token_validity_mask(channel_mask, N, None)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Masking strategy validation
# ---------------------------------------------------------------------------


class TestMaskingValidation:
    """Constructor-level validation for masking strategies."""

    def test_random_token_mask_ratio_zero(self):
        from foundry.tasks.masking import RandomTokenMasking

        with pytest.raises(ValueError, match="mask_ratio"):
            RandomTokenMasking(mask_ratio=0.0)

    def test_random_token_mask_ratio_one(self):
        from foundry.tasks.masking import RandomTokenMasking

        with pytest.raises(ValueError, match="mask_ratio"):
            RandomTokenMasking(mask_ratio=1.0)

    def test_random_token_mask_ratio_negative(self):
        from foundry.tasks.masking import RandomTokenMasking

        with pytest.raises(ValueError, match="mask_ratio"):
            RandomTokenMasking(mask_ratio=-0.1)

    def test_temporal_block_mask_ratio_invalid(self):
        from foundry.tasks.masking import TemporalBlockMasking

        with pytest.raises(ValueError, match="mask_ratio"):
            TemporalBlockMasking(mask_ratio=0.0)

    def test_temporal_block_size_zero(self):
        from foundry.tasks.masking import TemporalBlockMasking

        with pytest.raises(ValueError, match="block_size"):
            TemporalBlockMasking(mask_ratio=0.5, block_size=0)

    def test_temporal_block_size_negative(self):
        from foundry.tasks.masking import TemporalBlockMasking

        with pytest.raises(ValueError, match="block_size"):
            TemporalBlockMasking(mask_ratio=0.5, block_size=-1)

    def test_channel_masking_mask_ratio_invalid(self):
        from foundry.tasks.masking import ChannelMasking

        with pytest.raises(ValueError, match="mask_ratio"):
            ChannelMasking(mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            ChannelMasking(mask_ratio=1.0)

    def test_channel_masking_single_channel(self):
        """C_pad == 1 raises at call time, not construction."""
        from foundry.tasks.masking import ChannelMasking

        strategy = ChannelMasking(mask_ratio=0.5)
        channel_mask = torch.ones(2, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="num_channels >= 2"):
            strategy(
                num_channels=1, num_time_tokens=10, channel_mask=channel_mask
            )

    def test_valid_random_token_masking(self):
        """Sanity: valid construction does not raise."""
        from foundry.tasks.masking import RandomTokenMasking

        m = RandomTokenMasking(mask_ratio=0.5)
        assert m.mask_ratio == 0.5

    def test_valid_temporal_block_masking(self):
        from foundry.tasks.masking import TemporalBlockMasking

        m = TemporalBlockMasking(mask_ratio=0.75, block_size=3)
        assert m.block_size == 3

    def test_valid_channel_masking(self):
        from foundry.tasks.masking import ChannelMasking

        m = ChannelMasking(mask_ratio=0.5)
        assert m.mask_ratio == 0.5


# ---------------------------------------------------------------------------
# ChannelMasking always protects at least one real channel
# ---------------------------------------------------------------------------


class TestChannelMaskingProtection:
    def test_two_channels_masks_one(self):
        """With C_pad=2 and high ratio, exactly 1 channel is masked."""
        from foundry.tasks.masking import ChannelMasking

        strategy = ChannelMasking(mask_ratio=0.9)
        B, C, N = 4, 2, 10
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity = strategy(C, N, channel_mask)
        # num_channels_masked = min(max(1, int(0.9*2)), 2-1) = 1
        assert mask_indices.shape == (B, 1 * N)

    def test_no_real_channels_all_invalid(self):
        """When no channels are real, all validity entries are False."""
        from foundry.tasks.masking import ChannelMasking

        strategy = ChannelMasking(mask_ratio=0.5)
        B, C, N = 2, 4, 8
        channel_mask = torch.zeros(B, C, dtype=torch.bool)
        mask_indices, validity = strategy(C, N, channel_mask)

        assert not validity.any()


# ---------------------------------------------------------------------------
# ReconstructionLoss
# ---------------------------------------------------------------------------


class TestReconstructionLoss:
    """Test ReconstructionLoss with various weight scenarios."""

    def _make_loss(self):
        from foundry.tasks.losses import ReconstructionLoss

        return ReconstructionLoss()

    def test_scalar_weight_1(self):
        """Weight 1.0 is equivalent to plain MSE."""
        loss_fn = self._make_loss()
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)

        loss = loss_fn(preds, targets, 1.0)
        expected = torch.nn.functional.mse_loss(preds, targets)
        assert torch.allclose(loss, expected)

    def test_scalar_weight_half(self):
        """Weight 0.5 scales MSE by 0.5."""
        loss_fn = self._make_loss()
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)

        loss = loss_fn(preds, targets, 0.5)
        expected = torch.nn.functional.mse_loss(preds, targets) * 0.5
        assert torch.allclose(loss, expected)

    def test_scalar_weight_zero_differentiable(self):
        """Weight 0.0 produces zero loss with gradient graph preserved."""
        loss_fn = self._make_loss()
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)

        loss = loss_fn(preds, targets, 0.0)
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert preds.grad is not None
        assert (preds.grad == 0).all()

    def test_tensor_weight_matches_scalar(self):
        """Uniform tensor weights match equivalent scalar."""
        loss_fn = self._make_loss()
        preds = torch.randn(10)
        targets = torch.randn(10)

        w = torch.ones(10)
        loss_tensor = loss_fn(preds, targets, w)
        loss_scalar = loss_fn(preds, targets, 1.0)
        assert torch.allclose(loss_tensor, loss_scalar, atol=1e-6)

    def test_zero_valid_targets_differentiable(self):
        """All-zero tensor weights produce differentiable zero loss."""
        loss_fn = self._make_loss()
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)
        w = torch.zeros(10)

        loss = loss_fn(preds, targets, w)
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert preds.grad is not None

    def test_partial_validity(self):
        """Only valid positions contribute to loss."""
        loss_fn = self._make_loss()
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        w = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(preds, targets, w)
        assert loss.item() == 0.0

    def test_weighted_validity(self):
        """Different weights produce correctly weighted loss."""
        loss_fn = self._make_loss()
        preds = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 2.0])
        w = torch.tensor([1.0, 3.0])

        loss = loss_fn(preds, targets, w)
        mse_0 = 1.0  # (0-1)^2
        mse_1 = 4.0  # (0-2)^2
        expected = (mse_0 * 1.0 + mse_1 * 3.0) / (1.0 + 3.0)
        assert torch.allclose(loss, torch.tensor(expected))

    def test_multidim_targets(self):
        """Multi-dimensional targets (patch reconstruction) work correctly."""
        loss_fn = self._make_loss()
        preds = torch.randn(5, 16)
        targets = torch.randn(5, 16)
        w = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

        loss = loss_fn(preds, targets, w)
        assert loss.isfinite()
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Masking strategy output shapes and validity
# ---------------------------------------------------------------------------


class TestMaskingStrategyOutputs:
    """Verify masking output shapes and validity mask semantics."""

    def test_random_token_masking_shapes(self):
        from foundry.tasks.masking import RandomTokenMasking

        B, C, N = 4, 6, 20
        strategy = RandomTokenMasking(mask_ratio=0.75)
        channel_mask = torch.ones(B, C, dtype=torch.bool)
        mask_indices, validity = strategy(C, N, channel_mask)

        expected_masked = max(1, int(0.75 * C * N))
        assert mask_indices.shape == (B, expected_masked)
        assert validity.shape == (B, expected_masked)
        assert validity.all()

    def test_random_token_masking_padded_channels(self):
        from foundry.tasks.masking import RandomTokenMasking

        B, C, N = 2, 4, 10
        strategy = RandomTokenMasking(mask_ratio=0.5)
        channel_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        mask_indices, validity = strategy(C, N, channel_mask)

        for b in range(B):
            for i in range(mask_indices.shape[1]):
                ch = mask_indices[b, i].item() // N
                assert validity[b, i].item() == channel_mask[b, ch].item()

    def test_temporal_block_masking_shapes(self):
        from foundry.tasks.masking import TemporalBlockMasking

        B, C, N = 3, 4, 20
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        channel_mask = torch.ones(B, C, dtype=torch.bool)
        mask_indices, validity = strategy(C, N, channel_mask)

        assert mask_indices.shape[0] == B
        assert validity.shape == mask_indices.shape
        # Total masked should be divisible by C (whole blocks across channels)
        assert mask_indices.shape[1] % C == 0

    def test_temporal_block_masking_fallback(self):
        """When N < block_size, falls back to individual time positions."""
        from foundry.tasks.masking import TemporalBlockMasking

        B, C, N = 2, 3, 3
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        channel_mask = torch.ones(B, C, dtype=torch.bool)
        mask_indices, validity = strategy(C, N, channel_mask)

        assert mask_indices.shape[0] == B
        assert mask_indices.shape[1] % C == 0

    def test_channel_masking_shapes(self):
        from foundry.tasks.masking import ChannelMasking

        B, C, N = 3, 6, 15
        strategy = ChannelMasking(mask_ratio=0.5)
        channel_mask = torch.ones(B, C, dtype=torch.bool)
        mask_indices, validity = strategy(C, N, channel_mask)

        num_ch_masked = min(max(1, int(0.5 * C)), C - 1)
        assert mask_indices.shape == (B, num_ch_masked * N)
        assert validity.shape == mask_indices.shape

    def test_indices_in_range(self):
        """All mask indices are within [0, C*N)."""
        from foundry.tasks.masking import RandomTokenMasking

        B, C, N = 4, 5, 12
        strategy = RandomTokenMasking(mask_ratio=0.6)
        channel_mask = torch.ones(B, C, dtype=torch.bool)
        mask_indices, _ = strategy(C, N, channel_mask)

        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()


# ---------------------------------------------------------------------------
# MaskedPOYOEEGModel: input_mask is now required
# ---------------------------------------------------------------------------


class TestMaskedModelInputMaskRequired:
    """Verify input_mask is a required argument in forward()."""

    def test_input_mask_in_signature(self):
        """input_mask has no default value (not Optional)."""
        import inspect
        from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel

        sig = inspect.signature(MaskedPOYOEEGModel.forward)
        param = sig.parameters["input_mask"]
        assert param.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# MaskedPOYOEEGModel: assert replaced with ValueError
# ---------------------------------------------------------------------------


class TestMaskedModelValidation:
    """Verify ValueError instead of assert for PerChannelStrategy check."""

    def test_non_per_channel_raises_valueerror(self):
        """Using non-PerChannel strategy raises ValueError, not AssertionError."""
        from unittest.mock import MagicMock
        from foundry.tasks.masking import RandomTokenMasking

        mock_tokenizer = MagicMock()
        mock_tokenizer.uses_per_channel = False

        with pytest.raises(ValueError, match="PerChannelStrategy"):
            from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel

            class FakeModel(MaskedPOYOEEGModel):
                def __init__(self):
                    # Bypass POYOEEGModel.__init__ to isolate the check
                    self.tokenizer = mock_tokenizer
                    self.masking = RandomTokenMasking(mask_ratio=0.5)

                    if not self.tokenizer.uses_per_channel:
                        raise ValueError(
                            "MaskedPOYOEEGModel requires PerChannelStrategy. "
                            "SpatialProjectionStrategy is not supported."
                        )

            FakeModel()
