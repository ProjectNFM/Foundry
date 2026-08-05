"""Tests for LaBraM masking and forward pass.

Covers:
- apply_masking: uses MaskingStrategy, vectorized zero-fill, symmetric BEiT
- LaBraMForMaskedEEGModeling.forward: accepts pre-masked patches, produces logits
- Mask polarity: True = masked position
"""

import numpy as np
import pytest
import torch

from foundry.models.masked_labram import (
    MaskedLaBram,
    apply_masking,
)
from foundry.tasks.masking import RandomTokenMasking, TemporalBlockMasking


class TestApplyMasking:
    """Test masking application with optional symmetry."""

    @pytest.fixture
    def sample_patches(self):
        """Create sample patches [B, C, N, 200]."""
        B, C, N, P = 2, 3, 4, 200
        patches = torch.randn(B, C, N, P)
        return patches

    def test_mask_strategy_default(self, sample_patches):
        """Default masking strategy is RandomTokenMasking(0.5)."""
        masked, bool_mask = apply_masking(sample_patches, masking=None, symmetric=False)
        B, C, N, P = sample_patches.shape
        assert masked.shape == sample_patches.shape
        assert bool_mask.shape == (B, C * N)
        # At 0.5 ratio, roughly half the mask should be True
        assert (bool_mask.sum() / bool_mask.numel()) > 0.4
        assert (bool_mask.sum() / bool_mask.numel()) < 0.6

    def test_zero_fill_where_masked(self, sample_patches):
        """Positions where bool_mask=True are zeroed."""
        masking = RandomTokenMasking(mask_ratio=0.3)
        masked, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=False
        )
        B, C, N, P = sample_patches.shape
        # Reshape masked to [B, C*N, P] for easy indexing
        masked_flat = masked.reshape(B, C * N, P)
        # Where bool_mask is True, all patch samples should be zero
        for b in range(B):
            masked_positions = bool_mask[b]
            assert (masked_flat[b, masked_positions] == 0.0).all()

    def test_symmetric_masking_doubles_batch(self, sample_patches):
        """Symmetric masking returns 2B samples: mask and complement."""
        masking = RandomTokenMasking(mask_ratio=0.5)
        masked, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=True
        )
        B = sample_patches.shape[0]
        # Batch should be doubled
        assert masked.shape[0] == 2 * B
        assert bool_mask.shape[0] == 2 * B

    def test_symmetric_masks_are_complementary(self, sample_patches):
        """First B and second B masks are complements."""
        masking = RandomTokenMasking(mask_ratio=0.5)
        _, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=True
        )
        B = sample_patches.shape[0]
        # Check complement: mask_a XOR mask_b == all True
        mask_a = bool_mask[:B]
        mask_b = bool_mask[B:]
        assert (mask_a != mask_b).all()  # Every position is opposite

    def test_custom_masking_strategy(self, sample_patches):
        """Apply custom MaskingStrategy."""
        masking = TemporalBlockMasking(mask_ratio=0.25, block_size=2)
        masked, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=False
        )
        B, C, N, P = sample_patches.shape
        # Block masking should produce temporal structure
        assert bool_mask.shape == (B, C * N)
        # Some positions should be masked
        assert bool_mask.sum() > 0

    def test_mask_polarity_consistent(self, sample_patches):
        """True consistently means masked (zeroed) position."""
        masking = RandomTokenMasking(mask_ratio=0.5)
        masked, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=False
        )
        B, C, N, P = sample_patches.shape
        masked_flat = masked.reshape(B, C * N, P)
        # Check each position: if bool_mask[b, i] is True, then
        # all P patch values at (b, i) should be zero
        for b in range(B):
            for i in range(C * N):
                if bool_mask[b, i]:
                    assert (masked_flat[b, i] == 0.0).all()

    def test_unmasked_patches_unchanged(self, sample_patches):
        """Unmasked positions (bool_mask=False) should match input."""
        masking = RandomTokenMasking(mask_ratio=0.5)
        original = sample_patches.clone()
        masked, bool_mask = apply_masking(
            sample_patches, masking=masking, symmetric=False
        )
        B, C, N, P = sample_patches.shape
        masked_flat = masked.reshape(B, C * N, P)
        original_flat = original.reshape(B, C * N, P)
        # Check unmasked positions match
        for b in range(B):
            for i in range(C * N):
                if not bool_mask[b, i]:
                    assert torch.allclose(
                        masked_flat[b, i], original_flat[b, i], atol=1e-6
                    )


class TestMaskedLaBramForward:
    """Test MaskedLaBram Stage-2 model forward pass."""

    @pytest.fixture
    def model(self):
        """Create a small Stage-2 model."""
        return MaskedLaBram(
            num_channels=4,
            num_samples=1600,
            embed_dim=64,  # Smaller for testing
            num_layers=2,
            num_heads=4,
            vocab_size=512,
            drop_path_prob=0.0,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of patches."""
        B, C, N, P = 2, 4, 8, 200
        patches = torch.randn(B, C, N, P)
        channel_names = ["FPZ", "FZ", "CZ", "PZ"]
        return patches, channel_names

    def test_forward_with_channel_index(self, model, sample_batch):
        """MaskedLaBram forward pass accepts channel_index and produces logits."""
        patches, names = sample_batch
        from foundry.models.patch_utils import labram_names_to_index_tensor
        
        channel_index = labram_names_to_index_tensor(names)
        logits = model(input_patches=patches, channel_index=channel_index)

        B, C, N, P = patches.shape
        # Logits shape: [B, N_seq, vocab_size]
        # N_seq = N (patches per channel), assuming no pooling in return_all_tokens=True
        assert logits.shape[0] == B
        assert logits.shape[2] == model.vocab_size

    def test_forward_without_channel_names_raises(self, model, sample_batch):
        """MaskedLaBram forward without channel_index raises if names not initialized."""
        patches, _ = sample_batch
        with pytest.raises(
            RuntimeError, match="Channel names not initialized"
        ):
            model(input_patches=patches, channel_index=None)

    def test_forward_accepts_pre_masked_patches(self, model, sample_batch):
        """MaskedLaBram works with pre-masked patches (zeros in place)."""
        patches, names = sample_batch
        from foundry.models.patch_utils import labram_names_to_index_tensor
        
        # Pre-mask some patches
        patches_masked = patches.clone()
        patches_masked[0, 0, 0, :] = 0.0  # Zero out one patch
        patches_masked[0, 1, 2, :] = 0.0

        channel_index = labram_names_to_index_tensor(names)
        logits = model(
            input_patches=patches_masked, channel_index=channel_index
        )

        B = patches.shape[0]
        assert logits.shape[0] == B
        # Should produce valid logits despite masking
        assert not torch.isnan(logits).any()

    def test_mask_token_not_used(self, model):
        """mask_token parameter is not used (cleaned up)."""
        # Model should not have a learnable mask_token parameter
        assert not hasattr(model, "mask_token") or model.mask_token is None

    def test_unused_bool_mask_parameter_ignored(self, model, sample_batch):
        """bool_mask parameter is accepted but ignored."""
        patches, names = sample_batch
        from foundry.models.patch_utils import labram_names_to_index_tensor
        
        channel_index = labram_names_to_index_tensor(names)
        bool_mask = torch.ones(patches.shape[0], patches.shape[1] * patches.shape[2])

        # Should work with bool_mask (but ignores it)
        logits = model(
            input_patches=patches,
            bool_mask=bool_mask,
            channel_index=channel_index,
        )
        assert logits.shape[0] == patches.shape[0]

    def test_output_not_nan_or_inf(self, model, sample_batch):
        """Output logits are valid (not NaN/Inf)."""
        patches, names = sample_batch
        from foundry.models.patch_utils import labram_names_to_index_tensor
        
        channel_index = labram_names_to_index_tensor(names)
        logits = model(input_patches=patches, channel_index=channel_index)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
