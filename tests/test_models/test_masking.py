import pytest
import torch

from foundry.models.masking import (
    ContiguousSpanMasking,
    MaskingStrategy,
    RandomPatchMasking,
)


class TestRandomPatchMasking:
    def test_is_masking_strategy(self):
        assert issubclass(RandomPatchMasking, MaskingStrategy)

    def test_output_shape(self):
        masking = RandomPatchMasking(mask_ratio=0.5)
        mask = masking.generate_mask(100)
        assert mask.shape == (100,)
        assert mask.dtype == torch.bool

    def test_at_least_one_masked(self):
        masking = RandomPatchMasking(mask_ratio=0.01)
        for _ in range(20):
            mask = masking.generate_mask(10)
            assert mask.any(), "At least one token must be masked"

    def test_approximate_ratio(self):
        masking = RandomPatchMasking(mask_ratio=0.5)
        total_masked = 0
        total_tokens = 0
        for _ in range(100):
            mask = masking.generate_mask(200)
            total_masked += mask.sum().item()
            total_tokens += 200
        actual_ratio = total_masked / total_tokens
        assert 0.4 < actual_ratio < 0.6

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomPatchMasking(mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomPatchMasking(mask_ratio=1.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomPatchMasking(mask_ratio=-0.1)

    def test_single_token(self):
        masking = RandomPatchMasking(mask_ratio=0.5)
        mask = masking.generate_mask(1)
        assert mask.shape == (1,)
        assert mask[0].item() is True


class TestContiguousSpanMasking:
    def test_is_masking_strategy(self):
        assert issubclass(ContiguousSpanMasking, MaskingStrategy)

    def test_output_shape(self):
        masking = ContiguousSpanMasking(mask_ratio=0.3, mean_span_length=3)
        mask = masking.generate_mask(100)
        assert mask.shape == (100,)
        assert mask.dtype == torch.bool

    def test_at_least_one_masked(self):
        masking = ContiguousSpanMasking(mask_ratio=0.1, mean_span_length=2)
        for _ in range(20):
            mask = masking.generate_mask(10)
            assert mask.any()

    def test_contiguous_spans_exist(self):
        masking = ContiguousSpanMasking(mask_ratio=0.5, mean_span_length=5)
        mask = masking.generate_mask(100)
        masked_indices = torch.where(mask)[0]
        if len(masked_indices) > 1:
            diffs = torch.diff(masked_indices)
            assert (diffs == 1).any(), (
                "With mean_span_length=5, should have contiguous spans"
            )

    def test_approximate_ratio(self):
        masking = ContiguousSpanMasking(mask_ratio=0.3, mean_span_length=3)
        total_masked = 0
        total_tokens = 0
        for _ in range(100):
            mask = masking.generate_mask(200)
            total_masked += mask.sum().item()
            total_tokens += 200
        actual_ratio = total_masked / total_tokens
        assert 0.15 < actual_ratio < 0.55

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            ContiguousSpanMasking(mask_ratio=0.0)
        with pytest.raises(ValueError, match="mean_span_length"):
            ContiguousSpanMasking(mask_ratio=0.3, mean_span_length=0)

    def test_small_num_tokens(self):
        masking = ContiguousSpanMasking(mask_ratio=0.5, mean_span_length=3)
        mask = masking.generate_mask(2)
        assert mask.shape == (2,)
        assert mask.any()
