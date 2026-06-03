import torch

from foundry.training.masking import build_token_mask, generate_temporal_mask


class TestGenerateTemporalMask:
    def test_span_within_bounds(self):
        num_time_tokens = 100
        mask_ratio = 0.75
        gen = torch.Generator().manual_seed(0)

        start, end = generate_temporal_mask(
            num_time_tokens, mask_ratio, generator=gen
        )

        assert start >= 0
        assert end <= num_time_tokens
        assert start < end

    def test_span_length_matches_mask_ratio(self):
        num_time_tokens = 100
        mask_ratio = 0.75
        gen = torch.Generator().manual_seed(0)

        start, end = generate_temporal_mask(
            num_time_tokens, mask_ratio, generator=gen
        )

        expected_length = round(mask_ratio * num_time_tokens)
        assert end - start == expected_length


class TestBuildTokenMask:
    def test_same_time_indices_masked_on_all_channels(self):
        num_channels = 4
        num_time_tokens = 20
        start, end = 5, 15

        mask = build_token_mask(num_channels, num_time_tokens, start, end)

        assert mask.shape == (num_channels * num_time_tokens,)
        per_channel = mask.reshape(num_channels, num_time_tokens)
        for c in range(num_channels):
            assert per_channel[c, start:end].all()
            assert not per_channel[c, :start].any()
            assert not per_channel[c, end:].any()

    def test_total_masked_count(self):
        num_channels = 3
        num_time_tokens = 50
        start, end = 10, 40

        mask = build_token_mask(num_channels, num_time_tokens, start, end)

        expected_masked = num_channels * (end - start)
        assert mask.sum().item() == expected_masked
