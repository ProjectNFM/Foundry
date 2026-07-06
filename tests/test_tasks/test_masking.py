"""Tests for masking strategies in foundry.tasks.masking."""

import pytest
import torch

from foundry.tasks.masking import (
    ChannelMasking,
    MaskingStrategy,
    RandomTokenMasking,
    TemporalBlockMasking,
)


class TestMaskingStrategyInterface:
    """Verify the base-class contract is enforced."""

    def test_base_class_raises_not_implemented(self):
        strategy = MaskingStrategy(mask_ratio=0.5)
        channel_mask = torch.ones(2, 4, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            strategy(
                num_channels=4, num_time_tokens=10, channel_mask=channel_mask
            )

    def test_frozen_dataclass_is_immutable(self):
        strategy = RandomTokenMasking(mask_ratio=0.5)
        with pytest.raises(AttributeError):
            strategy.mask_ratio = 0.7


class TestRandomTokenMasking:
    @pytest.fixture
    def strategy(self):
        return RandomTokenMasking(mask_ratio=0.5)

    def test_output_shapes(self, strategy):
        B, C, N = 4, 8, 20
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)
        expected_num_masked = int(0.5 * C * N)

        assert mask_indices.shape == (B, expected_num_masked)
        assert validity_mask.shape == (B, expected_num_masked)

    def test_indices_within_range(self, strategy):
        B, C, N = 3, 6, 15
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()

    def test_no_duplicate_indices_per_sample(self, strategy):
        B, C, N = 2, 4, 10
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)
        for b in range(B):
            unique = torch.unique(mask_indices[b])
            assert unique.shape[0] == mask_indices.shape[1]

    def test_validity_mask_reflects_channel_mask(self):
        strategy = RandomTokenMasking(mask_ratio=0.75)
        B, C, N = 2, 4, 8
        channel_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        for b in range(B):
            for i in range(mask_indices.shape[1]):
                token_idx = mask_indices[b, i].item()
                channel_idx = token_idx // N
                expected_valid = channel_mask[b, channel_idx].item()
                assert validity_mask[b, i].item() == expected_valid

    def test_all_channels_real_yields_all_valid(self, strategy):
        B, C, N = 2, 4, 10
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        _, validity_mask = strategy(C, N, channel_mask)
        assert validity_mask.all()

    def test_minimum_one_masked_token(self):
        strategy = RandomTokenMasking(mask_ratio=0.001)
        B, C, N = 1, 2, 3
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)
        assert mask_indices.shape[1] >= 1

    def test_deterministic_count(self):
        """Mask count is deterministic from (C, N, mask_ratio), not stochastic."""
        strategy = RandomTokenMasking(mask_ratio=0.3)
        B, C, N = 5, 10, 20
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices_1, _ = strategy(C, N, channel_mask)
        mask_indices_2, _ = strategy(C, N, channel_mask)

        assert mask_indices_1.shape == mask_indices_2.shape


class TestTemporalBlockMasking:
    @pytest.fixture
    def strategy(self):
        return TemporalBlockMasking(mask_ratio=0.5, block_size=5)

    def test_output_shapes(self, strategy):
        B, C, N = 3, 6, 20
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)
        # num_blocks = min(int(0.5 * 20) // 5, 20 // 5) = min(2, 4) = 2
        # num_time_masked = 2 * 5 = 10
        # num_masked = 10 * 6 = 60
        assert mask_indices.shape == (B, 60)
        assert validity_mask.shape == (B, 60)

    def test_all_channels_masked_at_same_time_positions(self, strategy):
        B, C, N = 1, 4, 20
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)

        time_positions_per_channel = []
        for c in range(C):
            channel_tokens = mask_indices[0]
            channel_start = c * N
            channel_end = (c + 1) * N
            in_channel = (channel_tokens >= channel_start) & (
                channel_tokens < channel_end
            )
            times = (channel_tokens[in_channel] - channel_start).sort()[0]
            time_positions_per_channel.append(times)

        for c in range(1, C):
            assert torch.equal(
                time_positions_per_channel[0], time_positions_per_channel[c]
            )

    def test_indices_within_range(self, strategy):
        B, C, N = 2, 8, 30
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()

    def test_validity_mask_reflects_channel_mask(self):
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        _, C, N = 1, 4, 20
        channel_mask = torch.tensor([[True, True, False, False]])

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        for i in range(mask_indices.shape[1]):
            token_idx = mask_indices[0, i].item()
            channel_idx = token_idx // N
            expected_valid = channel_mask[0, channel_idx].item()
            assert validity_mask[0, i].item() == expected_valid


class TestTemporalBlockMaskingEdgeCases:
    """Reproduce bug where TemporalBlockMasking crashes when N < block_size."""

    def test_n_less_than_block_size_does_not_crash(self):
        """Bug 4 (Medium): num_slots becomes 0 but num_blocks is forced to 1
        via max(1, ...), causing argsort on a (B, 0) tensor to fail."""
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        B, C, N = 2, 4, 3
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        assert mask_indices.shape[0] == B
        assert validity_mask.shape == mask_indices.shape
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()

    def test_n_equals_one(self):
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        B, C, N = 1, 4, 1
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        assert mask_indices.shape[0] == B
        assert validity_mask.shape == mask_indices.shape
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()

    def test_n_equals_block_size(self):
        """Boundary: exactly one slot, should work normally."""
        strategy = TemporalBlockMasking(mask_ratio=0.5, block_size=5)
        B, C, N = 2, 4, 5
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        assert mask_indices.shape[0] == B
        assert validity_mask.shape == mask_indices.shape
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()


class TestChannelMasking:
    @pytest.fixture
    def strategy(self):
        return ChannelMasking(mask_ratio=0.5)

    def test_output_shapes(self, strategy):
        B, C, N = 3, 8, 10
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, validity_mask = strategy(C, N, channel_mask)
        num_channels_masked = int(0.5 * C)  # 4
        expected_num_masked = num_channels_masked * N  # 40

        assert mask_indices.shape == (B, expected_num_masked)
        assert validity_mask.shape == (B, expected_num_masked)

    def test_entire_channels_masked(self, strategy):
        """All N time positions of each selected channel are masked."""
        B, C, N = 1, 8, 10
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)

        masked_channels = set()
        for idx in mask_indices[0].tolist():
            masked_channels.add(idx // N)

        num_channels_masked = int(0.5 * C)
        assert len(masked_channels) == num_channels_masked

        for ch in masked_channels:
            expected_tokens = set(range(ch * N, (ch + 1) * N))
            actual_tokens = set(
                idx for idx in mask_indices[0].tolist() if idx // N == ch
            )
            assert actual_tokens == expected_tokens

    def test_biased_toward_real_channels(self):
        """Real channels should be preferentially selected over padded ones."""
        strategy = ChannelMasking(mask_ratio=0.25)
        B, C, N = 100, 8, 10
        channel_mask = torch.zeros(B, C, dtype=torch.bool)
        channel_mask[:, :2] = True  # only 2 real channels

        mask_indices, validity_mask = strategy(C, N, channel_mask)

        # With strong bias, most validity_mask entries should be True
        validity_rate = validity_mask.float().mean().item()
        assert validity_rate > 0.8

    def test_indices_within_range(self, strategy):
        B, C, N = 2, 6, 12
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        mask_indices, _ = strategy(C, N, channel_mask)
        assert (mask_indices >= 0).all()
        assert (mask_indices < C * N).all()


class TestComputeVisibleIndices:
    """Test the complement-of-mask utility."""

    def test_visible_plus_masked_equals_total(self):
        from foundry.models.masked_poyo_eeg import _compute_visible_indices

        B, total = 3, 20
        num_masked = 8
        mask_indices = torch.stack(
            [torch.randperm(total)[:num_masked] for _ in range(B)]
        )

        visible = _compute_visible_indices(total, mask_indices)
        assert visible.shape == (B, total - num_masked)

        for b in range(B):
            all_indices = torch.cat([mask_indices[b], visible[b]])
            assert torch.equal(all_indices.sort()[0], torch.arange(total))

    def test_no_overlap_between_visible_and_masked(self):
        from foundry.models.masked_poyo_eeg import _compute_visible_indices

        B, total = 2, 15
        num_masked = 5
        mask_indices = torch.stack(
            [torch.randperm(total)[:num_masked] for _ in range(B)]
        )

        visible = _compute_visible_indices(total, mask_indices)

        for b in range(B):
            mask_set = set(mask_indices[b].tolist())
            visible_set = set(visible[b].tolist())
            assert mask_set.isdisjoint(visible_set)
