import pytest
import torch

from foundry.models.embeddings.patching import (
    compute_patching_layout,
    compute_patch_timestamps,
    patch_signal,
)


class TestPatchSignal:
    def test_basic_shape(self):
        B, C, T = 2, 4, 200
        signal = torch.randn(B, C, T)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=250.0
        )
        assert patches.shape == (B, 8, C, 25)

    def test_single_patch(self):
        B, C, T = 1, 2, 10
        signal = torch.randn(B, C, T)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=100.0
        )
        assert patches.shape == (1, 1, 2, 10)
        torch.testing.assert_close(patches[0, 0], signal[0])

    def test_overlapping_patches(self):
        B, C, T = 2, 4, 200
        signal = torch.randn(B, C, T)
        patches = patch_signal(
            signal, patch_duration=0.2, stride=0.1, sampling_rate=250.0
        )
        # patch_samples = 50, stride_samples = 25, num_patches = (200-50)//25+1 = 7
        assert patches.shape == (B, 7, C, 50)

    def test_device_placement(self):
        if not torch.cuda.is_available():
            return
        signal = torch.randn(2, 4, 200, device="cuda")
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=250.0
        )
        assert patches.device.type == "cuda"

    def test_gradient_flow(self):
        signal = torch.randn(2, 4, 200, requires_grad=True)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=250.0
        )
        patches.sum().backward()
        assert signal.grad is not None

    def test_non_overlapping_full_coverage(self):
        """Non-overlapping patches tile the time dimension exactly."""
        B, C, T = 1, 2, 100
        signal = torch.arange(T, dtype=torch.float32).expand(B, C, T)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=100.0
        )
        # 10 samples per patch, 10 patches
        assert patches.shape == (1, 10, 2, 10)
        for p in range(10):
            expected = torch.arange(p * 10, (p + 1) * 10, dtype=torch.float32)
            torch.testing.assert_close(patches[0, p, 0], expected)

    def test_rounding_keeps_expected_patch_count_with_padding(self):
        """2.0s / 0.1s should yield 20 patches even after sample rounding."""
        signal = torch.arange(512, dtype=torch.float32).reshape(1, 1, 512)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=256.0
        )

        assert patches.shape == (1, 20, 1, 26)
        torch.testing.assert_close(
            patches[0, -1, 0, :18], torch.arange(494, 512, dtype=torch.float32)
        )
        torch.testing.assert_close(patches[0, -1, 0, 18:], torch.zeros(8))

    def test_short_sequence_is_right_padded_to_patch_width(self):
        signal = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        patches = patch_signal(
            signal, patch_duration=0.1, stride=0.1, sampling_rate=100.0
        )

        assert patches.shape == (1, 1, 1, 10)
        torch.testing.assert_close(
            patches[0, 0, 0],
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 0], dtype=torch.float32),
        )

    def test_invalid_arguments_raise(self):
        signal = torch.randn(1, 1, 16)
        with pytest.raises(ValueError, match="patch_duration must be > 0"):
            patch_signal(
                signal,
                patch_duration=0.0,
                stride=0.1,
                sampling_rate=100.0,
            )
        with pytest.raises(ValueError, match="stride must be > 0"):
            patch_signal(
                signal,
                patch_duration=0.1,
                stride=0.0,
                sampling_rate=100.0,
            )
        with pytest.raises(ValueError, match="sampling_rate must be > 0"):
            patch_signal(
                signal,
                patch_duration=0.1,
                stride=0.1,
                sampling_rate=0.0,
            )
        with pytest.raises(ValueError, match="shape \\(B, C, T\\)"):
            patch_signal(
                signal.squeeze(0),
                patch_duration=0.1,
                stride=0.1,
                sampling_rate=100.0,
            )


class TestComputePatchingLayout:
    def test_tail_padding_layout(self):
        num_patches, patch_samples, stride_samples, pad_right = (
            compute_patching_layout(
                sequence_length_samples=512,
                patch_duration=0.1,
                stride=0.1,
                sampling_rate=256.0,
            )
        )
        assert num_patches == 20
        assert patch_samples == 26
        assert stride_samples == 26
        assert pad_right == 8

    def test_exact_tiling_has_no_padding(self):
        num_patches, patch_samples, stride_samples, pad_right = (
            compute_patching_layout(
                sequence_length_samples=100,
                patch_duration=0.1,
                stride=0.1,
                sampling_rate=100.0,
            )
        )
        assert num_patches == 10
        assert patch_samples == 10
        assert stride_samples == 10
        assert pad_right == 0

    def test_overlapping_layout(self):
        num_patches, patch_samples, stride_samples, pad_right = (
            compute_patching_layout(
                sequence_length_samples=200,
                patch_duration=0.2,
                stride=0.1,
                sampling_rate=250.0,
            )
        )
        assert num_patches == 7
        assert patch_samples == 50
        assert stride_samples == 25
        assert pad_right == 0


class TestComputePatchTimestamps:
    def test_basic(self):
        ts = compute_patch_timestamps(
            0.0, num_patches=4, patch_duration=0.1, stride=0.1
        )
        expected = torch.tensor([0.05, 0.15, 0.25, 0.35])
        torch.testing.assert_close(ts, expected)

    def test_with_start_offset(self):
        ts = compute_patch_timestamps(
            1.0, num_patches=3, patch_duration=0.2, stride=0.2
        )
        expected = torch.tensor([1.1, 1.3, 1.5])
        torch.testing.assert_close(ts, expected)

    def test_single_patch(self):
        ts = compute_patch_timestamps(
            0.0, num_patches=1, patch_duration=0.5, stride=0.5
        )
        expected = torch.tensor([0.25])
        torch.testing.assert_close(ts, expected)

    def test_overlapping_stride(self):
        ts = compute_patch_timestamps(
            0.0, num_patches=5, patch_duration=0.2, stride=0.05
        )
        assert ts.shape == (5,)
        assert abs(ts[0].item() - 0.1) < 1e-6
        assert abs(ts[1].item() - 0.15) < 1e-6
