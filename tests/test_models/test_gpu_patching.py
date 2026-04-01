import numpy as np
import torch

from foundry.data.transforms import patch_time_series
from foundry.models.embeddings.patching import (
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

    def test_equivalence_with_numpy(self):
        """GPU patching matches numpy patch_time_series for evenly-divisible cases."""
        T, C = 250, 4
        fs = 250.0
        duration = T / fs

        signal_np = np.random.randn(T, C).astype(np.float32)
        timestamps = np.linspace(0, duration, T, endpoint=False)

        patches_np, _ = patch_time_series(
            signal=signal_np,
            timestamps=timestamps,
            patch_duration=0.1,
            stride=0.1,
        )

        signal_torch = torch.from_numpy(signal_np.T).unsqueeze(0)  # (1, C, T)
        patches_torch = patch_signal(
            signal_torch, patch_duration=0.1, stride=0.1, sampling_rate=fs
        )

        np.testing.assert_allclose(
            patches_torch[0].numpy(),
            patches_np,
            atol=1e-6,
        )

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
