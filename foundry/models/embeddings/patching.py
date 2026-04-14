import torch


def patch_signal(
    signal: torch.Tensor,
    patch_duration: float,
    stride: float,
    sampling_rate: float,
) -> torch.Tensor:
    """Patch a batched signal on GPU using ``torch.unfold``.

    Only valid when all items in the batch share the same sampling rate.

    Args:
        signal: (B, C, T) raw time series.
        patch_duration: Patch width in seconds.
        stride: Stride between patches in seconds.
        sampling_rate: Shared sampling rate (Hz) for the batch.

    Returns:
        (B, num_patches, C, patch_samples)
    """
    patch_samples = max(1, round(patch_duration * sampling_rate))
    stride_samples = max(1, round(stride * sampling_rate))

    # unfold along time dim -> (B, C, num_patches, patch_samples)
    patches = signal.unfold(
        dimension=2, size=patch_samples, step=stride_samples
    )
    # -> (B, num_patches, C, patch_samples)
    return patches.permute(0, 2, 1, 3)


def compute_patch_timestamps(
    start_time: float,
    num_patches: int,
    patch_duration: float,
    stride: float,
) -> torch.Tensor:
    """Compute center timestamps for each patch.

    Args:
        start_time: Start time of the signal in seconds.
        num_patches: Number of patches.
        patch_duration: Duration of each patch in seconds.
        stride: Stride between patches in seconds.

    Returns:
        (num_patches,) float tensor of patch center times.
    """
    offsets = torch.arange(num_patches, dtype=torch.float32) * stride
    return offsets + start_time + patch_duration / 2.0


__all__ = ["patch_signal", "compute_patch_timestamps"]
