import torch


def _validate_positive_finite(value: float, *, name: str) -> None:
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def compute_patching_layout(
    sequence_length_samples: int,
    patch_duration: float,
    stride: float,
    sampling_rate: float,
) -> tuple[int, int, int, int]:
    """Compute sample-domain patching geometry.

    Returns:
        (num_patches, patch_samples, stride_samples, pad_right_samples)
    """
    if sequence_length_samples <= 0:
        raise ValueError(
            "sequence_length_samples must be > 0, "
            f"got {sequence_length_samples}."
        )

    _validate_positive_finite(patch_duration, name="patch_duration")
    _validate_positive_finite(stride, name="stride")
    _validate_positive_finite(sampling_rate, name="sampling_rate")

    patch_samples = max(1, round(patch_duration * sampling_rate))
    stride_samples = max(1, round(stride * sampling_rate))

    if sequence_length_samples <= patch_samples:
        num_patches = 1
    else:
        numer = sequence_length_samples - patch_samples
        num_patches = (numer + stride_samples - 1) // stride_samples + 1

    covered_samples = (num_patches - 1) * stride_samples + patch_samples
    pad_right_samples = max(0, covered_samples - sequence_length_samples)

    return num_patches, patch_samples, stride_samples, pad_right_samples


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
    if signal.ndim != 3:
        raise ValueError(
            f"signal must have shape (B, C, T), got shape {tuple(signal.shape)}."
        )
    sequence_length = signal.shape[2]
    (
        _num_patches,
        patch_samples,
        stride_samples,
        pad_right_samples,
    ) = compute_patching_layout(
        sequence_length_samples=sequence_length,
        patch_duration=patch_duration,
        stride=stride,
        sampling_rate=sampling_rate,
    )
    if pad_right_samples:
        signal = torch.nn.functional.pad(signal, (0, pad_right_samples))

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


__all__ = [
    "compute_patching_layout",
    "patch_signal",
    "compute_patch_timestamps",
]
