import numpy as np


def patch_time_series(
    signal: np.ndarray,
    timestamps: np.ndarray,
    patch_duration: float,
    stride: float | None = None,
    timestamp_mode: str = "middle",
) -> tuple[np.ndarray, np.ndarray]:
    """Patch a 2D time series into fixed-duration windows.

    Args:
        signal: Time series with shape (num_samples, num_channels).
        timestamps: Timestamps for each sample with shape (num_samples,).
        patch_duration: Duration of each patch in seconds.
        stride: Step size between patches in seconds. Defaults to patch_duration.
        timestamp_mode: Patch timestamp convention:
            - "start": timestamp at each patch start
            - "middle": timestamp at each patch center

    Returns:
        Tuple of:
            - patched_signal with shape (num_patches, num_channels, patch_samples)
            - patch_timestamps with shape (num_patches,)
    """
    if signal.ndim != 2:
        raise ValueError(
            f"signal must be 2D with shape (time, channels), got {signal.shape}"
        )
    if timestamps.ndim != 1:
        raise ValueError(
            f"timestamps must be 1D with shape (time,), got {timestamps.shape}"
        )
    if len(signal) != len(timestamps):
        raise ValueError(
            "signal and timestamps must have the same number of samples"
        )
    if len(timestamps) < 2:
        raise ValueError(
            "at least 2 timestamps are required to infer sampling rate"
        )
    if patch_duration <= 0:
        raise ValueError("patch_duration must be > 0")
    if timestamp_mode not in {"start", "middle"}:
        raise ValueError(
            f"timestamp_mode must be 'start' or 'middle', got '{timestamp_mode}'"
        )

    stride_seconds = patch_duration if stride is None else stride
    if stride_seconds <= 0:
        raise ValueError("stride must be > 0")

    sample_deltas = np.diff(timestamps)
    if np.any(sample_deltas <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.allclose(sample_deltas, sample_deltas[0], atol=1e-6):
        raise ValueError(
            "timestamps must be regularly spaced to create fixed-sample patches"
        )

    sampling_rate = 1.0 / float(sample_deltas[0])
    patch_samples = int(np.round(patch_duration * sampling_rate))
    stride_samples = int(np.round(stride_seconds * sampling_rate))
    patch_samples = max(patch_samples, 1)
    stride_samples = max(stride_samples, 1)

    num_samples = signal.shape[0]
    if num_samples <= patch_samples:
        num_patches = 1
    else:
        num_patches = (
            int(np.ceil((num_samples - patch_samples) / stride_samples)) + 1
        )

    total_samples_needed = (num_patches - 1) * stride_samples + patch_samples
    pad_amount = max(total_samples_needed - num_samples, 0)
    padded_signal = (
        np.pad(signal, ((0, pad_amount), (0, 0)), mode="constant")
        if pad_amount > 0
        else signal
    )

    patch_indices = (
        np.arange(patch_samples)[None, :]
        + stride_samples * np.arange(num_patches)[:, None]
    )
    patched_signal = np.moveaxis(padded_signal[patch_indices], 2, 1)

    start_time = float(timestamps[0])
    patch_offsets = stride_seconds * np.arange(num_patches, dtype=np.float64)
    if timestamp_mode == "start":
        patch_timestamps = start_time + patch_offsets
    else:
        patch_timestamps = start_time + patch_offsets + patch_duration / 2

    return patched_signal, patch_timestamps
