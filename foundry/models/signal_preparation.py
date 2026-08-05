"""Canonical signal-preparation and token-grid contract.

This module owns the single implementation of:
- Signal-length normalization (trim/pad to match sampling_rate × sequence_length)
- Patch-count calculation (for any stride/patch-size combination)
- The immutable PreparedSignal contract consumed by both encoder inputs
  and reconstruction targets.
- Signal source resolution and sampling rate inference (shared across models)

Normalization stages are explicitly named and independently configurable:
- ``normalize_encoder_inputs``: optional per-channel z-scoring before embedding,
  ensuring scale invariance across datasets with different amplifier gains.
- ``normalize_reconstruction_targets``: objective-level z-scoring applied to
  the raw (sanitized, length-normalized) signal so the reconstruction loss
  operates on a standardized scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreparedSignal:
    """Immutable contract produced by signal preparation.

    Contains the modality-filtered, sanitized, length-normalized signal
    along with all token-grid metadata needed by both the tokenizer and
    reconstruction-target generation.

    Attributes:
        signal: (T_norm, C_filtered) length-normalized signal.
            Non-finite values have been replaced with zero.
        sampling_rate: Sampling rate in Hz.
        num_samples: Length of the normalized signal (T_norm).
        original_num_samples: Length before normalization.
        num_channels: Number of modality-filtered channels (C_filtered).
        modality_mask: Boolean mask over original channels indicating
            which were kept (for mapping back to channel metadata).
    """

    signal: np.ndarray
    sampling_rate: float
    num_samples: int
    original_num_samples: int
    num_channels: int
    modality_mask: np.ndarray


def infer_sampling_rate_from_timestamps(
    timestamps: np.ndarray,
) -> float:
    """Infer sampling rate from timestamp deltas (shared across models).

    Args:
        timestamps: Timestamp array with irregular or regular spacing.

    Returns:
        Estimated sampling rate in Hz (median of valid deltas).

    Raises:
        ValueError: If no valid timestamp deltas can be computed.
    """
    sample_deltas = np.diff(timestamps).astype(np.float64)
    valid_deltas = sample_deltas[
        np.isfinite(sample_deltas) & (sample_deltas > 0)
    ]
    if valid_deltas.size == 0:
        raise ValueError(
            "Could not infer a valid sampling rate from timestamps."
        )
    return 1.0 / float(np.median(valid_deltas))


def resolve_signal_source(
    data: Any,
) -> tuple[Any, str, float]:
    """Find the signal source, default modality type, and sampling rate.

    Searches for the first available signal source (eeg, ecog, seeg) and
    infers its sampling rate. This is shared across all models to ensure
    consistent signal resolution.

    Args:
        data: torch_brain Data object with optional eeg/ecog/seeg signals.

    Returns:
        Tuple of (signal_source, default_type, sampling_rate) where:
        - signal_source: The time series object (e.g., data.eeg)
        - default_type: Uppercase modality name ('EEG', 'ECOG', or 'SEEG')
        - sampling_rate: Inferred or stored sampling rate in Hz

    Raises:
        ValueError: If no signal source is found.
    """
    for modality in ["eeg", "ecog", "seeg"]:
        signal = getattr(data, modality, None)
        if signal is not None:
            if (
                hasattr(signal, "sampling_rate")
                and signal.sampling_rate is not None
            ):
                sampling_rate = float(signal.sampling_rate)
            else:
                sampling_rate = infer_sampling_rate_from_timestamps(
                    signal.timestamps
                )
            return signal, modality.upper(), sampling_rate

    raise ValueError("Data must have an 'eeg', 'ecog', or 'seeg' field")


def normalize_signal_length(
    signal: np.ndarray,
    sampling_rate: float,
    sequence_length: float,
) -> np.ndarray:
    """Trim or pad signal to exactly ``round(sampling_rate × sequence_length)`` samples.

    This is the single canonical implementation of length normalization.
    Both encoder inputs and reconstruction targets must use this function
    so their token grids are guaranteed to align.

    Args:
        signal: (T, ...) array where the first axis is time.
        sampling_rate: Sampling rate in Hz.
        sequence_length: Target duration in seconds.

    Returns:
        Array with T = ``round(sampling_rate × sequence_length)``.
    """
    expected_T = round(sampling_rate * sequence_length)
    T = signal.shape[0]
    if T > expected_T:
        signal = signal[:expected_T]
    elif T < expected_T:
        pad_width = [(0, expected_T - T)] + [(0, 0)] * (signal.ndim - 1)
        signal = np.pad(signal, pad_width)
    return signal


def compute_num_patches(
    num_samples: int,
    patch_samples: int,
    stride_samples: int,
) -> int:
    """Compute the number of patches produced by ``torch.unfold`` semantics.

    This is the single canonical implementation used by timestamp generation,
    target generation, patch embeddings, and tests.

    Args:
        num_samples: Total number of time samples (T).
        patch_samples: Number of samples per patch.
        stride_samples: Number of samples between patch starts.

    Returns:
        Number of patches.  At least 1 when num_samples > 0.
    """
    if num_samples <= 0:
        return 0
    if num_samples < patch_samples:
        return 1
    return (num_samples - patch_samples) // stride_samples + 1


def normalize_encoder_inputs(
    signal: np.ndarray,
) -> np.ndarray:
    """Per-channel z-scoring for encoder input scale invariance.

    Applied before embedding to ensure the model sees standardized
    amplitudes regardless of amplifier gain or physical units.
    This is independent of reconstruction-target normalization.

    Constant channels (std < 1e-8) are left at zero.

    Args:
        signal: (T, C) time-series array.

    Returns:
        Z-scored signal with same shape.
    """
    mu = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (signal - mu) / std


def normalize_reconstruction_targets(
    signal: np.ndarray,
    max_channels: int,
) -> np.ndarray:
    """Per-channel z-scoring for reconstruction targets.

    Applied to raw sanitized signal so the reconstruction loss operates
    on a standardized scale.  Padded channels remain at zero and never
    contribute to mean/std calculations.

    Args:
        signal: (T, C_actual) raw signal (sanitized, length-normalized).
        max_channels: Padded channel count (C_pad).

    Returns:
        (C_pad, T) z-scored target array with padded channels at zero.
    """
    T, C_actual = signal.shape
    C_pad = max_channels
    C_used = min(C_actual, C_pad)

    values = signal[:, :C_used].T.astype(np.float32)  # (C_used, T)
    mu = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    normalized = (values - mu) / std

    targets = np.zeros((C_pad, T), dtype=np.float32)
    targets[:C_used] = normalized
    return targets


__all__ = [
    "PreparedSignal",
    "infer_sampling_rate_from_timestamps",
    "resolve_signal_source",
    "normalize_signal_length",
    "compute_num_patches",
    "normalize_encoder_inputs",
    "normalize_reconstruction_targets",
]
