"""Shared utilities for LaBraM channel resolution and patch extraction."""

from typing import Optional
import warnings

import numpy as np
import torch
from torch_brain.data import Data
from torch_brain.batching import pad2d
import torchaudio.functional as F
from braindecode.models.labram import LABRAM_CHANNEL_ORDER

_LABRAM_UPPER = {ch.upper() for ch in LABRAM_CHANNEL_ORDER}
_LABRAM_UPPER_LIST = [ch.upper() for ch in LABRAM_CHANNEL_ORDER]


def to_labram_channel_name(channel_id: str) -> Optional[str]:
    """Map a dataset channel ID to a name in ``LABRAM_CHANNEL_ORDER``.

    Handles common Sleep-EDF / BrainVision style labels such as
    ``EEG Fpz-Cz`` or session-prefixed IDs by stripping prefixes and, for
    bipolar montages, taking the first electrode that appears in the
    canonical LaBraM order.

    Args:
        channel_id: Raw channel identifier from the dataset.

    Returns:
        Uppercase LaBraM channel name, or ``None`` if no match is found.
    """
    name = str(channel_id).upper().strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1].strip()
    if name.startswith("EEG"):
        name = name[3:].lstrip(" :-_")

    if name in _LABRAM_UPPER:
        return name

    if "-" in name:
        for part in name.replace(" ", "").split("-"):
            if part in _LABRAM_UPPER:
                return part

    return None


def resolve_labram_channels(
    channel_ids: np.ndarray | list[str],
) -> tuple[list[int], list[str]]:
    """Resolve dataset channel IDs to LaBraM-compatible names and keep indices.

    Returns channels in ``LABRAM_CHANNEL_ORDER`` (subset order), dropping any
    that cannot be mapped.

    Args:
        channel_ids: Channel IDs for the active modality mask.

    Returns:
        Tuple of (keep_indices, labram_names) where ``keep_indices`` indexes
        into ``channel_ids`` / the corresponding signal columns.
    """
    mapped: list[tuple[int, str]] = []
    for i, cid in enumerate(channel_ids):
        labram_name = to_labram_channel_name(str(cid))
        if labram_name is not None:
            mapped.append((i, labram_name))

    # Stable order matching LABRAM_CHANNEL_ORDER
    mapped.sort(key=lambda item: _LABRAM_UPPER_LIST.index(item[1]))

    if len(mapped) < 3:
        warnings.warn(
            f"Only {len(mapped)} channels matched LaBraM's canonical order. "
            "Model may not perform well with so few channels. "
            f"Matched channels: {[name for _, name in mapped]}",
            UserWarning,
        )

    if not mapped:
        raise ValueError(
            "No channels could be mapped to LABRAM_CHANNEL_ORDER. "
            f"Got channel IDs: {list(channel_ids)}"
        )

    keep_indices = [i for i, _ in mapped]
    labram_names = [name for _, name in mapped]
    return keep_indices, labram_names


def labram_names_to_index_tensor(ch_names: list[str]) -> torch.Tensor:
    """Encode LaBraM channel names as a long tensor of canonical indices."""
    return torch.tensor(
        [_LABRAM_UPPER_LIST.index(ch.upper()) for ch in ch_names],
        dtype=torch.long,
    )


def labram_index_tensor_to_names(channel_index: torch.Tensor) -> list[str]:
    """Decode a collated ``[B, C]`` (or ``[C]``) index tensor to channel names.

    Assumes a homogeneous batch and uses the first sample when batched.
    """
    if channel_index.ndim == 2:
        indices = channel_index[0].tolist()
    else:
        indices = channel_index.tolist()
    return [_LABRAM_UPPER_LIST[int(i)] for i in indices]


def extract_labram_patches(
    data: Data,
    num_channels: int,
    num_samples: int,
    target_sampling_rate: int = 200,
) -> tuple[torch.Tensor, list[str]]:
    """Extract, resample, and segment EEG into LaBraM patches.

    Converts raw torch_brain Data into patch tensors suitable for LaBraM pre-training.
    This shared utility is used by both VQNSPModel and LaBraMForMaskedEEGModeling
    for consistent preprocessing.

    Args:
        data: torch_brain Data object with eeg/ecog/seeg signal.
        num_channels: Expected number of channels after filtering.
        num_samples: Expected total samples at target_sampling_rate.
        target_sampling_rate: Target rate for resampling (default: 200 Hz, LaBraM standard).

    Returns:
        Tuple of (input_patches, channel_names) where:
        - input_patches: Padded tensor of shape [T, C] (after pad2d)
        - channel_names: List of channel names in LABRAM_CHANNEL_ORDER
    """
    signal_source = None
    default_type = None
    sampling_rate = None

    for modality in ["eeg", "ecog", "seeg"]:
        signal = getattr(data, modality, None)
        if signal is not None:
            signal_source = signal
            default_type = modality.upper()
            if (
                hasattr(signal, "sampling_rate")
                and signal.sampling_rate is not None
            ):
                sampling_rate = float(signal.sampling_rate)
            else:
                sampling_rate = _infer_sampling_rate_from_timestamps(
                    signal.timestamps
                )
            break

    if signal_source is None:
        raise ValueError("Data must have an 'eeg', 'ecog', or 'seeg' field")

    modality_field = (
        data.channels.type.astype(str)
        if hasattr(data.channels, "type")
        else np.array([default_type] * len(data.channels)).astype(str)
    )
    modality_mask = np.isin(
        np.char.lower(modality_field), ["eeg", "ecog", "seeg", "ieeg"]
    )

    signal = signal_source.signal[:, modality_mask]
    signal = np.asarray(signal, dtype=np.float32)

    if sampling_rate != target_sampling_rate:
        signal_tensor = torch.from_numpy(signal.T).unsqueeze(0)
        signal_tensor = F.resample(
            signal_tensor,
            orig_freq=int(sampling_rate),
            new_freq=target_sampling_rate,
        )
        signal = signal_tensor.squeeze(0).T.numpy()

    signal = np.where(~np.isfinite(signal), 0.0, signal)

    channel_ids = data.channels.id[modality_mask].astype(str)
    keep_indices, matching_channels = resolve_labram_channels(channel_ids)
    signal = signal[:, keep_indices]

    if signal.shape[1] != num_channels:
        warnings.warn(
            f"Expected {num_channels} channels after LaBraM filtering, "
            f"got {signal.shape[1]} ({matching_channels}).",
            UserWarning,
        )

    x = torch.from_numpy(signal)
    return pad2d(x), matching_channels


def _infer_sampling_rate_from_timestamps(
    timestamps: np.ndarray,
) -> float:
    """Infer sampling rate from timestamp deltas.

    Args:
        timestamps: Timestamp array.

    Returns:
        Estimated sampling rate in Hz.
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
