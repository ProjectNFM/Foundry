"""Shared utilities for LaBraM channel resolution and patch extraction."""

from typing import Optional
import warnings

import numpy as np
import torch
from torch_brain.data import Data
from torch_brain.batching import pad2d
import torchaudio.functional as F
from braindecode.models.labram import LABRAM_CHANNEL_ORDER

from foundry.models.signal_preparation import (
    normalize_signal_length,
    resolve_signal_source,
)
from foundry.models.embeddings.patching import patch_signal

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
    print("LABRAM_CHANNEL_ORDER", LABRAM_CHANNEL_ORDER)
    exit()
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


def prepare_labram_continuous_signal(
    data: Data,
    num_channels: int,
    num_samples: int,
    target_sampling_rate: int = 200,
) -> tuple[np.ndarray, list[str]]:
    """Prepare continuous LaBraM signal: resolve, resample, filter, normalize length.

    Shared utility for both tokenization and patch extraction. Returns a continuous
    (T, C) signal at the target sampling rate, with length normalized and channels
    filtered to LaBraM canonical order.

    Args:
        data: torch_brain Data object with eeg/ecog/seeg signal.
        num_channels: Expected number of channels after filtering.
        num_samples: Total samples at target_sampling_rate.
        target_sampling_rate: Target rate (default: 200 Hz).

    Returns:
        Tuple of (signal, channel_names) where:
        - signal: (T, C) float32 array at target_sampling_rate, length-normalized
        - channel_names: List of LaBraM channel names (canonical order)
    """
    signal_source, default_type, sampling_rate = resolve_signal_source(data)

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

    # Resample if needed
    if sampling_rate != target_sampling_rate:
        signal_tensor = torch.from_numpy(signal.T).unsqueeze(0)
        signal_tensor = F.resample(
            signal_tensor,
            orig_freq=int(sampling_rate),
            new_freq=target_sampling_rate,
        )
        signal = signal_tensor.squeeze(0).T.numpy()

    # Sanitize non-finite values
    signal = np.where(~np.isfinite(signal), 0.0, signal)

    # Map channels to LaBraM canonical order
    channel_ids = data.channels.id[modality_mask].astype(str)
    keep_indices, matching_channels = resolve_labram_channels(channel_ids)
    signal = signal[:, keep_indices]

    # Warn if channel count mismatch
    if signal.shape[1] != num_channels:
        warnings.warn(
            f"Expected {num_channels} channels after LaBraM filtering, "
            f"got {signal.shape[1]} ({matching_channels}).",
            UserWarning,
        )

    # Normalize length to exact sample count
    sequence_length = num_samples / target_sampling_rate
    signal = normalize_signal_length(signal, target_sampling_rate, sequence_length)

    return signal, matching_channels


def extract_labram_patches(
    data: Data,
    num_channels: int,
    num_samples: int,
    target_sampling_rate: int = 200,
) -> tuple[torch.Tensor, list[str]]:
    """Extract and patch EEG for LaBraM pre-training.

    Converts raw torch_brain Data into (C, N, patch_size) patch tensors.
    Used by VQNSPModel and LaBraMForMaskedEEGModeling.

    Args:
        data: torch_brain Data object with eeg/ecog/seeg signal.
        num_channels: Expected number of channels after filtering.
        num_samples: Total samples at target_sampling_rate (must be divisible by patch_size for 1s patches).
        target_sampling_rate: Target rate (default: 200 Hz, LaBraM standard).

    Returns:
        Tuple of (input_patches, channel_names) where:
        - input_patches: Tensor of shape (C, N, 200) with N patches of 200 samples each
        - channel_names: List of channel names in LABRAM_CHANNEL_ORDER
    """
    # Prepare continuous signal
    signal, channel_names = prepare_labram_continuous_signal(
        data, num_channels, num_samples, target_sampling_rate
    )

    # Unfold into patches: 1-second patches at 200 Hz = 200 samples, no stride
    patch_duration = 1.0  # seconds
    stride = 1.0  # seconds (non-overlapping)
    signal_tensor = torch.from_numpy(signal.T).unsqueeze(0)  # (1, C, T)

    # patch_signal returns (B, N_patches, C, patch_samples)
    patches = patch_signal(
        signal_tensor,
        patch_duration=patch_duration,
        stride=stride,
        sampling_rate=target_sampling_rate,
    )  # (1, N_patches, C, 200)

    # Rearrange to (C, N_patches, 200) and remove batch dim
    output_patches = patches.squeeze(0).permute(1, 0, 2)  # (C, N_patches, 200)

    return output_patches, channel_names


