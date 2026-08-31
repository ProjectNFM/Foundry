"""Utilities for introspecting datasets to derive model configuration values.

These functions bridge the gap between data and model configuration by
computing values like channel counts, sampling rates, and patch sizes
directly from a dataset, removing the need to hard-code them in YAML configs.

Typical usage::

    from foundry.data.utils import (
        get_sampling_rate,
        get_max_channels,
        get_session_configs,
        compute_patch_samples,
    )

    dataset = MyDataset(root="./data/processed/")
    sr = get_sampling_rate(dataset)
    num_channels = get_max_channels(dataset)
    patch_samples = compute_patch_samples(patch_duration=0.1, sampling_rate=sr)

    # For SpatialProjectionStrategy:
    session_configs = get_session_configs(dataset)
"""

from __future__ import annotations

import numpy as np

NEURAL_MODALITIES = frozenset({"eeg", "ecog", "seeg", "ieeg"})


def compute_patch_samples(patch_duration: float, sampling_rate: float) -> int:
    """Compute the number of time samples per patch.

    Matches the rounding logic used by :class:`~foundry.models.tokenizer.EEGTokenizer`
    and :func:`~foundry.models.embeddings.patching.patch_signal`.

    Args:
        patch_duration: Duration of each patch in seconds.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Number of samples per patch (minimum 1).
    """
    return max(1, round(patch_duration * sampling_rate))


def _resolve_signal_modality(data) -> str:
    """Auto-detect which neural signal modality is present in a recording."""
    for modality in ("eeg", "ecog", "seeg"):
        if getattr(data, modality, None) is not None:
            return modality
    raise ValueError("Recording has no 'eeg', 'ecog', or 'seeg' field")


def _count_modality_channels(data) -> int:
    """Count channels belonging to neural modalities in a single recording.

    If the recording's ``channels`` object has a ``type`` attribute, only
    channels whose type (case-insensitive) is in :data:`NEURAL_MODALITIES`
    are counted.  Otherwise all channels are counted.
    """
    if hasattr(data.channels, "type"):
        types = np.char.lower(data.channels.type.astype(str))
        return int(np.isin(types, list(NEURAL_MODALITIES)).sum())
    return len(data.channels.id)


def get_sampling_rate(dataset) -> float:
    """Infer sampling rate from the first recording in the dataset.

    Reads the neural signal timestamps and computes the rate from the first
    sample delta, matching the logic in
    :meth:`~foundry.models.poyo_eeg.POYOEEGModel.tokenize`.

    Args:
        dataset: A :class:`torch_brain.datasets.Dataset` instance with at
            least one recording containing an ``eeg``, ``ecog``, or ``seeg``
            field.

    Returns:
        Sampling rate in Hz.

    Raises:
        ValueError: If no neural signal field is found.
    """
    rid = dataset.recording_ids[0]
    data = dataset.get_recording(rid)
    modality = _resolve_signal_modality(data)
    signal_source = getattr(data, modality)
    deltas = np.diff(signal_source.timestamps)
    return 1.0 / float(deltas[0])


def get_channel_counts(dataset) -> dict[str, int]:
    """Get per-session channel counts for all recordings.

    Only channels belonging to neural modalities (EEG, ECoG, sEEG, iEEG)
    are counted.  If multiple recordings share a ``session.id``, the
    maximum channel count across those recordings is kept.

    Args:
        dataset: A :class:`torch_brain.datasets.Dataset` instance.

    Returns:
        Mapping of ``session.id`` strings to channel counts.
    """
    counts: dict[str, int] = {}
    for rid in dataset.recording_ids:
        data = dataset.get_recording(rid)
        session_id = str(data.session.id)
        n = _count_modality_channels(data)
        counts[session_id] = max(counts.get(session_id, 0), n)
    return counts


def get_max_channels(dataset) -> int:
    """Maximum channel count across all sessions in the dataset.

    Useful for configuring :class:`~foundry.models.embeddings.FixedChannelStrategy`,
    :class:`~foundry.models.embeddings.PerChannelStrategy`, or
    :class:`~foundry.models.embeddings.SpatialProjectionStrategy` where
    signals are padded to a common size.

    Args:
        dataset: A :class:`torch_brain.datasets.Dataset` instance.

    Returns:
        Maximum number of neural channels in any session.
    """
    return max(get_channel_counts(dataset).values())


def get_min_channels(dataset) -> int:
    """Minimum channel count across all sessions in the dataset.

    Args:
        dataset: A :class:`torch_brain.datasets.Dataset` instance.

    Returns:
        Minimum number of neural channels in any session.
    """
    return min(get_channel_counts(dataset).values())


def resolve_neural_signal(
    data,
    supported_modalities: frozenset[str] = NEURAL_MODALITIES,
) -> tuple[str, object, np.ndarray, np.ndarray]:
    """Resolve the first present neural signal field and supported channel mask.

    Matches the selection logic in ``NeurosoftConvBiGRU.tokenize``: pick the
    first present field from ``(eeg, ecog, seeg, ieeg)`` and filter channels
    whose type (case-insensitive) is in *supported_modalities*.

    Args:
        data: A ``torch_brain.data.Data`` object (recording or window).
        supported_modalities: Lowercase modality strings to keep.

    Returns:
        ``(field_name, signal_source, keep_mask, channel_names)`` where
        *keep_mask* is a boolean array over all channels and *channel_names*
        are the IDs of the kept channels in their original order.

    Raises:
        ValueError: If no supported neural signal field is present.
    """
    signal_source = None
    field_name = None
    default_type = None
    for name, modality in (
        ("eeg", "EEG"),
        ("ecog", "ECOG"),
        ("seeg", "SEEG"),
        ("ieeg", "IEEG"),
    ):
        if hasattr(data, name) and getattr(data, name) is not None:
            signal_source = getattr(data, name)
            field_name = name
            default_type = modality
            break

    if signal_source is None:
        raise ValueError(
            "Data must contain an EEG, ECoG, sEEG, or iEEG signal field"
        )

    channel_types = (
        data.channels.type.astype(str)
        if hasattr(data.channels, "type")
        else np.array([default_type] * len(data.channels.id), dtype=str)
    )
    normalized_modalities = frozenset(
        str(modality).lower() for modality in supported_modalities
    )
    if not normalized_modalities <= NEURAL_MODALITIES:
        raise ValueError(
            "supported_modalities must be drawn from "
            f"{sorted(NEURAL_MODALITIES)}"
        )
    keep = np.isin(np.char.lower(channel_types), sorted(normalized_modalities))
    channel_names = np.asarray(data.channels.id)[keep]

    return field_name, signal_source, keep, channel_names


def get_session_configs(dataset) -> dict[str, int]:
    """Build the ``session_configs`` mapping for :class:`SpatialProjectionStrategy`.

    Returns a ``{session_id: num_channels}`` dictionary suitable for passing
    directly to
    :class:`~foundry.models.embeddings.SpatialProjectionStrategy`::

        from foundry.data.utils import get_session_configs

        session_configs = get_session_configs(dataset)
        strategy = SpatialProjectionStrategy(
            num_channels=max(session_configs.values()),
            num_sources=32,
            session_configs=session_configs,
        )

    Args:
        dataset: A :class:`torch_brain.datasets.Dataset` instance.

    Returns:
        Mapping of session ID strings to their channel counts.
    """
    return get_channel_counts(dataset)


__all__ = [
    "NEURAL_MODALITIES",
    "compute_patch_samples",
    "get_sampling_rate",
    "get_channel_counts",
    "get_max_channels",
    "get_min_channels",
    "get_session_configs",
    "resolve_neural_signal",
]
