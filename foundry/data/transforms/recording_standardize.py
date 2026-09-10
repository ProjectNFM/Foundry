"""Non-mutating per-channel standardization using frozen train-split statistics.

Applied immediately before model tokenization, while the signal is still in
its recording-native time-by-channel layout.  Independent of batch size,
padding, and the Conv--BiGRU adapter.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from torch_brain.data import Data

from foundry.data.normalization import RecordingChannelStats
from foundry.data.utils import NEURAL_MODALITIES, resolve_neural_signal


class RecordingChannelStandardize:
    """Per-channel z-score using frozen recording-level train statistics.

    For each fetched ``Data`` window the transform:

    1. selects the same neural signal field and supported channels as the
       model tokenizer;
    2. looks up frozen statistics by ``data.session.id``;
    3. verifies the expected channel count and order; and
    4. produces a normalized window *without mutating the cached raw
       recording*, unsupported channels, time domain, sampling rate, or
       metadata.

    Args:
        stats_by_recording: Mapping from canonical session/recording ID
            to frozen :class:`RecordingChannelStats`.
        supported_modalities: Lowercase modality strings to normalize.
        scale_floor: Positive lower bound used during fitting (stored for
            provenance; the actual floor is baked into the frozen scales).
    """

    def __init__(
        self,
        stats_by_recording: Mapping[str, RecordingChannelStats],
        supported_modalities: frozenset[str] = NEURAL_MODALITIES,
        scale_floor: float = 1e-8,
    ):
        if not stats_by_recording:
            raise ValueError(
                "stats_by_recording must contain at least one recording"
            )
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be a finite positive value")
        self.stats_by_recording = dict(stats_by_recording)
        self.supported_modalities = frozenset(
            str(modality).lower() for modality in supported_modalities
        )
        if not self.supported_modalities <= NEURAL_MODALITIES:
            raise ValueError(
                "supported_modalities must be drawn from "
                f"{sorted(NEURAL_MODALITIES)}"
            )
        self.scale_floor = scale_floor

    def __call__(self, data: Data) -> Data:
        """Apply per-channel standardization to a single data window.

        Returns the *same* ``Data`` object with the signal field replaced
        by a normalized copy.  The original cached signal array is never
        modified in-place.
        """
        field_name, signal_source, keep_mask, channel_names = (
            resolve_neural_signal(data, self.supported_modalities)
        )

        session_id = str(data.session.id)
        if session_id not in self.stats_by_recording:
            raise KeyError(
                f"No normalization statistics for session {session_id!r}; "
                f"available: {sorted(self.stats_by_recording)}"
            )

        stats = self.stats_by_recording[session_id]
        if stats.recording_id != session_id:
            raise ValueError(
                f"Statistics key {session_id!r} does not match its recording "
                f"ID {stats.recording_id!r}"
            )
        if field_name != stats.signal_field:
            raise ValueError(
                f"Signal field mismatch for session {session_id!r}: got "
                f"{field_name!r}, expected {stats.signal_field!r}"
            )

        n_supported = int(keep_mask.sum())
        if n_supported != len(stats.channel_names):
            raise ValueError(
                f"Session {session_id!r} has {n_supported} supported "
                f"channels but statistics expect "
                f"{len(stats.channel_names)}"
            )

        actual_names = tuple(str(n) for n in channel_names)
        if actual_names != stats.channel_names:
            raise ValueError(
                f"Channel order mismatch for session {session_id!r}: "
                f"got {actual_names}, expected {stats.channel_names}"
            )

        source_signal = np.asarray(signal_source.signal)
        if source_signal.ndim != 2 or source_signal.shape[1] != len(keep_mask):
            raise ValueError(
                f"Session {session_id!r} has signal shape {source_signal.shape}; "
                "expected (time, channels) matching data.channels"
            )
        if not np.issubdtype(source_signal.dtype, np.floating):
            raise ValueError(
                f"Session {session_id!r} signal must use a floating dtype, got "
                f"{source_signal.dtype}"
            )
        supported_signal = source_signal[:, keep_mask]
        if not np.all(np.isfinite(supported_signal)):
            bad = np.where(~np.isfinite(supported_signal).all(axis=0))[0]
            raise ValueError(
                f"Non-finite signal values for session {session_id!r}, channels "
                f"{[actual_names[i] for i in bad]}"
            )

        # Copy only because window objects may share their backing recording
        # array.  Keeping the original dtype preserves unsupported channels
        # byte-for-byte; tokenization performs its existing float32 cast later.
        signal = source_signal.copy()
        signal[:, keep_mask] = (supported_signal - stats.mean) / stats.scale
        signal_source.signal = signal

        return data

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"recordings={len(self.stats_by_recording)}, "
            f"modalities={sorted(self.supported_modalities)}, "
            f"scale_floor={self.scale_floor:.1e})"
        )
