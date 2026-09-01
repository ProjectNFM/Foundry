"""Immutable per-recording per-channel train-split normalization statistics.

Fits per-channel mean and scale from the training partition only for each
recording. Statistics are frozen before any validation or test data is
accessed, preventing information leakage.

The production formula is::

    x_normalized[r, c, t] = (x[r, c, t] - mean_train[r, c])
                            / max(std_train[r, c], scale_floor)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from foundry.data.utils import NEURAL_MODALITIES, resolve_neural_signal


@dataclass(frozen=True)
class RecordingChannelStats:
    """Immutable normalization statistics for one recording's neural channels.

    Fitted from the training partition only. Applied at runtime to standardize
    all partitions (train, validation, test) from that recording.
    """

    recording_id: str
    signal_field: str
    channel_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    sample_count: int
    floored_channels: tuple[int, ...]
    sampling_rate: float

    def __post_init__(self):
        if self.mean.dtype != np.float32:
            object.__setattr__(self, "mean", self.mean.astype(np.float32))
        if self.scale.dtype != np.float32:
            object.__setattr__(self, "scale", self.scale.astype(np.float32))

        n = len(self.channel_names)
        if len(self.mean) != n:
            raise ValueError(
                f"mean length {len(self.mean)} != channel count {n} "
                f"for recording {self.recording_id!r}"
            )
        if len(self.scale) != n:
            raise ValueError(
                f"scale length {len(self.scale)} != channel count {n} "
                f"for recording {self.recording_id!r}"
            )
        if not np.all(np.isfinite(self.mean)):
            bad = list(np.where(~np.isfinite(self.mean))[0])
            raise ValueError(
                f"Non-finite mean values for recording {self.recording_id!r}, "
                f"channels {[self.channel_names[i] for i in bad]}"
            )
        if not np.all(np.isfinite(self.scale)):
            bad = list(np.where(~np.isfinite(self.scale))[0])
            raise ValueError(
                f"Non-finite scale values for recording {self.recording_id!r}, "
                f"channels {[self.channel_names[i] for i in bad]}"
            )
        if not np.all(self.scale > 0):
            raise ValueError(
                f"Non-positive scale values for recording {self.recording_id!r}"
            )
        if self.sample_count <= 0:
            raise ValueError(
                f"Empty train population for recording {self.recording_id!r}"
            )


def merge_time_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping time intervals into disjoint sorted intervals.

    Ensures each sample is counted exactly once when computing statistics.

    Args:
        starts: Interval start times.
        ends: Interval end times.

    Returns:
        Tuple of ``(merged_starts, merged_ends)`` arrays.
    """
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if len(starts) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Interval bounds must be finite")
    if np.any(ends <= starts):
        raise ValueError("Each interval end must be greater than its start")

    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    merged_starts = [starts[0]]
    merged_ends = [ends[0]]

    for i in range(1, len(starts)):
        if starts[i] <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], ends[i])
        else:
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])

    return np.array(merged_starts), np.array(merged_ends)


def _interval_sample_bounds(
    start: float,
    end: float,
    *,
    domain_start: float,
    sampling_rate: float,
    n_samples: int,
) -> tuple[int, int]:
    """Convert a half-open time interval to clipped raw sample bounds."""
    # Sampling intervals are time based, whereas raw arrays are indexed from
    # their recording-domain origin.  The small tolerance avoids turning an
    # exactly aligned value such as 0.6 * 100 into the next sample because of
    # floating-point representation noise.
    offset = np.array([start, end], dtype=np.float64) - domain_start
    indices = np.ceil(offset * sampling_rate - 1e-9).astype(np.int64)
    indices = np.clip(indices, 0, n_samples)
    return int(indices[0]), int(indices[1])


def _sampling_rate_from_signal_source(signal_source: object) -> float:
    """Return a source's sampling rate, deriving it from regular timestamps.

    ``RegularTimeSeries`` exposes ``sampling_rate`` directly, while the
    ArrayDict-backed NeuroSoft recordings expose only regular timestamps.
    Both representations describe the same sampled signal.
    """
    sampling_rate = getattr(signal_source, "sampling_rate", None)
    if sampling_rate is None:
        timestamps = np.asarray(getattr(signal_source, "timestamps"))
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError(
                "Signal source must provide a positive sampling_rate or at "
                "least two one-dimensional timestamps"
            )
        sampling_rate = 1.0 / float(timestamps[1] - timestamps[0])

    sampling_rate = float(sampling_rate)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            f"Signal source has invalid sampling_rate {sampling_rate!r}"
        )
    return sampling_rate


def _interval_sample_bounds_from_timestamps(
    start: float,
    end: float,
    timestamps: np.ndarray,
) -> tuple[int, int]:
    """Map a half-open time interval onto a timestamped sample array."""
    if timestamps.ndim != 1 or len(timestamps) < 1:
        raise ValueError(
            "Signal timestamps must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "Signal timestamps must be finite and strictly increasing"
        )
    return (
        int(np.searchsorted(timestamps, start, side="left")),
        int(np.searchsorted(timestamps, end, side="left")),
    )


def _validate_fit_parameters(
    scale_floor: float, accumulator_dtype: np.dtype
) -> np.dtype:
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be a finite positive value")
    dtype = np.dtype(accumulator_dtype)
    if dtype != np.dtype(np.float64):
        raise ValueError(
            "accumulator_dtype must be float64 to satisfy the numerical policy"
        )
    return dtype


def _collect_recording_moments(
    dataset,
    recording_id: str,
    train_intervals,
    supported_modalities: frozenset[str] = NEURAL_MODALITIES,
    scale_floor: float = 1e-8,
    accumulator_dtype=np.float64,
    chunk_samples: int = 1_000_000,
) -> tuple[str, tuple[str, ...], float, np.ndarray, np.ndarray, int]:
    """Accumulate per-channel train-split moments for one recording."""
    accumulator_dtype = _validate_fit_parameters(scale_floor, accumulator_dtype)
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")

    data = dataset.get_recording(recording_id)
    field_name, signal_source, keep_mask, channel_names = resolve_neural_signal(
        data, supported_modalities
    )

    signal = np.asarray(signal_source.signal)
    if signal.ndim != 2 or signal.shape[1] != len(keep_mask):
        raise ValueError(
            f"Recording {recording_id!r} has signal shape {signal.shape}; "
            "expected (time, channels) matching data.channels"
        )
    n_channels = int(keep_mask.sum())

    if n_channels == 0:
        raise ValueError(
            f"Recording {recording_id!r} has no channels with supported "
            f"modalities {supported_modalities}"
        )

    starts = np.asarray(train_intervals.start)
    ends = np.asarray(train_intervals.end)
    merged_starts, merged_ends = merge_time_intervals(starts, ends)

    sum_accum = np.zeros(n_channels, dtype=accumulator_dtype)
    sq_sum_accum = np.zeros(n_channels, dtype=accumulator_dtype)
    count_accum = 0

    sampling_rate = _sampling_rate_from_signal_source(signal_source)
    timestamps = np.asarray(
        getattr(signal_source, "timestamps", ()), dtype=np.float64
    )
    has_sample_timestamps = (
        timestamps.ndim == 1 and len(timestamps) == signal.shape[0]
    )
    if not has_sample_timestamps:
        domain_starts = np.asarray(data.domain.start, dtype=np.float64).reshape(
            -1
        )
        if len(domain_starts) != 1:
            raise ValueError(
                f"Recording {recording_id!r} has {len(domain_starts)} domains "
                "but its signal source provides no timestamp per sample"
            )
        domain_start = float(domain_starts[0])

    for iv_start, iv_end in zip(merged_starts, merged_ends):
        if has_sample_timestamps:
            idx_start, idx_end = _interval_sample_bounds_from_timestamps(
                iv_start, iv_end, timestamps
            )
        else:
            idx_start, idx_end = _interval_sample_bounds(
                iv_start,
                iv_end,
                domain_start=domain_start,
                sampling_rate=sampling_rate,
                n_samples=signal.shape[0],
            )
        if idx_start >= idx_end:
            continue
        for chunk_start in range(idx_start, idx_end, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, idx_end)
            chunk = signal[chunk_start:chunk_end, keep_mask].astype(
                accumulator_dtype, copy=False
            )
            if not np.all(np.isfinite(chunk)):
                bad = np.where(~np.isfinite(chunk).all(axis=0))[0]
                raise ValueError(
                    f"Non-finite train samples for recording {recording_id!r}, "
                    f"channels {[str(channel_names[i]) for i in bad]}"
                )
            sum_accum += chunk.sum(axis=0)
            sq_sum_accum += np.square(chunk).sum(axis=0)
            count_accum += chunk.shape[0]

    if count_accum == 0:
        raise RuntimeError(
            f"Recording {recording_id!r} has no train samples within "
            "the provided intervals"
        )
    return (
        field_name,
        tuple(str(n) for n in channel_names),
        sampling_rate,
        sum_accum,
        sq_sum_accum,
        count_accum,
    )


def fit_recording_stats(
    dataset,
    recording_id: str,
    train_intervals,
    supported_modalities: frozenset[str] = NEURAL_MODALITIES,
    scale_floor: float = 1e-8,
    accumulator_dtype=np.float64,
    chunk_samples: int = 1_000_000,
) -> RecordingChannelStats:
    """Fit per-channel mean and scale from training-split samples only.

    Statistics are accumulated in *accumulator_dtype* (default ``float64``)
    over merged, deduplicated time intervals.  The population standard
    deviation (``ddof=0``) is used, and a configurable *scale_floor*
    prevents division by near-zero values.

    Args:
        dataset: A ``torch_brain`` dataset with ``get_recording()``.
        recording_id: Canonical recording identifier.
        train_intervals: An ``Interval`` object with ``.start`` / ``.end``
            arrays for the training partition of this recording.
        supported_modalities: Modality strings to include.
        scale_floor: Positive lower bound on per-channel scale.
        accumulator_dtype: Numpy dtype for intermediate sums.

    Returns:
        Frozen :class:`RecordingChannelStats` for the recording.

    Raises:
        RuntimeError: If the recording has no train samples.
        ValueError: If non-finite statistics are produced.
    """
    (
        field_name,
        channel_names,
        sampling_rate,
        sum_accum,
        sq_sum_accum,
        count_accum,
    ) = _collect_recording_moments(
        dataset,
        recording_id,
        train_intervals,
        supported_modalities=supported_modalities,
        scale_floor=scale_floor,
        accumulator_dtype=accumulator_dtype,
        chunk_samples=chunk_samples,
    )

    mean = sum_accum / count_accum
    variance = sq_sum_accum / count_accum - mean**2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    if not np.all(np.isfinite(std)):
        bad = np.where(~np.isfinite(std))[0]
        raise ValueError(
            f"Non-finite scales for recording {recording_id!r}, channels "
            f"{[str(channel_names[i]) for i in bad]}"
        )
    near_zero = np.where(std <= scale_floor)[0]
    if len(near_zero):
        raise ValueError(
            f"Recording {recording_id!r} has channel(s) at or below "
            f"scale_floor={scale_floor:.2e}: "
            f"{[str(channel_names[i]) for i in near_zero]}"
        )

    return RecordingChannelStats(
        recording_id=recording_id,
        signal_field=field_name,
        channel_names=channel_names,
        mean=mean.astype(np.float32),
        scale=std.astype(np.float32),
        sample_count=count_accum,
        floored_channels=(),
        sampling_rate=sampling_rate,
    )


def fit_recording_global_stats(
    dataset,
    recording_id: str,
    train_intervals,
    supported_modalities: frozenset[str] = NEURAL_MODALITIES,
    scale_floor: float = 1e-8,
    accumulator_dtype=np.float64,
    chunk_samples: int = 1_000_000,
) -> RecordingChannelStats:
    """Fit one recording-wide z-score from training-split samples only.

    The scalar mean and standard deviation are accumulated across both time
    and supported channels, then broadcast over channels for compatibility
    with :class:`RecordingChannelStandardize`.  This preserves all relative
    channel amplitudes while correcting recording-wide offset and scale.
    """
    (
        field_name,
        channel_names,
        sampling_rate,
        sum_accum,
        sq_sum_accum,
        count_per_channel,
    ) = _collect_recording_moments(
        dataset,
        recording_id,
        train_intervals,
        supported_modalities=supported_modalities,
        scale_floor=scale_floor,
        accumulator_dtype=accumulator_dtype,
        chunk_samples=chunk_samples,
    )
    sample_count = count_per_channel * len(channel_names)
    mean = float(sum_accum.sum() / sample_count)
    variance = float(sq_sum_accum.sum() / sample_count - mean**2)
    scale = float(np.sqrt(max(variance, 0.0)))
    if not np.isfinite(scale):
        raise ValueError(
            f"Non-finite global scale for recording {recording_id!r}"
        )
    if scale <= scale_floor:
        raise ValueError(
            f"Recording {recording_id!r} has global scale at or below "
            f"scale_floor={scale_floor:.2e}"
        )
    return RecordingChannelStats(
        recording_id=recording_id,
        signal_field=field_name,
        channel_names=channel_names,
        mean=np.full(len(channel_names), mean, dtype=np.float32),
        scale=np.full(len(channel_names), scale, dtype=np.float32),
        sample_count=sample_count,
        floored_channels=(),
        sampling_rate=sampling_rate,
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_normalization_stats(
    stats: Mapping[str, RecordingChannelStats],
    output_dir: str | Path,
    normalization_config: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Persist frozen statistics and their hash-verifiable manifest.

    The arrays are stored separately from the human-readable manifest so a
    later evaluation can reject both a corrupt array artifact and mismatched
    recording/channel metadata before producing predictions.
    """
    if not stats:
        raise ValueError(
            "Cannot save an empty normalization-statistics mapping"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "input_normalization_stats.npz"
    npz_data = {
        f"{recording_id}/mean": recording_stats.mean
        for recording_id, recording_stats in stats.items()
    }
    npz_data.update(
        {
            f"{recording_id}/scale": recording_stats.scale
            for recording_id, recording_stats in stats.items()
        }
    )
    np.savez(npz_path, **npz_data)

    manifest: dict[str, Any] = {
        "mode": normalization_config.get("mode"),
        "supported_modalities": sorted(
            normalization_config.get("supported_modalities", NEURAL_MODALITIES)
        ),
        "scale_floor": normalization_config.get("scale_floor", 1e-8),
        "accumulator_dtype": normalization_config.get(
            "accumulator_dtype", "float64"
        ),
        "stats_artifact_sha256": _sha256_file(npz_path),
        "recordings": {
            recording_id: {
                "signal_field": recording_stats.signal_field,
                "channel_names": list(recording_stats.channel_names),
                "sample_count": recording_stats.sample_count,
                "sampling_rate": recording_stats.sampling_rate,
                "mean": recording_stats.mean.tolist(),
                "scale": recording_stats.scale.tolist(),
                "floored_channels": list(recording_stats.floored_channels),
            }
            for recording_id, recording_stats in stats.items()
        },
    }
    if provenance:
        manifest["provenance"] = dict(provenance)

    manifest_path = output_dir / "input_normalization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return npz_path, manifest_path


def load_normalization_stats(
    npz_path: str | Path, manifest_path: str | Path
) -> dict[str, RecordingChannelStats]:
    """Load statistics only after verifying their manifest and content hash."""
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Both normalization stats NPZ and manifest JSON must exist"
        )
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Could not read normalization manifest {manifest_path}"
        ) from exc

    expected_hash = manifest.get("stats_artifact_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _sha256_file(
        npz_path
    ):
        raise ValueError("Normalization statistics artifact SHA-256 mismatch")
    recordings = manifest.get("recordings")
    if not isinstance(recordings, dict) or not recordings:
        raise ValueError("Normalization manifest has no recording metadata")

    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            loaded = {
                recording_id: RecordingChannelStats(
                    recording_id=recording_id,
                    signal_field=metadata["signal_field"],
                    channel_names=tuple(metadata["channel_names"]),
                    mean=arrays[f"{recording_id}/mean"],
                    scale=arrays[f"{recording_id}/scale"],
                    sample_count=metadata["sample_count"],
                    floored_channels=tuple(metadata["floored_channels"]),
                    sampling_rate=metadata["sampling_rate"],
                )
                for recording_id, metadata in recordings.items()
            }
            for recording_id, recording_stats in loaded.items():
                metadata = recordings[recording_id]
                if not np.array_equal(
                    recording_stats.mean,
                    np.asarray(metadata["mean"], dtype=np.float32),
                ) or not np.array_equal(
                    recording_stats.scale,
                    np.asarray(metadata["scale"], dtype=np.float32),
                ):
                    raise ValueError(
                        "Normalization stats artifact does not match its manifest"
                    )
            return loaded
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise ValueError(
            "Normalization stats artifact does not match its manifest"
        ) from exc
