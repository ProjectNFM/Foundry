"""Pure metric functions for embedding visualization.

All functions are testable without Lightning or W&B. They operate on
L2-normalized representation vectors and produce deterministic metric
dictionaries with stable keys and coverage metadata.

Cosine metrics use the original normalized representation vectors, not PCA
coordinates. PCA is an event-level visual summary only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)

NORM_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Normalization and invalid-vector accounting
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Output of :func:`normalize_representations`.

    Attributes:
        vectors: L2-normalized vectors for valid rows only, shape ``(N_valid, D)``.
        norms: Raw L2 norms for all input rows, shape ``(N,)``.
        valid_mask: Boolean mask over input rows, shape ``(N,)``.
        n_total: Total number of input vectors.
        n_valid: Number of valid (finite, non-zero) vectors.
        n_zero: Number of zero-norm vectors excluded.
        n_nonfinite: Number of non-finite vectors excluded.
    """

    vectors: np.ndarray
    norms: np.ndarray
    valid_mask: np.ndarray
    n_total: int
    n_valid: int
    n_zero: int
    n_nonfinite: int


def normalize_representations(
    vectors: np.ndarray, epsilon: float = NORM_EPSILON
) -> NormalizationResult:
    """L2-normalize representations, excluding zero and non-finite vectors.

    Args:
        vectors: Input array of shape ``(N, D)``.
        epsilon: Smallest norm considered safe to normalize. Finite vectors
            at or below this threshold are counted as zero-norm and excluded.

    Returns:
        A :class:`NormalizationResult` with the normalized valid subset,
        raw norms, and exclusion counts.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    norms = np.linalg.norm(vectors, axis=1)

    is_finite = np.isfinite(norms) & np.all(np.isfinite(vectors), axis=1)
    is_nonzero = norms > epsilon
    valid_mask = is_finite & is_nonzero

    n_total = len(vectors)
    n_nonfinite = int(np.sum(~is_finite))
    n_zero = int(np.sum(is_finite & ~is_nonzero))
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        return NormalizationResult(
            vectors=np.empty((0, vectors.shape[1]), dtype=vectors.dtype),
            norms=norms,
            valid_mask=valid_mask,
            n_total=n_total,
            n_valid=0,
            n_zero=n_zero,
            n_nonfinite=n_nonfinite,
        )

    valid_vectors = vectors[valid_mask]
    valid_norms = norms[valid_mask]
    # Invalid zero-norm vectors have already been removed above, so dividing
    # by the raw norm preserves a true unit norm. Adding epsilon here would
    # systematically make every output shorter than one, which in turn means
    # downstream dot products are no longer cosine similarities.
    normalized = valid_vectors / valid_norms[:, np.newaxis]

    return NormalizationResult(
        vectors=normalized,
        norms=norms,
        valid_mask=valid_mask,
        n_total=n_total,
        n_valid=n_valid,
        n_zero=n_zero,
        n_nonfinite=n_nonfinite,
    )


def compute_norm_statistics(norms: np.ndarray) -> dict[str, float]:
    """Summary statistics for raw L2 norms of valid vectors."""
    valid = norms[np.isfinite(norms) & (norms > 0)]
    if len(valid) == 0:
        return {}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
    }


# ---------------------------------------------------------------------------
# Cosine geometry utilities
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray | None = None
) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` and ``b``.

    Both inputs must already be L2-normalized. If ``b`` is None, computes
    the self-similarity matrix of ``a``.
    """
    if b is None:
        b = a
    return a @ b.T


def cosine_distance_matrix(normalized_vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix: ``1 - cos(a, b)``.

    Input must be L2-normalized. The result is clipped to ``[0, 2]``.
    """
    sim = cosine_similarity_matrix(normalized_vectors)
    return np.clip(1.0 - sim, 0.0, 2.0)


# ---------------------------------------------------------------------------
# Backbone cosine silhouette (Section 10.3)
# ---------------------------------------------------------------------------


@dataclass
class SilhouetteResult:
    """Result of a cosine silhouette computation for one grouping.

    Attributes:
        score: Macro silhouette score (mean over per-sample silhouettes),
            or None if fewer than 2 eligible groups remain.
        n_samples: Number of samples included in the score.
        n_excluded: Number of samples excluded (singleton or invalid groups).
        n_groups: Number of eligible groups (each with ≥2 members).
        n_excluded_groups: Number of groups excluded (singletons or invalid).
    """

    score: float | None
    n_samples: int
    n_excluded: int
    n_groups: int
    n_excluded_groups: int


def cosine_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> SilhouetteResult:
    """Cosine-distance silhouette score for a single grouping.

    Samples in groups with fewer than 2 members are excluded. The score is
    omitted (None) when fewer than 2 eligible groups remain.

    Exclusion rule: a group is eligible if it contains at least 2 members.
    Samples in ineligible groups are excluded from the score.

    Args:
        distance_matrix: Precomputed pairwise cosine distance matrix of shape
            ``(N, N)``, computed from L2-normalized original vectors.
        labels: Integer or string group labels of length ``N``. Negative
            integers or empty strings are treated as invalid/unlabeled.

    Returns:
        A :class:`SilhouetteResult` with the score and coverage counts.
    """
    n = len(labels)
    if n == 0:
        return SilhouetteResult(
            score=None,
            n_samples=0,
            n_excluded=0,
            n_groups=0,
            n_excluded_groups=0,
        )

    labels = np.asarray(labels)

    if labels.dtype.kind in ("i", "u"):
        valid_label_mask = labels >= 0
    elif labels.dtype.kind in ("U", "S", "O"):
        valid_label_mask = np.array([bool(str(lb).strip()) for lb in labels])
    else:
        valid_label_mask = np.ones(n, dtype=bool)

    unique_labels = np.unique(labels[valid_label_mask])
    group_sizes = {
        lb: int(np.sum((labels == lb) & valid_label_mask))
        for lb in unique_labels
    }
    eligible_labels = {lb for lb, sz in group_sizes.items() if sz >= 2}
    n_excluded_groups = len(unique_labels) - len(eligible_labels)

    eligible_mask = valid_label_mask & np.array(
        [lb in eligible_labels for lb in labels]
    )
    n_eligible = int(np.sum(eligible_mask))
    n_excluded = n - n_eligible

    if len(eligible_labels) < 2:
        return SilhouetteResult(
            score=None,
            n_samples=n_eligible,
            n_excluded=n_excluded,
            n_groups=len(eligible_labels),
            n_excluded_groups=n_excluded_groups,
        )

    eligible_indices = np.where(eligible_mask)[0]
    eligible_labels_arr = labels[eligible_indices]
    sub_dist = distance_matrix[np.ix_(eligible_indices, eligible_indices)]

    silhouettes = np.zeros(n_eligible)
    for i in range(n_eligible):
        own_label = eligible_labels_arr[i]
        same_mask = eligible_labels_arr == own_label
        same_mask[i] = False  # exclude self
        n_same = int(np.sum(same_mask))

        if n_same == 0:
            silhouettes[i] = 0.0
            continue

        a_i = float(np.mean(sub_dist[i, same_mask]))

        b_i = np.inf
        for other_label in eligible_labels:
            if other_label == own_label:
                continue
            other_mask = eligible_labels_arr == other_label
            mean_dist = float(np.mean(sub_dist[i, other_mask]))
            b_i = min(b_i, mean_dist)

        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return SilhouetteResult(
        score=float(np.mean(silhouettes)),
        n_samples=n_eligible,
        n_excluded=n_excluded,
        n_groups=len(eligible_labels),
        n_excluded_groups=n_excluded_groups,
    )


def compute_backbone_silhouettes(
    distance_matrix: np.ndarray,
    groupings: dict[str, np.ndarray],
) -> dict[str, SilhouetteResult]:
    """Compute cosine silhouette for every backbone grouping.

    Reuses a single precomputed cosine-distance matrix for all groupings
    (dataset, subject, session, per-task class).

    Args:
        distance_matrix: Pairwise cosine distance, shape ``(N, N)``.
        groupings: ``{name: labels}`` where labels has length ``N``.

    Returns:
        ``{name: SilhouetteResult}`` for each grouping.
    """
    return {
        name: cosine_silhouette(distance_matrix, labels)
        for name, labels in groupings.items()
    }


# ---------------------------------------------------------------------------
# Channel metrics (Section 9.4)
# ---------------------------------------------------------------------------


@dataclass
class TemporalConsistencyResult:
    """Result of channel temporal-consistency computation.

    For dynamic mode, each channel with ≥2 windows contributes its mean
    cosine similarity from each observation to its leave-one-out centroid.
    The score is macro-averaged over channels, then recordings.

    For static mode, temporal consistency is 1.0 by construction because the
    representation is a fixed lookup vector per channel identity.
    """

    score: float | None
    n_recordings: int
    n_channels: int
    n_observations: int
    is_static: bool


def channel_temporal_consistency(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    channel_mode: str,
) -> TemporalConsistencyResult:
    """Temporal consistency of dynamic channel representations.

    For each recording-specific channel with at least two windows, computes
    the mean cosine similarity from each observation to its leave-one-out
    channel centroid. Macro-averages over channels within a recording, then
    over recordings, so high-channel-count recordings do not dominate.

    Static channel representations return 1.0 by construction with
    ``is_static=True``.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording/session identifier per observation, length ``N``.
        channel_ids: Channel identifier per observation, length ``N``.
        channel_mode: ``"static"`` or ``"dynamic"``.

    Returns:
        :class:`TemporalConsistencyResult`.
    """
    n = len(normalized_vectors)
    if n == 0:
        return TemporalConsistencyResult(
            score=None,
            n_recordings=0,
            n_channels=0,
            n_observations=0,
            is_static=channel_mode == "static",
        )

    if channel_mode == "static":
        unique_channels = len(set(zip(recording_ids, channel_ids)))
        unique_recordings = len(set(recording_ids))
        return TemporalConsistencyResult(
            score=1.0,
            n_recordings=unique_recordings,
            n_channels=unique_channels,
            n_observations=n,
            is_static=True,
        )

    recording_ids = np.asarray(recording_ids)
    channel_ids = np.asarray(channel_ids)
    unique_recordings = np.unique(recording_ids)

    recording_scores: list[float] = []
    total_channels = 0
    total_obs = 0

    for rec_id in unique_recordings:
        rec_mask = recording_ids == rec_id
        rec_vectors = normalized_vectors[rec_mask]
        rec_channels = channel_ids[rec_mask]
        unique_channels_in_rec = np.unique(rec_channels)

        channel_scores: list[float] = []
        for ch_id in unique_channels_in_rec:
            ch_mask = rec_channels == ch_id
            ch_vectors = rec_vectors[ch_mask]
            n_ch = len(ch_vectors)
            if n_ch < 2:
                continue

            total_channels += 1
            total_obs += n_ch

            obs_sims: list[float] = []
            for i in range(n_ch):
                others = np.delete(ch_vectors, i, axis=0)
                centroid = others.mean(axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0:
                    centroid /= c_norm
                sim = float(ch_vectors[i] @ centroid)
                obs_sims.append(sim)

            channel_scores.append(float(np.mean(obs_sims)))

        if channel_scores:
            recording_scores.append(float(np.mean(channel_scores)))

    score = float(np.mean(recording_scores)) if recording_scores else None

    return TemporalConsistencyResult(
        score=score,
        n_recordings=len(recording_scores),
        n_channels=total_channels,
        n_observations=total_obs,
        is_static=False,
    )


@dataclass
class SeparabilityResult:
    """Result of within-recording channel separability.

    Each eligible dynamic observation is classified by the nearest
    leave-one-out channel centroid within its recording.

    Attributes:
        accuracy: Macro accuracy over channels and recordings.
        margin: Mean cosine margin between correct and nearest incorrect
            centroid, macro-averaged over channels and recordings.
        n_recordings: Number of recordings with ≥2 eligible channels.
        n_channels: Total eligible channels across recordings.
        n_observations: Total classified observations.
        unavailable_reason: If set, explains why the metric was not computed
            (e.g. static mode).
    """

    accuracy: float | None
    margin: float | None
    n_recordings: int
    n_channels: int
    n_observations: int
    unavailable_reason: str | None = None


def channel_within_recording_separability(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    channel_mode: str,
) -> SeparabilityResult:
    """Within-recording separability of dynamic channel representations.

    For each eligible dynamic observation, classifies it by the nearest
    leave-one-out channel centroid within its recording. Reports macro
    accuracy over channels and recordings, plus the mean cosine margin
    between the correct centroid and the nearest incorrect centroid.

    Omitted for static mode because a single lookup vector per channel
    cannot support a non-leaking leave-one-out estimate.

    Requires at least 2 channels per recording (each with ≥2 windows) to
    compute a meaningful classification.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording/session identifier per observation.
        channel_ids: Channel identifier per observation.
        channel_mode: ``"static"`` or ``"dynamic"``.

    Returns:
        :class:`SeparabilityResult`.
    """
    n = len(normalized_vectors)
    if channel_mode == "static":
        return SeparabilityResult(
            accuracy=None,
            margin=None,
            n_recordings=0,
            n_channels=0,
            n_observations=0,
            unavailable_reason="static mode: single lookup vector per channel",
        )
    if n == 0:
        return SeparabilityResult(
            accuracy=None,
            margin=None,
            n_recordings=0,
            n_channels=0,
            n_observations=0,
        )

    recording_ids = np.asarray(recording_ids)
    channel_ids = np.asarray(channel_ids)
    unique_recordings = np.unique(recording_ids)

    recording_accuracies: list[float] = []
    recording_margins: list[float] = []
    total_channels = 0
    total_obs = 0

    for rec_id in unique_recordings:
        rec_mask = recording_ids == rec_id
        rec_vectors = normalized_vectors[rec_mask]
        rec_channels = channel_ids[rec_mask]
        unique_ch = np.unique(rec_channels)

        eligible_channels = [
            ch for ch in unique_ch if np.sum(rec_channels == ch) >= 2
        ]
        if len(eligible_channels) < 2:
            continue

        channel_accuracies: list[float] = []
        channel_margins: list[float] = []

        for ch_id in eligible_channels:
            ch_mask = rec_channels == ch_id
            ch_indices = np.where(ch_mask)[0]
            n_ch = len(ch_indices)
            total_channels += 1
            total_obs += n_ch

            obs_correct: list[bool] = []
            obs_margins: list[float] = []

            for i_local, i_global in enumerate(ch_indices):
                other_ch_vectors = np.delete(
                    rec_vectors[ch_mask], i_local, axis=0
                )
                own_centroid = other_ch_vectors.mean(axis=0)
                own_norm = np.linalg.norm(own_centroid)
                if own_norm > 0:
                    own_centroid /= own_norm
                own_sim = float(rec_vectors[i_global] @ own_centroid)

                best_other_sim = -np.inf
                for other_ch in eligible_channels:
                    if other_ch == ch_id:
                        continue
                    other_vecs = rec_vectors[rec_channels == other_ch]
                    other_cent = other_vecs.mean(axis=0)
                    o_norm = np.linalg.norm(other_cent)
                    if o_norm > 0:
                        other_cent /= o_norm
                    sim = float(rec_vectors[i_global] @ other_cent)
                    best_other_sim = max(best_other_sim, sim)

                is_correct = own_sim > best_other_sim
                obs_correct.append(is_correct)
                obs_margins.append(own_sim - best_other_sim)

            channel_accuracies.append(float(np.mean(obs_correct)))
            channel_margins.append(float(np.mean(obs_margins)))

        if channel_accuracies:
            recording_accuracies.append(float(np.mean(channel_accuracies)))
            recording_margins.append(float(np.mean(channel_margins)))

    if not recording_accuracies:
        return SeparabilityResult(
            accuracy=None,
            margin=None,
            n_recordings=0,
            n_channels=total_channels,
            n_observations=total_obs,
        )

    return SeparabilityResult(
        accuracy=float(np.mean(recording_accuracies)),
        margin=float(np.mean(recording_margins)),
        n_recordings=len(recording_accuracies),
        n_channels=total_channels,
        n_observations=total_obs,
    )


@dataclass
class CanonicalConsistencyResult:
    """Cross-recording canonical-electrode consistency.

    For canonical electrode labels present in at least two recordings,
    classifies each recording-specific channel centroid against canonical
    electrode centroids built from other recordings only (leave-one-recording-out).

    Attributes:
        accuracy: Macro accuracy over canonical electrodes.
        margin: Mean cosine margin, macro-averaged.
        n_electrodes: Eligible canonical electrodes (present in ≥2 recordings).
        n_centroids: Total recording-specific centroids classified.
        n_excluded_electrodes: Electrodes present in only 1 recording.
    """

    accuracy: float | None
    margin: float | None
    n_electrodes: int
    n_centroids: int
    n_excluded_electrodes: int


def channel_canonical_consistency(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    canonical_labels: np.ndarray,
    channel_ids: np.ndarray | None = None,
) -> CanonicalConsistencyResult:
    """Cross-recording canonical-electrode consistency.

    For canonical electrode labels present in at least two recordings,
    classifies each recording-specific channel centroid against canonical
    electrode centroids built from other recordings only.

    This metric applies to both static and dynamic channel modes.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording/session identifier per observation.
        canonical_labels: Canonical electrode label per observation (e.g.
            normalized bare electrode name).
        channel_ids: Recording-specific channel identifier per observation.
            When omitted, ``canonical_labels`` are used for backwards
            compatibility. Supplying identifiers preserves distinct channels
            that resolve to the same canonical label in one recording.

    Returns:
        :class:`CanonicalConsistencyResult`.
    """
    n = len(normalized_vectors)
    if n == 0:
        return CanonicalConsistencyResult(
            accuracy=None,
            margin=None,
            n_electrodes=0,
            n_centroids=0,
            n_excluded_electrodes=0,
        )

    recording_ids = np.asarray(recording_ids)
    canonical_labels = np.asarray(canonical_labels)
    if channel_ids is None:
        channel_ids = canonical_labels
    channel_ids = np.asarray(channel_ids)

    recording_channel_centroids: dict[tuple[str, str], np.ndarray] = {}
    channel_labels: dict[tuple[str, str], str] = {}
    for rec_id in np.unique(recording_ids):
        rec_mask = recording_ids == rec_id
        for channel_id in np.unique(channel_ids[rec_mask]):
            mask = rec_mask & (channel_ids == channel_id)
            vecs = normalized_vectors[mask]
            centroid = vecs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm
            key = (str(rec_id), str(channel_id))
            recording_channel_centroids[key] = centroid

            labels = np.unique(canonical_labels[mask])
            if len(labels) != 1:
                raise ValueError(
                    "A recording-specific channel must have one canonical label"
                )
            channel_labels[key] = str(labels[0])

    electrode_recordings: dict[str, set[str]] = {}
    for key, can_label in channel_labels.items():
        rec_id, _ = key
        electrode_recordings.setdefault(can_label, set()).add(rec_id)

    eligible = {
        e: recs for e, recs in electrode_recordings.items() if len(recs) >= 2
    }
    n_excluded = len(electrode_recordings) - len(eligible)

    if len(eligible) < 2:
        return CanonicalConsistencyResult(
            accuracy=None,
            margin=None,
            n_electrodes=len(eligible),
            n_centroids=0,
            n_excluded_electrodes=n_excluded,
        )

    electrode_accuracies: list[float] = []
    electrode_margins: list[float] = []
    total_centroids = 0

    for electrode, recordings in eligible.items():
        centroid_correct: list[bool] = []
        centroid_margins: list[float] = []

        target_keys = [
            key for key, label in channel_labels.items() if label == electrode
        ]
        for target_rec, target_channel in target_keys:
            target_centroid = recording_channel_centroids[
                (target_rec, target_channel)
            ]

            def leave_one_recording_out_prototype(label: str) -> np.ndarray:
                other_centroids = [
                    centroid
                    for (
                        rec,
                        channel,
                    ), centroid in recording_channel_centroids.items()
                    if rec != target_rec
                    and channel_labels[(rec, channel)] == label
                ]
                prototype = np.mean(other_centroids, axis=0)
                prototype_norm = np.linalg.norm(prototype)
                if prototype_norm > 0:
                    prototype /= prototype_norm
                return prototype

            canonical_centroid = leave_one_recording_out_prototype(electrode)
            own_sim = float(target_centroid @ canonical_centroid)

            other_sims = [
                float(
                    target_centroid
                    @ leave_one_recording_out_prototype(other_electrode)
                )
                for other_electrode in eligible
                if other_electrode != electrode
            ]
            best_other_sim = max(other_sims)

            centroid_correct.append(own_sim > best_other_sim)
            centroid_margins.append(own_sim - best_other_sim)
            total_centroids += 1

        if centroid_correct:
            electrode_accuracies.append(float(np.mean(centroid_correct)))
            electrode_margins.append(float(np.mean(centroid_margins)))

    if not electrode_accuracies:
        return CanonicalConsistencyResult(
            accuracy=None,
            margin=None,
            n_electrodes=len(eligible),
            n_centroids=total_centroids,
            n_excluded_electrodes=n_excluded,
        )

    return CanonicalConsistencyResult(
        accuracy=float(np.mean(electrode_accuracies)),
        margin=float(np.mean(electrode_margins)),
        n_electrodes=len(eligible),
        n_centroids=total_centroids,
        n_excluded_electrodes=n_excluded,
    )


# ---------------------------------------------------------------------------
# Anatomical organization (Section 9.4)
# ---------------------------------------------------------------------------


@dataclass
class AnatomyResult:
    """Anatomical organization scores.

    Attributes:
        centroid_spearman: Median Spearman correlation (physical distance vs.
            cosine distance) using recording-specific channel centroids,
            across eligible recordings.
        window_spearman: Median Spearman correlation computed separately per
            eligible window (dynamic mode only).
        centroid_iqr: Interquartile range of per-recording centroid Spearman.
        window_iqr: Interquartile range of per-window Spearman.
        n_eligible_recordings: Recordings with ≥ ``min_positioned`` resolved
            channels.
        n_eligible_windows: Windows with ≥ ``min_positioned`` resolved channels.
        n_resolved_channels: Total channel observations with resolved positions.
        n_undefined: Correlations that were undefined (constant input) and excluded.
    """

    centroid_spearman: float | None
    window_spearman: float | None
    centroid_iqr: float | None
    window_iqr: float | None
    n_eligible_recordings: int
    n_eligible_windows: int
    n_resolved_channels: int
    n_undefined: int


def channel_anatomical_scores(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    positions_3d: dict[str, np.ndarray],
    channel_mode: str,
    min_positioned: int = 9,
) -> AnatomyResult:
    """Anatomical organization of channel representations.

    Computes Spearman correlation between physical 3D distances and pairwise
    cosine distances between channel representations, for recordings with at
    least ``min_positioned`` resolved channel positions.

    **Centroid score:** Uses recording-specific channel centroids across
    sampled windows.

    **Per-window score:** In dynamic mode, computes the correlation separately
    for each eligible window containing at least ``min_positioned`` resolved
    channels.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording identifier per observation, length ``N``.
        channel_ids: Channel identifier per observation, length ``N``.
        positions_3d: ``{canonical_label: np.ndarray(3,)}`` mapping resolved
            channel labels to 3D coordinates.
        channel_mode: ``"static"`` or ``"dynamic"``.
        min_positioned: Minimum resolved channels for a recording/window to
            be eligible (default 9 per plan).

    Returns:
        :class:`AnatomyResult`.
    """
    n = len(normalized_vectors)
    if n == 0 or not positions_3d:
        return AnatomyResult(
            centroid_spearman=None,
            window_spearman=None,
            centroid_iqr=None,
            window_iqr=None,
            n_eligible_recordings=0,
            n_eligible_windows=0,
            n_resolved_channels=0,
            n_undefined=0,
        )

    recording_ids = np.asarray(recording_ids)
    channel_ids = np.asarray(channel_ids)

    resolved_mask = np.array([str(ch) in positions_3d for ch in channel_ids])
    n_resolved = int(np.sum(resolved_mask))

    centroid_spearman_values: list[float] = []
    n_undefined = 0
    n_eligible_recordings = 0

    for rec_id in np.unique(recording_ids):
        rec_mask = recording_ids == rec_id
        rec_resolved = rec_mask & resolved_mask
        resolved_channels_in_rec = np.unique(channel_ids[rec_resolved])

        if len(resolved_channels_in_rec) < min_positioned:
            continue
        n_eligible_recordings += 1

        centroids = []
        phys_positions = []
        for ch_id in resolved_channels_in_rec:
            ch_mask = rec_resolved & (channel_ids == ch_id)
            ch_vecs = normalized_vectors[ch_mask]
            centroid = ch_vecs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm
            centroids.append(centroid)
            phys_positions.append(positions_3d[str(ch_id)])

        centroids_arr = np.array(centroids)
        phys_arr = np.array(phys_positions)

        rho = _spearman_distance_correlation(centroids_arr, phys_arr)
        if rho is not None:
            centroid_spearman_values.append(rho)
        else:
            n_undefined += 1

    centroid_result = _median_iqr(centroid_spearman_values)

    return AnatomyResult(
        centroid_spearman=centroid_result[0],
        window_spearman=None,
        centroid_iqr=centroid_result[1],
        window_iqr=None,
        n_eligible_recordings=n_eligible_recordings,
        n_eligible_windows=0,
        n_resolved_channels=n_resolved,
        n_undefined=n_undefined,
    )


def channel_anatomical_scores_with_windows(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    window_ids: np.ndarray,
    positions_3d: dict[str, np.ndarray],
    channel_mode: str,
    min_positioned: int = 9,
) -> AnatomyResult:
    """Anatomical organization with explicit window identifiers.

    Like :func:`channel_anatomical_scores` but also computes per-window
    Spearman correlations when ``channel_mode == "dynamic"``.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording identifier per observation, length ``N``.
        channel_ids: Channel identifier per observation, length ``N``.
        window_ids: Window identifier per observation (e.g. stringified
            ``ObservationIdentity``), length ``N``.
        positions_3d: ``{canonical_label: np.ndarray(3,)}`` resolved positions.
        channel_mode: ``"static"`` or ``"dynamic"``.
        min_positioned: Minimum resolved channels per recording/window.

    Returns:
        :class:`AnatomyResult`.
    """
    n = len(normalized_vectors)
    if n == 0 or not positions_3d:
        return AnatomyResult(
            centroid_spearman=None,
            window_spearman=None,
            centroid_iqr=None,
            window_iqr=None,
            n_eligible_recordings=0,
            n_eligible_windows=0,
            n_resolved_channels=0,
            n_undefined=0,
        )

    recording_ids = np.asarray(recording_ids)
    channel_ids = np.asarray(channel_ids)
    window_ids = np.asarray(window_ids)

    resolved_mask = np.array([str(ch) in positions_3d for ch in channel_ids])
    n_resolved = int(np.sum(resolved_mask))

    centroid_spearman_values: list[float] = []
    window_spearman_values: list[float] = []
    n_undefined = 0
    n_eligible_recordings = 0
    n_eligible_windows = 0

    for rec_id in np.unique(recording_ids):
        rec_mask = recording_ids == rec_id
        rec_resolved = rec_mask & resolved_mask
        resolved_channels_in_rec = np.unique(channel_ids[rec_resolved])

        if len(resolved_channels_in_rec) < min_positioned:
            continue
        n_eligible_recordings += 1

        centroids = []
        phys_positions = []
        for ch_id in resolved_channels_in_rec:
            ch_mask = rec_resolved & (channel_ids == ch_id)
            ch_vecs = normalized_vectors[ch_mask]
            centroid = ch_vecs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm
            centroids.append(centroid)
            phys_positions.append(positions_3d[str(ch_id)])

        centroids_arr = np.array(centroids)
        phys_arr = np.array(phys_positions)

        rho = _spearman_distance_correlation(centroids_arr, phys_arr)
        if rho is not None:
            centroid_spearman_values.append(rho)
        else:
            n_undefined += 1

        if channel_mode == "dynamic":
            rec_windows = np.unique(window_ids[rec_resolved])
            for win_id in rec_windows:
                win_mask = rec_resolved & (window_ids == win_id)
                win_channels = np.unique(channel_ids[win_mask])
                win_resolved = [
                    ch for ch in win_channels if str(ch) in positions_3d
                ]
                if len(win_resolved) < min_positioned:
                    continue
                n_eligible_windows += 1

                win_vecs = []
                win_phys = []
                for ch_id in win_resolved:
                    ch_mask = win_mask & (channel_ids == ch_id)
                    ch_vecs = normalized_vectors[ch_mask]
                    win_vecs.append(ch_vecs.mean(axis=0))
                    win_phys.append(positions_3d[str(ch_id)])

                win_vecs_arr = np.array(win_vecs)
                win_phys_arr = np.array(win_phys)
                rho = _spearman_distance_correlation(win_vecs_arr, win_phys_arr)
                if rho is not None:
                    window_spearman_values.append(rho)
                else:
                    n_undefined += 1

    centroid_result = _median_iqr(centroid_spearman_values)
    window_result = _median_iqr(window_spearman_values)

    return AnatomyResult(
        centroid_spearman=centroid_result[0],
        window_spearman=window_result[0],
        centroid_iqr=centroid_result[1],
        window_iqr=window_result[1],
        n_eligible_recordings=n_eligible_recordings,
        n_eligible_windows=n_eligible_windows,
        n_resolved_channels=n_resolved,
        n_undefined=n_undefined,
    )


def _spearman_distance_correlation(
    representation_vectors: np.ndarray,
    physical_positions: np.ndarray,
) -> float | None:
    """Spearman correlation between pairwise cosine and physical distances.

    Returns None if either distance vector is constant (correlation undefined).
    Physical distances are angular/geodesic distances after normalizing the
    canonical coordinates to the scalp sphere.
    """
    n = len(representation_vectors)
    if n < 2:
        return None

    position_norms = np.linalg.norm(physical_positions, axis=1)
    if np.any(~np.isfinite(position_norms)) or np.any(position_norms == 0):
        return None
    unit_positions = physical_positions / position_norms[:, np.newaxis]

    cos_dist = []
    phys_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_d = 1.0 - float(
                representation_vectors[i] @ representation_vectors[j]
            )
            cos_dist.append(max(cos_d, 0.0))
            cosine_angle = float(unit_positions[i] @ unit_positions[j])
            phys_d = float(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            phys_dist.append(phys_d)

    cos_dist = np.array(cos_dist)
    phys_dist = np.array(phys_dist)

    if np.std(cos_dist) == 0 or np.std(phys_dist) == 0:
        return None

    rho, _ = scipy_stats.spearmanr(phys_dist, cos_dist)
    return float(rho)


def _median_iqr(values: list[float]) -> tuple[float | None, float | None]:
    """Compute median and IQR, returning None for empty input."""
    if not values:
        return None, None
    arr = np.array(values)
    median = float(np.median(arr))
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return median, q3 - q1


# ---------------------------------------------------------------------------
# Electrode name normalization (Section 9.2)
# ---------------------------------------------------------------------------

_VETTED_ALIASES: dict[str, str] = {
    "t3": "t7",
    "t4": "t8",
    "t5": "p7",
    "t6": "p8",
}


def normalize_electrode_name(namespaced_id: str) -> str:
    """Conservative canonical electrode name from a namespaced channel ID.

    Strips namespace components, normalizes case and whitespace, and applies
    only explicitly vetted aliases. Does not collapse bipolar, ambiguous,
    ECoG, SEEG, or dataset-specific channel names.

    Examples:
        >>> normalize_electrode_name("dataset_a/sub-01/Fp1")
        'fp1'
        >>> normalize_electrode_name("  T3 ")
        't7'
        >>> normalize_electrode_name("ECoG-LAH1")
        'ecog-lah1'
    """
    parts = namespaced_id.split("/")
    bare = parts[-1].strip().lower()
    bare = re.sub(r"\s+", "", bare)
    return _VETTED_ALIASES.get(bare, bare)


def get_electrode_positions_3d(
    montage_names: Sequence[str] = ("standard_1020", "standard_1005"),
) -> dict[str, np.ndarray]:
    """Canonical 3D positions from MNE standard montages.

    Returns ``{lowercase_name: np.ndarray(3,)}`` in MNE head coordinates
    (meters). Only includes channels that can be resolved reliably.

    The returned positions are canonical inferred positions, not measured
    recording-specific coordinates.
    """
    try:
        import mne

        mne.set_log_level("ERROR")
    except ImportError:
        log.info("MNE not installed — anatomical metrics unavailable.")
        return {}

    positions: dict[str, np.ndarray] = {}
    for montage_name in montage_names:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            for ch, xyz in montage.get_positions()["ch_pos"].items():
                key = ch.lower()
                if key not in positions:
                    positions[key] = np.array(
                        [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                    )
        except Exception:
            continue

    for old, new in _VETTED_ALIASES.items():
        if new in positions and old not in positions:
            positions[old] = positions[new]

    return positions


def resolve_channel_positions(
    channel_ids: np.ndarray,
    positions_3d: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Resolve channel IDs to 3D positions using canonical montages.

    Args:
        channel_ids: Channel identifier per observation.
        positions_3d: Canonical position lookup from :func:`get_electrode_positions_3d`.

    Returns:
        Tuple of (resolved positions dict, n_resolved, n_unresolved) where
        the dict maps normalized electrode name to 3D position for channels
        present in the observations.
    """
    unique_channels = np.unique(channel_ids)
    resolved = {}
    n_unresolved = 0
    for ch in unique_channels:
        normalized = normalize_electrode_name(str(ch))
        if normalized in positions_3d:
            resolved[str(ch)] = positions_3d[normalized]
        else:
            n_unresolved += 1
    return resolved, len(resolved), n_unresolved


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------


def compute_channel_metrics(
    normalized_vectors: np.ndarray,
    recording_ids: np.ndarray,
    channel_ids: np.ndarray,
    channel_mode: str,
    window_ids: np.ndarray | None = None,
    positions_3d: dict[str, np.ndarray] | None = None,
    min_positioned: int = 9,
) -> dict[str, object]:
    """Compute all channel metrics in a single call.

    Returns a flat dictionary with stable keys matching the W&B output
    contract (Section 13). Values are metric result dataclasses.

    Args:
        normalized_vectors: L2-normalized channel vectors, shape ``(N, D)``.
        recording_ids: Recording/session identifier per observation.
        channel_ids: Channel identifier per observation.
        channel_mode: ``"static"`` or ``"dynamic"``.
        window_ids: Optional window identifier per observation for per-window
            anatomical scores.
        positions_3d: Optional 3D position lookup for anatomical metrics.
        min_positioned: Minimum resolved channels for anatomical eligibility.

    Returns:
        Dictionary with keys ``temporal_consistency``, ``separability``,
        ``canonical_consistency``, and ``anatomy``.
    """
    canonical_labels = np.array(
        [normalize_electrode_name(str(ch)) for ch in channel_ids]
    )

    temporal = channel_temporal_consistency(
        normalized_vectors, recording_ids, channel_ids, channel_mode
    )
    separability = channel_within_recording_separability(
        normalized_vectors, recording_ids, channel_ids, channel_mode
    )
    canonical = channel_canonical_consistency(
        normalized_vectors, recording_ids, canonical_labels, channel_ids
    )

    if positions_3d and window_ids is not None:
        anatomy = channel_anatomical_scores_with_windows(
            normalized_vectors,
            recording_ids,
            canonical_labels,
            window_ids,
            positions_3d,
            channel_mode,
            min_positioned,
        )
    elif positions_3d:
        anatomy = channel_anatomical_scores(
            normalized_vectors,
            recording_ids,
            canonical_labels,
            positions_3d,
            channel_mode,
            min_positioned,
        )
    else:
        anatomy = AnatomyResult(
            centroid_spearman=None,
            window_spearman=None,
            centroid_iqr=None,
            window_iqr=None,
            n_eligible_recordings=0,
            n_eligible_windows=0,
            n_resolved_channels=0,
            n_undefined=0,
        )

    return {
        "temporal_consistency": temporal,
        "separability": separability,
        "canonical_consistency": canonical,
        "anatomy": anatomy,
    }


def format_channel_metrics_for_logging(
    metrics: dict[str, object],
) -> dict[str, float | int]:
    """Flatten channel metrics into W&B-compatible scalar dict.

    Keys follow the ``val/embedding_viz/channel/`` namespace from the output
    contract (Section 13).
    """
    prefix = "val/embedding_viz/channel"
    result: dict[str, float | int] = {}

    tc: TemporalConsistencyResult = metrics["temporal_consistency"]  # type: ignore[assignment]
    if tc.score is not None:
        result[f"{prefix}/temporal_consistency"] = tc.score
    result[f"{prefix}/temporal_consistency/n_recordings"] = tc.n_recordings
    result[f"{prefix}/temporal_consistency/n_channels"] = tc.n_channels
    result[f"{prefix}/temporal_consistency/n_observations"] = tc.n_observations
    result[f"{prefix}/temporal_consistency/is_static"] = int(tc.is_static)

    sep: SeparabilityResult = metrics["separability"]  # type: ignore[assignment]
    if sep.accuracy is not None:
        result[f"{prefix}/within_recording_accuracy"] = sep.accuracy
    if sep.margin is not None:
        result[f"{prefix}/within_recording_margin"] = sep.margin
    result[f"{prefix}/within_recording/n_recordings"] = sep.n_recordings
    result[f"{prefix}/within_recording/n_channels"] = sep.n_channels
    result[f"{prefix}/within_recording/n_observations"] = sep.n_observations
    if sep.unavailable_reason:
        result[f"{prefix}/within_recording/unavailable"] = 1

    can: CanonicalConsistencyResult = metrics["canonical_consistency"]  # type: ignore[assignment]
    if can.accuracy is not None:
        result[f"{prefix}/canonical_accuracy"] = can.accuracy
    if can.margin is not None:
        result[f"{prefix}/canonical_margin"] = can.margin
    result[f"{prefix}/canonical/n_electrodes"] = can.n_electrodes
    result[f"{prefix}/canonical/n_centroids"] = can.n_centroids
    result[f"{prefix}/canonical/n_excluded_electrodes"] = (
        can.n_excluded_electrodes
    )

    anat: AnatomyResult = metrics["anatomy"]  # type: ignore[assignment]
    if anat.centroid_spearman is not None:
        result[f"{prefix}/anatomy_centroid_spearman"] = anat.centroid_spearman
    if anat.window_spearman is not None:
        result[f"{prefix}/anatomy_window_spearman"] = anat.window_spearman
    if anat.centroid_iqr is not None:
        result[f"{prefix}/anatomy_centroid_iqr"] = anat.centroid_iqr
    if anat.window_iqr is not None:
        result[f"{prefix}/anatomy_window_iqr"] = anat.window_iqr
    result[f"{prefix}/anatomy/n_eligible_recordings"] = (
        anat.n_eligible_recordings
    )
    result[f"{prefix}/anatomy/n_eligible_windows"] = anat.n_eligible_windows
    result[f"{prefix}/anatomy/n_resolved_channels"] = anat.n_resolved_channels
    result[f"{prefix}/anatomy/n_undefined"] = anat.n_undefined

    return result


def format_backbone_silhouettes_for_logging(
    silhouettes: dict[str, SilhouetteResult],
) -> dict[str, float | int]:
    """Flatten backbone silhouette results into W&B-compatible scalar dict.

    Keys follow the ``val/embedding_viz/backbone/silhouette/`` namespace.
    """
    prefix = "val/embedding_viz/backbone/silhouette"
    result: dict[str, float | int] = {}

    for name, sil in silhouettes.items():
        key = f"{prefix}/{name}"
        if sil.score is not None:
            result[key] = sil.score
        result[f"{key}/n_samples"] = sil.n_samples
        result[f"{key}/n_excluded"] = sil.n_excluded
        result[f"{key}/n_groups"] = sil.n_groups
        result[f"{key}/n_excluded_groups"] = sil.n_excluded_groups

    return result
