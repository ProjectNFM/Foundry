"""Deterministic observation selection for embedding visualization.

Implements stable identities, hierarchical window allocation, channel-observation
budgets, and distributed aggregation. All functions are pure and testable without
Lightning or W&B.

The selected global observation set is invariant to batch order, worker count,
device speed, and distributed partitioning.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationIdentity:
    """Stable identity for a single validation window.

    Uses explicit metadata fields that are serializable and stable across
    processes. Does not use Python's randomized ``hash()``.
    """

    dataset_id: str
    subject_id: str
    session_id: str
    absolute_start: float
    window_duration: float

    def to_bytes(self) -> bytes:
        """Canonical byte representation for hashing."""
        return (
            self.dataset_id.encode("utf-8")
            + b"\x00"
            + self.subject_id.encode("utf-8")
            + b"\x00"
            + self.session_id.encode("utf-8")
            + b"\x00"
            + struct.pack("<dd", self.absolute_start, self.window_duration)
        )


def stable_hash(identity: ObservationIdentity, seed: int) -> int:
    """Process-stable hash combining an observation identity with a seed.

    Uses SHA-256 truncated to 64 bits for deterministic ordering that is
    independent of Python's hash randomization.
    """
    h = hashlib.sha256()
    h.update(struct.pack("<q", seed))
    h.update(identity.to_bytes())
    return int.from_bytes(h.digest()[:8], "little")


def stable_key_hash(key: str, seed: int) -> int:
    """Process-stable hash for a string key (dataset/subject/session) with seed."""
    h = hashlib.sha256()
    h.update(struct.pack("<q", seed))
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")


def compute_fingerprint(identities: list[ObservationIdentity]) -> str:
    """Compute a compact fingerprint from sorted selected identities.

    The same fingerprint is required before treating two plots as paired
    samples across runs or events.
    """
    h = hashlib.sha256()
    for identity in sorted(identities, key=lambda x: x.to_bytes()):
        h.update(identity.to_bytes())
    return h.hexdigest()[:16]


def compute_window_budget(total_windows: int, config: "SelectionConfig") -> int:
    """Window budget per Section 7.1 of the implementation plan.

    ``min(N, max(min_windows, ceil(fraction * N)), max_windows)``
    """
    if total_windows <= 0:
        return 0
    scaled = math.ceil(config.window_fraction * total_windows)
    budget = min(
        total_windows, max(config.min_windows, scaled), config.max_windows
    )
    return budget


@dataclass
class SelectionConfig:
    """Configuration for the observation selection algorithm."""

    seed: int = 42
    window_fraction: float = 0.10
    min_windows: int = 256
    max_windows: int = 2048
    max_channel_observations: int = 16384
    max_sessions_per_dataset: int = 8
    min_windows_per_session: int = 16
    max_recording_panels: int = 8


@dataclass
class SelectedObservations:
    """Result of the hierarchical selection process.

    Contains the selected window identities, their indices into the original
    population, and the fingerprint for cross-run verification.
    """

    window_identities: list[ObservationIdentity]
    window_indices: list[int]
    fingerprint: str
    channel_window_indices: list[int] = field(default_factory=list)
    channel_observation_count: int = 0

    @property
    def window_count(self) -> int:
        return len(self.window_identities)


def hierarchical_select_windows(
    identities: list[ObservationIdentity],
    config: SelectionConfig,
) -> SelectedObservations:
    """Select windows using deterministic hierarchical allocation.

    Allocation strategy (Section 7.2):
    1. Balance allocation across datasets before considering dataset size.
    2. Within each dataset, maximize subject diversity.
    3. Select at most ``max_sessions_per_dataset`` sessions per dataset.
    4. Target at least ``min_windows_per_session`` windows per selected session.
    5. When budget cannot support depth, reduce sessions before windows per session.
    6. When a small set has fewer observations than quota, include all and
       redistribute unused capacity.

    Uses stable hashes for all selection decisions—never arrival order.
    """
    if not identities:
        return SelectedObservations(
            window_identities=[],
            window_indices=[],
            fingerprint=compute_fingerprint([]),
        )

    budget = compute_window_budget(len(identities), config)
    if budget >= len(identities):
        indices = list(range(len(identities)))
        return SelectedObservations(
            window_identities=list(identities),
            window_indices=indices,
            fingerprint=compute_fingerprint(identities),
        )

    indexed = list(enumerate(identities))

    datasets: dict[str, list[tuple[int, ObservationIdentity]]] = {}
    for idx, ident in indexed:
        datasets.setdefault(ident.dataset_id, []).append((idx, ident))

    sorted_datasets = sorted(
        datasets.keys(), key=lambda d: stable_key_hash(d, config.seed)
    )
    n_datasets = len(sorted_datasets)

    per_dataset_budget = _distribute_budget(budget, n_datasets)

    selected_indices: list[int] = []

    for ds_idx, dataset_id in enumerate(sorted_datasets):
        ds_items = datasets[dataset_id]
        ds_budget = per_dataset_budget[ds_idx]
        if ds_budget == 0:
            continue

        ds_selected = _select_within_dataset(ds_items, ds_budget, config)
        selected_indices.extend(ds_selected)

    remaining = budget - len(selected_indices)
    if remaining > 0:
        selected_set = set(selected_indices)
        selected_sessions = {identities[i].session_id for i in selected_indices}
        unselected = [
            (idx, ident)
            for idx, ident in indexed
            if idx not in selected_set and ident.session_id in selected_sessions
        ]
        unselected.sort(key=lambda x: stable_hash(x[1], config.seed))
        selected_indices.extend(idx for idx, _ in unselected[:remaining])

    selected_indices.sort(
        key=lambda idx: stable_hash(identities[idx], config.seed)
    )

    selected_identities = [identities[i] for i in selected_indices]
    fingerprint = compute_fingerprint(selected_identities)

    return SelectedObservations(
        window_identities=selected_identities,
        window_indices=selected_indices,
        fingerprint=fingerprint,
    )


def _distribute_budget(total_budget: int, n_groups: int) -> list[int]:
    """Distribute budget evenly across groups, remainder goes to first groups."""
    if n_groups == 0:
        return []
    base = total_budget // n_groups
    remainder = total_budget % n_groups
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def _select_within_dataset(
    items: list[tuple[int, ObservationIdentity]],
    budget: int,
    config: SelectionConfig,
) -> list[int]:
    """Select windows within a single dataset respecting session and subject diversity.

    If the initially selected sessions cannot fill the budget, iteratively adds
    more sessions (up to the cap) rather than breaking session constraints.
    """
    if budget >= len(items):
        return [idx for idx, _ in items]

    subjects: dict[str, list[tuple[int, ObservationIdentity]]] = {}
    for idx, ident in items:
        subjects.setdefault(ident.subject_id, []).append((idx, ident))

    sessions: dict[str, list[tuple[int, ObservationIdentity]]] = {}
    for idx, ident in items:
        sessions.setdefault(ident.session_id, []).append((idx, ident))

    sorted_sessions = sorted(
        sessions.keys(), key=lambda s: stable_key_hash(s, config.seed)
    )

    max_sess = config.max_sessions_per_dataset
    min_win = config.min_windows_per_session

    n_sessions_that_fit = max(1, budget // max(min_win, 1))
    n_selected_sessions = min(
        max_sess, len(sorted_sessions), n_sessions_that_fit
    )

    selected_sessions = _select_sessions_by_subject_diversity(
        sorted_sessions, sessions, subjects, n_selected_sessions, config
    )

    selected: list[int] = []
    _fill_from_sessions(selected, selected_sessions, sessions, budget, config)

    if len(selected) < budget and len(selected_sessions) < min(
        max_sess, len(sorted_sessions)
    ):
        remaining_sessions = [
            s for s in sorted_sessions if s not in set(selected_sessions)
        ]
        additional_needed = min(
            max_sess - len(selected_sessions), len(remaining_sessions)
        )
        extra_sessions = _select_sessions_by_subject_diversity(
            remaining_sessions, sessions, subjects, additional_needed, config
        )
        _fill_from_sessions(selected, extra_sessions, sessions, budget, config)

    return selected[:budget]


def _fill_from_sessions(
    selected: list[int],
    session_list: list[str],
    sessions: dict[str, list[tuple[int, ObservationIdentity]]],
    budget: int,
    config: SelectionConfig,
) -> None:
    """Fill selected indices from the given sessions up to budget."""
    if not session_list:
        return
    remaining = budget - len(selected)
    if remaining <= 0:
        return

    already_selected = set(selected)
    per_session_budget = _distribute_budget(remaining, len(session_list))

    for sess_idx, session_id in enumerate(session_list):
        sess_items = [
            (idx, ident)
            for idx, ident in sessions[session_id]
            if idx not in already_selected
        ]
        sess_budget = per_session_budget[sess_idx]
        if sess_budget >= len(sess_items):
            for idx, _ in sess_items:
                selected.append(idx)
        else:
            sess_items_sorted = sorted(
                sess_items, key=lambda x: stable_hash(x[1], config.seed)
            )
            for idx, _ in sess_items_sorted[:sess_budget]:
                selected.append(idx)


def _select_sessions_by_subject_diversity(
    sorted_sessions: list[str],
    sessions: dict[str, list[tuple[int, ObservationIdentity]]],
    subjects: dict[str, list[tuple[int, ObservationIdentity]]],
    n_select: int,
    config: SelectionConfig,
) -> list[str]:
    """Select sessions maximizing subject diversity.

    Prefers sessions from distinct subjects. When subjects are exhausted,
    falls back to stable hash ordering.
    """
    if n_select >= len(sorted_sessions):
        return sorted_sessions

    session_to_subject: dict[str, str] = {}
    for session_id, items in sessions.items():
        session_to_subject[session_id] = items[0][1].subject_id

    sorted_subjects = sorted(
        subjects.keys(), key=lambda s: stable_key_hash(s, config.seed)
    )

    selected: list[str] = []
    used_subjects: set[str] = set()

    for subject_id in sorted_subjects:
        if len(selected) >= n_select:
            break
        subject_sessions = [
            s
            for s in sorted_sessions
            if session_to_subject.get(s) == subject_id and s not in selected
        ]
        if subject_sessions:
            selected.append(subject_sessions[0])
            used_subjects.add(subject_id)

    if len(selected) < n_select:
        remaining_sessions = [s for s in sorted_sessions if s not in selected]
        remaining_sessions.sort(key=lambda s: stable_key_hash(s, config.seed))
        selected.extend(remaining_sessions[: n_select - len(selected)])

    return selected[:n_select]


def select_channel_observations(
    selected_window_indices: list[int],
    identities: list[ObservationIdentity],
    channel_counts: list[int],
    config: SelectionConfig,
) -> list[int]:
    """Select windows for channel-observation capture respecting the budget.

    Args:
        selected_window_indices: Indices of windows already selected by
            hierarchical allocation.
        identities: Stable identities for the full window population.
        channel_counts: Number of valid channels per window (same indexing as
            the full population).
        config: Selection configuration.

    Returns:
        Subset of ``selected_window_indices`` whose complete channel sets fit
        within ``max_channel_observations``. Windows are admitted whole (no
        partial channel sets) and balanced across sessions.

    Admits complete channel sets so within-window geometry remains valid.
    Stops before admitting a window that would exceed the cap.
    """
    if not selected_window_indices:
        return []

    cap = config.max_channel_observations
    if cap <= 0:
        return []

    if len(identities) != len(channel_counts):
        raise ValueError(
            "identities and channel_counts must describe the same population"
        )

    session_groups: dict[str, list[int]] = {}
    for win_idx in selected_window_indices:
        sess = identities[win_idx].session_id
        session_groups.setdefault(sess, []).append(win_idx)

    for sess_windows in session_groups.values():
        sess_windows.sort(
            key=lambda idx: (
                stable_hash(identities[idx], config.seed),
                identities[idx].to_bytes(),
            )
        )

    sorted_session_keys = sorted(
        session_groups.keys(), key=lambda s: stable_key_hash(s, config.seed)
    )

    admitted: list[int] = []
    total_channels = 0

    session_cursors = {sess: 0 for sess in sorted_session_keys}
    exhausted = set()

    while len(exhausted) < len(sorted_session_keys):
        for sess in sorted_session_keys:
            if sess in exhausted:
                continue
            cursor = session_cursors[sess]
            windows = session_groups[sess]
            if cursor >= len(windows):
                exhausted.add(sess)
                continue

            win_idx = windows[cursor]
            ch_count = channel_counts[win_idx]

            if total_channels + ch_count > cap:
                exhausted.add(sess)
                continue

            admitted.append(win_idx)
            total_channels += ch_count
            session_cursors[sess] = cursor + 1

            if total_channels >= cap:
                return admitted

    return admitted


# ---------------------------------------------------------------------------
# Distributed aggregation utilities
# ---------------------------------------------------------------------------


@dataclass
class RankObservations:
    """Observations captured on a single rank for aggregation."""

    identities: list[ObservationIdentity]
    backbone_representations: torch.Tensor | None = None
    channel_representations: torch.Tensor | None = None
    channel_indices: torch.Tensor | None = None
    channel_masks: torch.Tensor | None = None
    channel_counts: list[int] = field(default_factory=list)
    target_values: dict[str, torch.Tensor] = field(default_factory=dict)


def gather_and_deduplicate(
    local: RankObservations,
    world_size: int = 1,
    rank: int = 0,
) -> RankObservations | None:
    """Gather observations across ranks and deduplicate by identity.

    In single-GPU mode (world_size=1), returns the local observations directly.
    In distributed mode, gathers identities and tensors, deduplicates by stable
    identity bytes, and returns the merged result only on rank 0.

    Args:
        local: Observations captured on this rank.
        world_size: Total number of distributed processes.
        rank: This process's global rank.

    Returns:
        Merged observations on rank 0, None on other ranks.
    """
    if world_size <= 1:
        return local

    import torch.distributed as dist

    # Object collectives preserve the variable-length identity strings and keep
    # every per-window payload aligned with it.  In contrast, manually padded
    # CPU tensors fail with NCCL and cannot reconstruct a remote identity.
    gathered: list[RankObservations | None] = [None] * world_size
    dist.all_gather_object(gathered, local)

    if rank != 0:
        return None

    return _merge_rank_observations(
        [observation for observation in gathered if observation is not None]
    )


def _merge_rank_observations(
    observations: list[RankObservations],
) -> RankObservations:
    """Merge rank payloads by identity while preserving aligned fields.

    This is intentionally pure so the identity/payload alignment can be tested
    without starting a distributed process group.
    """
    if not observations:
        return RankObservations(identities=[])

    target_keys = set(observations[0].target_values)
    for observation in observations:
        _validate_rank_observations(observation, target_keys)

    rows: list[tuple[ObservationIdentity, RankObservations, int]] = []
    seen: set[bytes] = set()
    for observation in observations:
        for row_index, identity in enumerate(observation.identities):
            identity_key = identity.to_bytes()
            if identity_key in seen:
                continue
            seen.add(identity_key)
            rows.append((identity, observation, row_index))

    rows.sort(key=lambda row: row[0].to_bytes())
    merged = RankObservations(
        identities=[identity for identity, _, _ in rows],
        channel_counts=[
            source.channel_counts[i] if source.channel_counts else 0
            for _, source, i in rows
        ],
        target_values={
            key: torch.stack(
                [source.target_values[key][i] for _, source, i in rows]
            )
            for key in target_keys
        },
    )
    merged.backbone_representations = _stack_optional_rows(
        rows, "backbone_representations"
    )
    merged.channel_representations = _stack_optional_rows(
        rows, "channel_representations"
    )
    merged.channel_indices = _stack_optional_rows(rows, "channel_indices")
    merged.channel_masks = _stack_optional_rows(rows, "channel_masks")
    return merged


def _validate_rank_observations(
    observation: RankObservations, expected_target_keys: set[str]
) -> None:
    """Ensure every payload has one leading row per identity."""
    n_rows = len(observation.identities)
    if len(observation.channel_counts) not in (0, n_rows):
        raise ValueError("channel_counts must have one entry per identity")
    for name in (
        "backbone_representations",
        "channel_representations",
        "channel_indices",
        "channel_masks",
    ):
        value = getattr(observation, name)
        if value is not None and value.shape[0] != n_rows:
            raise ValueError(f"{name} must have one row per identity")
    if set(observation.target_values) != expected_target_keys:
        raise ValueError("target_values keys must match across ranks")
    for name, value in observation.target_values.items():
        if value.shape[0] != n_rows:
            raise ValueError(
                f"target_values[{name!r}] must have one row per identity"
            )


def _stack_optional_rows(
    rows: list[tuple[ObservationIdentity, RankObservations, int]], name: str
) -> torch.Tensor | None:
    """Stack an optional payload, rejecting inconsistent availability."""
    values = [getattr(source, name) for _, source, _ in rows]
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} must be present on every rank")
    row_tensors = [getattr(source, name)[i] for _, source, i in rows]
    try:
        return torch.stack(row_tensors)
    except RuntimeError as error:
        # Channel tensors have shape (C, ...) per window and C may differ between
        # ranks. Preserve all values by padding C; channel_counts identifies the
        # valid prefix of each resulting row.
        if not all(
            value.ndim >= 1 and value.shape[1:] == row_tensors[0].shape[1:]
            for value in row_tensors
        ):
            raise ValueError(
                f"{name} has incompatible shapes across ranks"
            ) from error
        max_channels = max(value.shape[0] for value in row_tensors)
        padded = row_tensors[0].new_zeros(
            (len(row_tensors), max_channels, *row_tensors[0].shape[1:])
        )
        for row_index, value in enumerate(row_tensors):
            padded[row_index, : value.shape[0]] = value
        return padded


def gather_identities_and_select(
    local_identities: list[ObservationIdentity],
    local_indices: list[int],
    config: SelectionConfig,
    world_size: int = 1,
    rank: int = 0,
) -> SelectedObservations | None:
    """Distributed-aware selection: gather identities, select globally on rank 0.

    In single-GPU mode, performs selection directly. In distributed mode,
    gathers all identities to rank 0 and runs hierarchical selection there.

    Args:
        local_identities: Identities captured on this rank.
        local_indices: Original indices corresponding to local identities.
        config: Selection configuration.
        world_size: Total number of distributed processes.
        rank: This process's global rank.

    Returns:
        SelectedObservations on rank 0 (or single-GPU), None on other ranks.
    """
    if world_size <= 1:
        return hierarchical_select_windows(local_identities, config)

    import torch.distributed as dist

    if len(local_identities) != len(local_indices):
        raise ValueError(
            "local_identities and local_indices must have equal length"
        )

    local_pairs = list(zip(local_identities, local_indices))
    gathered: list[list[tuple[ObservationIdentity, int]] | None] = [
        None
    ] * world_size
    dist.all_gather_object(gathered, local_pairs)

    if rank != 0:
        return None

    all_identities: list[ObservationIdentity] = []
    seen: set[bytes] = set()

    for pairs in gathered:
        if pairs is None:
            continue
        for ident, _idx in pairs:
            key = ident.to_bytes()
            if key not in seen:
                seen.add(key)
                all_identities.append(ident)

    return hierarchical_select_windows(all_identities, config)


def build_identities_from_metadata(
    dataset_ids: list[str],
    subject_ids: list[str],
    session_ids: list[str],
    absolute_starts: torch.Tensor | np.ndarray,
    window_durations: torch.Tensor | np.ndarray,
) -> list[ObservationIdentity]:
    """Construct ObservationIdentity list from batch metadata arrays.

    This bridges the Phase 2 SampleMetadata contract to Phase 3 identities.
    """
    if isinstance(absolute_starts, torch.Tensor):
        absolute_starts = absolute_starts.cpu().numpy()
    if isinstance(window_durations, torch.Tensor):
        window_durations = window_durations.cpu().numpy()

    n = len(dataset_ids)
    return [
        ObservationIdentity(
            dataset_id=dataset_ids[i],
            subject_id=subject_ids[i],
            session_id=session_ids[i],
            absolute_start=float(absolute_starts[i]),
            window_duration=float(window_durations[i]),
        )
        for i in range(n)
    ]
