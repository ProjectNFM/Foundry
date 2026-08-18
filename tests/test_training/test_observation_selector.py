"""Tests for the deterministic observation selector (Phase 3).

Covers:
- Stable hash determinism and process independence
- Fingerprint stability across orderings
- Window budget boundary cases
- Hierarchical allocation, redistribution, and session-depth guarantees
- Channel cap with complete-window admission
- Batch-order invariance
- Distributed partitioning invariance (simulated)
- Integration with Phase 2 SampleMetadata
"""

from __future__ import annotations

import random

import numpy as np
import torch

from foundry.training.callbacks.observation_selector import (
    ObservationIdentity,
    RankObservations,
    SelectionConfig,
    _merge_rank_observations,
    build_identities_from_metadata,
    compute_fingerprint,
    compute_window_budget,
    gather_and_deduplicate,
    hierarchical_select_windows,
    select_channel_observations,
    stable_hash,
    stable_key_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity(
    dataset: str = "ds_a",
    subject: str = "sub_01",
    session: str = "sess_001",
    start: float = 0.0,
    duration: float = 2.0,
) -> ObservationIdentity:
    return ObservationIdentity(
        dataset_id=dataset,
        subject_id=subject,
        session_id=session,
        absolute_start=start,
        window_duration=duration,
    )


def _make_population(
    n_datasets: int = 2,
    n_subjects_per: int = 3,
    n_sessions_per: int = 2,
    n_windows_per: int = 10,
) -> list[ObservationIdentity]:
    """Generate a structured population of identities."""
    identities = []
    for d in range(n_datasets):
        for s in range(n_subjects_per):
            for sess in range(n_sessions_per):
                for w in range(n_windows_per):
                    identities.append(
                        ObservationIdentity(
                            dataset_id=f"dataset_{d}",
                            subject_id=f"dataset_{d}/subject_{s}",
                            session_id=f"dataset_{d}/subject_{s}/session_{sess}",
                            absolute_start=float(w * 2),
                            window_duration=2.0,
                        )
                    )
    return identities


# ---------------------------------------------------------------------------
# Stable hash tests
# ---------------------------------------------------------------------------


class TestStableHash:
    def test_deterministic_across_calls(self):
        ident = _make_identity()
        h1 = stable_hash(ident, seed=42)
        h2 = stable_hash(ident, seed=42)
        assert h1 == h2

    def test_different_seeds_produce_different_hashes(self):
        ident = _make_identity()
        h1 = stable_hash(ident, seed=42)
        h2 = stable_hash(ident, seed=99)
        assert h1 != h2

    def test_different_identities_produce_different_hashes(self):
        id1 = _make_identity(start=0.0)
        id2 = _make_identity(start=2.0)
        assert stable_hash(id1, seed=42) != stable_hash(id2, seed=42)

    def test_hash_is_integer(self):
        ident = _make_identity()
        h = stable_hash(ident, seed=42)
        assert isinstance(h, int)

    def test_key_hash_deterministic(self):
        h1 = stable_key_hash("dataset_0", seed=42)
        h2 = stable_key_hash("dataset_0", seed=42)
        assert h1 == h2

    def test_key_hash_different_keys(self):
        h1 = stable_key_hash("dataset_0", seed=42)
        h2 = stable_key_hash("dataset_1", seed=42)
        assert h1 != h2

    def test_hash_does_not_depend_on_python_hash_seed(self):
        """stable_hash uses SHA-256, not Python's hash()."""
        ident = _make_identity()
        h = stable_hash(ident, seed=42)
        assert h == stable_hash(ident, seed=42)


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_same_identities_same_fingerprint(self):
        ids = [_make_identity(start=float(i)) for i in range(5)]
        fp1 = compute_fingerprint(ids)
        fp2 = compute_fingerprint(ids)
        assert fp1 == fp2

    def test_order_independent(self):
        ids = [_make_identity(start=float(i)) for i in range(5)]
        fp1 = compute_fingerprint(ids)
        fp2 = compute_fingerprint(list(reversed(ids)))
        assert fp1 == fp2

    def test_different_sets_different_fingerprints(self):
        ids1 = [_make_identity(start=float(i)) for i in range(5)]
        ids2 = [_make_identity(start=float(i)) for i in range(1, 6)]
        assert compute_fingerprint(ids1) != compute_fingerprint(ids2)

    def test_fingerprint_is_hex_string(self):
        ids = [_make_identity()]
        fp = compute_fingerprint(ids)
        assert len(fp) == 16
        int(fp, 16)  # should not raise

    def test_empty_list_has_stable_fingerprint(self):
        fp1 = compute_fingerprint([])
        fp2 = compute_fingerprint([])
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# Window budget tests
# ---------------------------------------------------------------------------


class TestWindowBudget:
    def test_empty_population(self):
        config = SelectionConfig()
        assert compute_window_budget(0, config) == 0

    def test_small_population_takes_all(self):
        config = SelectionConfig(min_windows=256)
        assert compute_window_budget(100, config) == 100

    def test_medium_population_uses_fraction(self):
        config = SelectionConfig(
            window_fraction=0.10, min_windows=256, max_windows=2048
        )
        budget = compute_window_budget(5000, config)
        assert budget == 500

    def test_large_population_capped(self):
        config = SelectionConfig(max_windows=2048)
        budget = compute_window_budget(100_000, config)
        assert budget == 2048

    def test_exactly_at_min(self):
        config = SelectionConfig(
            window_fraction=0.10, min_windows=256, max_windows=2048
        )
        budget = compute_window_budget(256, config)
        assert budget == 256

    def test_fraction_below_min_uses_min(self):
        config = SelectionConfig(
            window_fraction=0.10, min_windows=256, max_windows=2048
        )
        budget = compute_window_budget(1000, config)
        assert budget == 256

    def test_fraction_above_min(self):
        config = SelectionConfig(
            window_fraction=0.10, min_windows=256, max_windows=2048
        )
        budget = compute_window_budget(3000, config)
        assert budget == 300


# ---------------------------------------------------------------------------
# Hierarchical allocation tests
# ---------------------------------------------------------------------------


class TestHierarchicalSelection:
    def test_empty_input(self):
        result = hierarchical_select_windows([], SelectionConfig())
        assert result.window_count == 0
        assert result.fingerprint is not None

    def test_small_population_selects_all(self):
        ids = _make_population(
            n_datasets=1, n_subjects_per=1, n_sessions_per=1, n_windows_per=10
        )
        config = SelectionConfig(min_windows=256)
        result = hierarchical_select_windows(ids, config)
        assert result.window_count == 10
        assert set(result.window_indices) == set(range(10))

    def test_deterministic_across_calls(self):
        ids = _make_population()
        config = SelectionConfig(seed=42, max_windows=50)
        r1 = hierarchical_select_windows(ids, config)
        r2 = hierarchical_select_windows(ids, config)
        assert r1.window_indices == r2.window_indices
        assert r1.fingerprint == r2.fingerprint

    def test_batch_order_invariant(self):
        """Selection is independent of the order identities arrive in."""
        ids = _make_population(
            n_datasets=3, n_subjects_per=4, n_sessions_per=3, n_windows_per=20
        )
        config = SelectionConfig(seed=7, max_windows=100)

        r_ordered = hierarchical_select_windows(ids, config)

        perm = list(range(len(ids)))
        rng = random.Random(999)
        rng.shuffle(perm)
        shuffled_ids = [ids[i] for i in perm]

        r_shuffled = hierarchical_select_windows(shuffled_ids, config)

        selected_set_ordered = set(
            ids[i].to_bytes() for i in r_ordered.window_indices
        )
        selected_set_shuffled = set(
            shuffled_ids[i].to_bytes() for i in r_shuffled.window_indices
        )
        assert selected_set_ordered == selected_set_shuffled
        assert r_ordered.fingerprint == r_shuffled.fingerprint

    def test_balances_across_datasets(self):
        ids = _make_population(
            n_datasets=4, n_subjects_per=2, n_sessions_per=2, n_windows_per=50
        )
        config = SelectionConfig(seed=42, max_windows=80)
        result = hierarchical_select_windows(ids, config)

        dataset_counts: dict[str, int] = {}
        for idx in result.window_indices:
            ds = ids[idx].dataset_id
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        counts = list(dataset_counts.values())
        assert max(counts) - min(counts) <= 1

    def test_maximizes_subject_diversity(self):
        """Within a dataset, sessions from different subjects are preferred."""
        ids = _make_population(
            n_datasets=1, n_subjects_per=8, n_sessions_per=2, n_windows_per=20
        )
        config = SelectionConfig(
            seed=42,
            max_windows=64,
            max_sessions_per_dataset=4,
            min_windows_per_session=16,
        )
        result = hierarchical_select_windows(ids, config)

        subjects_selected = set()
        for idx in result.window_indices:
            subjects_selected.add(ids[idx].subject_id)

        assert len(subjects_selected) >= 4

    def test_respects_max_sessions_per_dataset(self):
        ids = _make_population(
            n_datasets=1, n_subjects_per=5, n_sessions_per=4, n_windows_per=30
        )
        config = SelectionConfig(
            seed=42,
            max_windows=200,
            max_sessions_per_dataset=3,
            min_windows_per_session=16,
        )
        result = hierarchical_select_windows(ids, config)

        sessions_selected = set()
        for idx in result.window_indices:
            sessions_selected.add(ids[idx].session_id)

        assert len(sessions_selected) <= 3

    def test_reduces_sessions_before_windows(self):
        """When budget is tight, reduce sessions rather than windows per session."""
        ids = _make_population(
            n_datasets=1, n_subjects_per=4, n_sessions_per=4, n_windows_per=30
        )
        config = SelectionConfig(
            seed=42,
            max_windows=32,
            max_sessions_per_dataset=8,
            min_windows_per_session=16,
        )
        result = hierarchical_select_windows(ids, config)

        session_window_counts: dict[str, int] = {}
        for idx in result.window_indices:
            sess = ids[idx].session_id
            session_window_counts[sess] = session_window_counts.get(sess, 0) + 1

        n_sessions = len(session_window_counts)
        assert n_sessions <= 2
        for count in session_window_counts.values():
            assert count >= config.min_windows_per_session

    def test_small_population_redistribution(self):
        """When some sessions have fewer windows than quota, redistribute capacity."""
        identities = []
        for w in range(5):
            identities.append(
                _make_identity(
                    dataset="ds_a",
                    subject="sub_0",
                    session="sess_small",
                    start=float(w * 2),
                )
            )
        for w in range(100):
            identities.append(
                _make_identity(
                    dataset="ds_a",
                    subject="sub_1",
                    session="sess_large",
                    start=float(w * 2),
                )
            )
        config = SelectionConfig(
            seed=42, max_windows=60, min_windows_per_session=16
        )
        result = hierarchical_select_windows(identities, config)

        small_count = sum(
            1
            for idx in result.window_indices
            if identities[idx].session_id == "sess_small"
        )
        assert small_count == 5

    def test_different_seeds_give_different_selections(self):
        ids = _make_population(
            n_datasets=2, n_subjects_per=4, n_sessions_per=3, n_windows_per=20
        )
        r1 = hierarchical_select_windows(
            ids, SelectionConfig(seed=1, max_windows=50)
        )
        r2 = hierarchical_select_windows(
            ids, SelectionConfig(seed=2, max_windows=50)
        )
        assert r1.window_indices != r2.window_indices

    def test_fingerprint_matches_content(self):
        ids = _make_population()
        config = SelectionConfig(seed=42, max_windows=50)
        result = hierarchical_select_windows(ids, config)
        expected_fp = compute_fingerprint(result.window_identities)
        assert result.fingerprint == expected_fp


# ---------------------------------------------------------------------------
# Channel observation budget tests
# ---------------------------------------------------------------------------


class TestChannelObservationBudget:
    def test_empty_input(self):
        result = select_channel_observations([], [], [], SelectionConfig())
        assert result == []

    def test_all_fit_under_cap(self):
        selected = [0, 1, 2]
        channel_counts = [10, 10, 10]
        identities = [
            _make_identity(session="s0", start=0.0),
            _make_identity(session="s0", start=2.0),
            _make_identity(session="s1", start=4.0),
        ]
        config = SelectionConfig(max_channel_observations=100)
        result = select_channel_observations(
            selected, identities, channel_counts, config
        )
        assert set(result) == {0, 1, 2}

    def test_stops_before_exceeding_cap(self):
        selected = [0, 1, 2, 3]
        channel_counts = [5000, 5000, 5000, 5000]
        identities = [
            _make_identity(session=f"s{i // 2}", start=float(i))
            for i in selected
        ]
        config = SelectionConfig(max_channel_observations=16384)
        result = select_channel_observations(
            selected, identities, channel_counts, config
        )
        total = sum(channel_counts[i] for i in result)
        assert total <= 16384
        assert len(result) == 3

    def test_admits_complete_windows_only(self):
        """No partial channel sets are admitted."""
        selected = [0, 1, 2]
        channel_counts = [100, 16300, 100]
        identities = [
            _make_identity(session=f"s{i}", start=float(i)) for i in selected
        ]
        config = SelectionConfig(max_channel_observations=16384)
        result = select_channel_observations(
            selected, identities, channel_counts, config
        )
        total = sum(channel_counts[i] for i in result)
        assert total <= 16384

    def test_balances_across_sessions(self):
        """Channel windows are distributed across sessions."""
        selected = list(range(10))
        channel_counts = [100] * 10
        identities = [
            _make_identity(session=f"s{i // 5}", start=float(i))
            for i in selected
        ]
        config = SelectionConfig(max_channel_observations=600)
        result = select_channel_observations(
            selected, identities, channel_counts, config
        )

        s0_count = sum(1 for i in result if identities[i].session_id == "s0")
        s1_count = sum(1 for i in result if identities[i].session_id == "s1")
        assert abs(s0_count - s1_count) <= 1

    def test_large_window_excludes_if_would_exceed(self):
        """A single large window that would exceed cap is skipped."""
        selected = [0, 1, 2]
        channel_counts = [100, 20000, 100]
        identities = [
            _make_identity(session=f"s{i}", start=float(i)) for i in selected
        ]
        config = SelectionConfig(max_channel_observations=16384)
        result = select_channel_observations(
            selected, identities, channel_counts, config
        )
        assert 1 not in result


# ---------------------------------------------------------------------------
# Simulated distributed partitioning tests
# ---------------------------------------------------------------------------


class TestDistributedInvariance:
    def test_single_rank_passthrough(self):
        ids = [_make_identity(start=float(i)) for i in range(5)]
        local = RankObservations(
            identities=ids,
            backbone_representations=torch.randn(5, 16),
        )
        result = gather_and_deduplicate(local, world_size=1, rank=0)
        assert result is not None
        assert len(result.identities) == 5
        assert result.backbone_representations is not None
        assert result.backbone_representations.shape == (5, 16)

    def test_simulated_partitioning_invariance(self):
        """Selection is the same regardless of how identities are partitioned across ranks."""
        ids = _make_population(
            n_datasets=3, n_subjects_per=3, n_sessions_per=2, n_windows_per=15
        )
        config = SelectionConfig(seed=42, max_windows=60)

        full_result = hierarchical_select_windows(ids, config)

        rng = random.Random(123)
        perm = list(range(len(ids)))
        rng.shuffle(perm)
        mid = len(perm) // 2
        rank0_ids = [ids[i] for i in perm[:mid]]
        rank1_ids = [ids[i] for i in perm[mid:]]

        combined = rank0_ids + rank1_ids
        combined_result = hierarchical_select_windows(combined, config)

        assert full_result.fingerprint == combined_result.fingerprint

    def test_merge_preserves_remote_identities_and_payloads(self):
        first = _make_identity(session="s0", start=0.0)
        second = _make_identity(session="s1", start=2.0)
        duplicate_first = _make_identity(session="s0", start=0.0)
        rank0 = RankObservations(
            identities=[second],
            backbone_representations=torch.tensor([[2.0, 20.0]]),
            channel_representations=torch.tensor([[[2.0], [3.0]]]),
            channel_counts=[2],
            target_values={"task": torch.tensor([2])},
        )
        rank1 = RankObservations(
            identities=[first, duplicate_first],
            backbone_representations=torch.tensor([[1.0, 10.0], [9.0, 90.0]]),
            channel_representations=torch.tensor([[[1.0]], [[9.0]]]),
            channel_counts=[1, 1],
            target_values={"task": torch.tensor([1, 9])},
        )

        merged = _merge_rank_observations([rank0, rank1])

        assert merged.identities == [first, second]
        assert merged.backbone_representations is not None
        assert merged.backbone_representations.tolist() == [
            [1.0, 10.0],
            [2.0, 20.0],
        ]
        assert merged.channel_representations is not None
        assert merged.channel_representations.shape == (2, 2, 1)
        assert merged.channel_counts == [1, 2]
        assert merged.target_values["task"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Integration with Phase 2 SampleMetadata
# ---------------------------------------------------------------------------


class TestBuildIdentitiesFromMetadata:
    def test_builds_from_lists_and_tensors(self):
        dataset_ids = ["ds_a", "ds_b", "ds_a"]
        subject_ids = ["sub_0", "sub_1", "sub_2"]
        session_ids = ["sess_0", "sess_1", "sess_2"]
        starts = torch.tensor([0.0, 2.0, 4.0])
        durations = torch.tensor([2.0, 2.0, 2.0])

        identities = build_identities_from_metadata(
            dataset_ids, subject_ids, session_ids, starts, durations
        )

        assert len(identities) == 3
        assert identities[0].dataset_id == "ds_a"
        assert identities[1].absolute_start == 2.0
        assert identities[2].window_duration == 2.0

    def test_builds_from_numpy_arrays(self):
        dataset_ids = ["ds_a"] * 4
        subject_ids = ["sub_0"] * 4
        session_ids = ["sess_0"] * 4
        starts = np.array([0.0, 1.0, 2.0, 3.0])
        durations = np.array([1.0, 1.0, 1.0, 1.0])

        identities = build_identities_from_metadata(
            dataset_ids, subject_ids, session_ids, starts, durations
        )

        assert len(identities) == 4
        assert identities[3].absolute_start == 3.0

    def test_round_trips_through_hash(self):
        """Identities built from metadata produce stable hashes."""
        dataset_ids = ["ds_a", "ds_b"]
        subject_ids = ["sub_0", "sub_1"]
        session_ids = ["sess_0", "sess_1"]
        starts = torch.tensor([0.0, 2.0])
        durations = torch.tensor([2.0, 2.0])

        ids = build_identities_from_metadata(
            dataset_ids, subject_ids, session_ids, starts, durations
        )
        h1 = stable_hash(ids[0], seed=42)
        h2 = stable_hash(ids[0], seed=42)
        assert h1 == h2


# ---------------------------------------------------------------------------
# End-to-end selection invariance
# ---------------------------------------------------------------------------


class TestEndToEndInvariance:
    """Verify the deliverable: selection is invariant to batch order and partitioning."""

    def test_multiple_shuffles_same_result(self):
        ids = _make_population(
            n_datasets=4, n_subjects_per=5, n_sessions_per=3, n_windows_per=25
        )
        config = SelectionConfig(seed=42, max_windows=150)

        reference = hierarchical_select_windows(ids, config)

        for shuffle_seed in [1, 2, 3, 4, 5]:
            rng = random.Random(shuffle_seed)
            perm = list(range(len(ids)))
            rng.shuffle(perm)
            shuffled = [ids[i] for i in perm]
            result = hierarchical_select_windows(shuffled, config)
            assert result.fingerprint == reference.fingerprint

    def test_channel_selection_deterministic(self):
        ids = _make_population(
            n_datasets=2, n_subjects_per=3, n_sessions_per=2, n_windows_per=20
        )
        config = SelectionConfig(
            seed=42, max_windows=100, max_channel_observations=500
        )

        result = hierarchical_select_windows(ids, config)
        channel_counts = [10 + (i % 5) for i in range(len(ids))]

        ch_sel1 = select_channel_observations(
            result.window_indices, ids, channel_counts, config
        )
        ch_sel2 = select_channel_observations(
            result.window_indices, ids, channel_counts, config
        )
        assert ch_sel1 == ch_sel2

    def test_channel_selection_is_invariant_to_population_order(self):
        ids = _make_population(
            n_datasets=2, n_subjects_per=3, n_sessions_per=2, n_windows_per=20
        )
        config = SelectionConfig(
            seed=42, max_windows=100, max_channel_observations=500
        )
        channel_counts = [10 + (i % 5) for i in range(len(ids))]
        selected = hierarchical_select_windows(ids, config)
        expected = {
            ids[index].to_bytes()
            for index in select_channel_observations(
                selected.window_indices, ids, channel_counts, config
            )
        }

        permutation = list(range(len(ids)))
        random.Random(17).shuffle(permutation)
        shuffled_ids = [ids[index] for index in permutation]
        shuffled_counts = [channel_counts[index] for index in permutation]
        shuffled_selected = hierarchical_select_windows(shuffled_ids, config)
        actual = {
            shuffled_ids[index].to_bytes()
            for index in select_channel_observations(
                shuffled_selected.window_indices,
                shuffled_ids,
                shuffled_counts,
                config,
            )
        }

        assert actual == expected
