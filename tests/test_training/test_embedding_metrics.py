"""Tests for embedding visualization metrics (Phase 4).

Validates all metrics against small hand-computable synthetic examples.
All tests are pure-function tests requiring no Lightning or W&B.

Covers:
- L2 normalization, zero/non-finite exclusion, and norm statistics
- Channel temporal consistency (dynamic and static)
- Within-recording separability (dynamic, static unavailable)
- Cross-recording canonical-electrode leave-one-recording-out
- Anatomical Spearman (centroid and per-window, eligibility thresholds)
- Backbone cosine silhouette exclusions and reuse
- Electrode name normalization and alias resolution
- Logging key formatting
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foundry.training.callbacks.embedding_metrics import (
    AnatomyResult,
    CanonicalConsistencyResult,
    SeparabilityResult,
    SilhouetteResult,
    TemporalConsistencyResult,
    _spearman_distance_correlation,
    channel_anatomical_scores,
    channel_anatomical_scores_with_windows,
    channel_canonical_consistency,
    channel_temporal_consistency,
    channel_within_recording_separability,
    compute_backbone_silhouettes,
    compute_channel_metrics,
    compute_norm_statistics,
    cosine_distance_matrix,
    cosine_silhouette,
    cosine_similarity_matrix,
    format_backbone_silhouettes_for_logging,
    format_channel_metrics_for_logging,
    get_electrode_positions_3d,
    normalize_electrode_name,
    normalize_representations,
    resolve_channel_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a single vector or batch of vectors."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return v / norms


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_basic_normalization(self):
        vectors = np.array([[3.0, 4.0], [0.0, 5.0]])
        result = normalize_representations(vectors)

        assert result.n_total == 2
        assert result.n_valid == 2
        assert result.n_zero == 0
        assert result.n_nonfinite == 0
        assert result.vectors.shape == (2, 2)

        norms_after = np.linalg.norm(result.vectors, axis=1)
        np.testing.assert_allclose(norms_after, 1.0, atol=1e-6)

        assert abs(result.norms[0] - 5.0) < 1e-6
        assert abs(result.norms[1] - 5.0) < 1e-6

    def test_zero_vector_excluded(self):
        vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        result = normalize_representations(vectors)

        assert result.n_total == 3
        assert result.n_valid == 1
        assert result.n_zero == 2
        assert result.n_nonfinite == 0
        assert result.vectors.shape == (1, 2)
        np.testing.assert_allclose(result.vectors[0], [1.0, 0.0], atol=1e-6)

    def test_nonfinite_excluded(self):
        vectors = np.array([[1.0, 2.0], [np.inf, 0.0], [np.nan, 1.0]])
        result = normalize_representations(vectors)

        assert result.n_total == 3
        assert result.n_valid == 1
        assert result.n_nonfinite == 2
        assert result.vectors.shape == (1, 2)

    def test_all_invalid(self):
        vectors = np.array([[0.0, 0.0], [np.nan, 0.0]])
        result = normalize_representations(vectors)

        assert result.n_valid == 0
        assert result.vectors.shape == (0, 2)

    def test_valid_mask_alignment(self):
        vectors = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
        result = normalize_representations(vectors)

        expected_mask = np.array([True, False, True])
        np.testing.assert_array_equal(result.valid_mask, expected_mask)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            normalize_representations(np.array([1.0, 2.0]))

    def test_preserves_direction(self):
        vectors = np.array([[3.0, 4.0]])
        result = normalize_representations(vectors)
        expected = np.array([3.0 / 5.0, 4.0 / 5.0])
        np.testing.assert_allclose(result.vectors[0], expected, atol=1e-6)

    def test_outputs_are_exactly_unit_normalized(self):
        vectors = np.array([[1e-3, 0.0], [3.0, 4.0]])
        result = normalize_representations(vectors)

        np.testing.assert_allclose(
            np.linalg.norm(result.vectors, axis=1), 1.0, atol=1e-12
        )


class TestNormStatistics:
    def test_basic_stats(self):
        norms = np.array([2.0, 4.0, 6.0])
        stats = compute_norm_statistics(norms)

        assert stats["mean"] == pytest.approx(4.0)
        assert stats["min"] == pytest.approx(2.0)
        assert stats["max"] == pytest.approx(6.0)
        assert stats["median"] == pytest.approx(4.0)

    def test_excludes_zero_and_nonfinite(self):
        norms = np.array([0.0, 3.0, np.inf, 5.0])
        stats = compute_norm_statistics(norms)
        assert stats["mean"] == pytest.approx(4.0)
        assert stats["min"] == pytest.approx(3.0)

    def test_all_invalid(self):
        norms = np.array([0.0, np.nan])
        stats = compute_norm_statistics(norms)
        assert stats == {}


# ---------------------------------------------------------------------------
# Cosine geometry tests
# ---------------------------------------------------------------------------


class TestCosineGeometry:
    def test_similarity_identity(self):
        a = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
        sim = cosine_similarity_matrix(a)

        np.testing.assert_allclose(sim[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sim[1, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(sim[0, 1], 0.0, atol=1e-6)

    def test_similarity_cross(self):
        a = _l2_normalize(np.array([[1.0, 0.0]]))
        b = _l2_normalize(np.array([[1.0, 1.0]]))
        sim = cosine_similarity_matrix(a, b)
        expected = 1.0 / math.sqrt(2)
        np.testing.assert_allclose(sim[0, 0], expected, atol=1e-6)

    def test_distance_matrix_properties(self):
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        dist = cosine_distance_matrix(vecs)

        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)
        np.testing.assert_allclose(dist[0, 2], 2.0, atol=1e-6)
        np.testing.assert_allclose(dist[0, 1], 1.0, atol=1e-6)
        assert (dist >= 0).all()
        assert (dist <= 2.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Backbone silhouette tests
# ---------------------------------------------------------------------------


class TestCosinesilhouette:
    def test_perfect_separation(self):
        """Two well-separated clusters should have silhouette near 1."""
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [1.0, -0.01],
                    [-1.0, 0.0],
                    [-1.0, 0.01],
                    [-1.0, -0.01],
                ]
            )
        )
        labels = np.array([0, 0, 0, 1, 1, 1])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.score is not None
        assert result.score > 0.9
        assert result.n_samples == 6
        assert result.n_excluded == 0
        assert result.n_groups == 2

    def test_single_group_returns_none(self):
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
        labels = np.array([0, 0])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.score is None
        assert result.n_groups == 1

    def test_singleton_excluded(self):
        """Groups with <2 members are excluded."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]))
        labels = np.array([0, 0, 1])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.score is None
        assert result.n_excluded_groups == 1
        assert result.n_groups == 1

    def test_negative_labels_excluded(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [0.0, 1.0],
                    [0.01, 1.0],
                    [0.5, 0.5],
                ]
            )
        )
        labels = np.array([0, 0, 1, 1, -1])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.score is not None
        assert result.n_excluded == 1
        assert result.n_samples == 4

    def test_empty_input(self):
        result = cosine_silhouette(np.empty((0, 0)), np.array([]))
        assert result.score is None
        assert result.n_samples == 0

    def test_string_labels(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [-1.0, 0.0],
                    [-1.0, 0.01],
                ]
            )
        )
        labels = np.array(["A", "A", "B", "B"])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.score is not None
        assert result.score > 0.5

    def test_empty_string_labels_excluded(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [-1.0, 0.0],
                    [-1.0, 0.01],
                    [0.0, 1.0],
                ]
            )
        )
        labels = np.array(["A", "A", "B", "B", ""])
        dist = cosine_distance_matrix(vecs)
        result = cosine_silhouette(dist, labels)

        assert result.n_excluded == 1


class TestComputeBackboneSilhouettes:
    def test_multiple_groupings_reuse_distance(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [-1.0, 0.0],
                    [-1.0, 0.01],
                ]
            )
        )
        dist = cosine_distance_matrix(vecs)
        groupings = {
            "dataset": np.array([0, 0, 1, 1]),
            "subject": np.array([0, 1, 0, 1]),
        }
        results = compute_backbone_silhouettes(dist, groupings)

        assert "dataset" in results
        assert "subject" in results
        assert results["dataset"].score is not None


# ---------------------------------------------------------------------------
# Channel temporal consistency tests
# ---------------------------------------------------------------------------


class TestTemporalConsistency:
    def test_static_returns_one(self):
        vecs = _l2_normalize(np.random.randn(5, 4))
        rec = np.array(["r0"] * 5)
        ch = np.array(["c0", "c0", "c1", "c1", "c1"])
        result = channel_temporal_consistency(vecs, rec, ch, "static")

        assert result.score == 1.0
        assert result.is_static is True

    def test_dynamic_identical_vectors(self):
        """If all windows for a channel have identical vectors, consistency = 1."""
        v = _l2_normalize(np.array([1.0, 0.0, 0.0]).reshape(1, -1))
        vecs = np.tile(v, (4, 1))
        rec = np.array(["r0"] * 4)
        ch = np.array(["c0", "c0", "c1", "c1"])
        result = channel_temporal_consistency(vecs, rec, ch, "dynamic")

        assert result.score is not None
        assert result.score == pytest.approx(1.0, abs=1e-5)
        assert result.is_static is False

    def test_dynamic_orthogonal_vectors(self):
        """Two windows with orthogonal vectors for the same channel → consistency near 0."""
        vecs = _l2_normalize(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        )
        rec = np.array(["r0"] * 4)
        ch = np.array(["c0", "c0", "c1", "c1"])
        result = channel_temporal_consistency(vecs, rec, ch, "dynamic")

        assert result.score is not None
        assert result.score < 0.1

    def test_channel_needs_two_windows(self):
        """Channels with only 1 window are excluded from the computation."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
        rec = np.array(["r0", "r0", "r0"])
        ch = np.array(["c0", "c1", "c2"])
        result = channel_temporal_consistency(vecs, rec, ch, "dynamic")

        assert result.score is None
        assert result.n_channels == 0

    def test_macro_averaging_across_recordings(self):
        """High-channel-count recordings don't dominate thanks to macro-averaging."""
        v_tight = _l2_normalize(np.array([[1.0, 0.0]]))
        v_spread = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))

        vecs = np.concatenate(
            [
                np.tile(
                    v_tight, (10, 1)
                ),  # rec0: 5 channels × 2 windows, all identical
                v_spread[0:1],
                v_spread[1:2],  # rec1: 1 channel × 2 windows, orthogonal
            ]
        )
        rec = np.array(["r0"] * 10 + ["r1"] * 2)
        ch = np.array(
            [
                "c0",
                "c0",
                "c1",
                "c1",
                "c2",
                "c2",
                "c3",
                "c3",
                "c4",
                "c4",
                "cx",
                "cx",
            ]
        )
        result = channel_temporal_consistency(vecs, rec, ch, "dynamic")

        assert result.score is not None
        assert result.n_recordings == 2

    def test_empty_input(self):
        result = channel_temporal_consistency(
            np.empty((0, 4)), np.array([]), np.array([]), "dynamic"
        )
        assert result.score is None
        assert result.n_observations == 0


# ---------------------------------------------------------------------------
# Within-recording separability tests
# ---------------------------------------------------------------------------


class TestSeparability:
    def test_static_unavailable(self):
        vecs = _l2_normalize(np.random.randn(4, 3))
        rec = np.array(["r0"] * 4)
        ch = np.array(["c0", "c0", "c1", "c1"])
        result = channel_within_recording_separability(vecs, rec, ch, "static")

        assert result.accuracy is None
        assert result.unavailable_reason is not None

    def test_perfect_separability(self):
        """Well-separated channels → accuracy near 1.0, positive margin."""
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.01, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.01],
                ]
            )
        )
        rec = np.array(["r0"] * 4)
        ch = np.array(["c0", "c0", "c1", "c1"])
        result = channel_within_recording_separability(vecs, rec, ch, "dynamic")

        assert result.accuracy is not None
        assert result.accuracy == pytest.approx(1.0, abs=0.01)
        assert result.margin is not None
        assert result.margin > 0

    def test_needs_two_channels(self):
        """Recording with only 1 channel cannot compute separability."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [1.0, 0.01]]))
        rec = np.array(["r0", "r0"])
        ch = np.array(["c0", "c0"])
        result = channel_within_recording_separability(vecs, rec, ch, "dynamic")

        assert result.accuracy is None

    def test_channels_need_two_windows(self):
        """Channels with <2 windows are ineligible."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.01]]))
        rec = np.array(["r0"] * 3)
        ch = np.array(["c0", "c1", "c0"])
        result = channel_within_recording_separability(vecs, rec, ch, "dynamic")

        assert result.accuracy is None

    def test_empty_input(self):
        result = channel_within_recording_separability(
            np.empty((0, 3)), np.array([]), np.array([]), "dynamic"
        )
        assert result.accuracy is None


# ---------------------------------------------------------------------------
# Canonical-electrode consistency tests
# ---------------------------------------------------------------------------


class TestCanonicalConsistency:
    def test_perfect_consistency_across_recordings(self):
        """Same electrode in different recordings has similar centroids → high accuracy."""
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.01, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.01],
                ]
            )
        )
        rec = np.array(["r0", "r1", "r0", "r1"])
        can = np.array(["fp1", "fp1", "fp2", "fp2"])
        result = channel_canonical_consistency(vecs, rec, can)

        assert result.accuracy is not None
        assert result.accuracy == pytest.approx(1.0, abs=0.01)
        assert result.margin is not None
        assert result.margin > 0
        assert result.n_electrodes == 2

    def test_electrode_in_single_recording_excluded(self):
        """Electrodes present in only one recording are excluded."""
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ]
            )
        )
        rec = np.array(["r0", "r1", "r0", "r1", "r0"])
        can = np.array(["fp1", "fp1", "fp2", "fp2", "unique"])
        result = channel_canonical_consistency(vecs, rec, can)

        assert result.n_excluded_electrodes == 1
        assert result.n_electrodes == 2

    def test_needs_two_eligible_electrodes(self):
        """Cannot compute classification with <2 eligible electrodes."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [1.0, 0.01]]))
        rec = np.array(["r0", "r1"])
        can = np.array(["fp1", "fp1"])
        result = channel_canonical_consistency(vecs, rec, can)

        assert result.accuracy is None

    def test_empty_input(self):
        result = channel_canonical_consistency(
            np.empty((0, 3)), np.array([]), np.array([])
        )
        assert result.accuracy is None

    def test_preserves_distinct_channels_with_the_same_canonical_label(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.9, 0.1],
                ]
            )
        )
        rec = np.array(["r0", "r0", "r0", "r1", "r1", "r1"])
        canonical = np.array(["t7", "t7", "fp1", "fp1", "t7", "t7"])
        channels = np.array(["T3", "T7", "Fp1", "Fp1", "T3", "T7"])

        result = channel_canonical_consistency(vecs, rec, canonical, channels)

        assert result.n_centroids == 6


# ---------------------------------------------------------------------------
# Anatomical organization tests
# ---------------------------------------------------------------------------


class TestAnatomicalScores:
    @staticmethod
    def _make_positions(n: int) -> dict[str, np.ndarray]:
        """Create n channel positions on a circle in the XY plane."""
        positions = {}
        for i in range(n):
            angle = 2 * math.pi * i / n
            positions[f"ch{i}"] = np.array(
                [math.cos(angle), math.sin(angle), 0.0]
            )
        return positions

    def test_correlated_anatomy(self):
        """Channels whose embeddings correlate with physical position → positive Spearman."""
        n_channels = 12
        positions = self._make_positions(n_channels)

        vecs = []
        rec_ids = []
        ch_ids = []
        for i in range(n_channels):
            angle = 2 * math.pi * i / n_channels
            # Embedding direction correlated with physical position
            emb = np.array([math.cos(angle), math.sin(angle), 0.0, 0.0])
            vecs.append(emb)
            rec_ids.append("r0")
            ch_ids.append(f"ch{i}")

        vecs_arr = _l2_normalize(np.array(vecs))
        result = channel_anatomical_scores(
            vecs_arr,
            np.array(rec_ids),
            np.array(ch_ids),
            positions,
            "dynamic",
            min_positioned=9,
        )

        assert result.centroid_spearman is not None
        assert result.centroid_spearman > 0.5
        assert result.n_eligible_recordings == 1

    def test_below_threshold_excluded(self):
        """Recordings with <min_positioned resolved channels are excluded."""
        positions = self._make_positions(8)
        vecs = _l2_normalize(np.random.randn(8, 4))
        rec = np.array(["r0"] * 8)
        ch = np.array([f"ch{i}" for i in range(8)])

        result = channel_anatomical_scores(
            vecs,
            rec,
            ch,
            positions,
            "dynamic",
            min_positioned=9,
        )
        assert result.centroid_spearman is None
        assert result.n_eligible_recordings == 0

    def test_at_threshold_included(self):
        """Exactly min_positioned channels is eligible."""
        positions = self._make_positions(9)
        vecs = _l2_normalize(np.random.randn(9, 4))
        rec = np.array(["r0"] * 9)
        ch = np.array([f"ch{i}" for i in range(9)])

        result = channel_anatomical_scores(
            vecs,
            rec,
            ch,
            positions,
            "dynamic",
            min_positioned=9,
        )
        assert result.n_eligible_recordings == 1

    def test_empty_positions(self):
        vecs = _l2_normalize(np.random.randn(5, 3))
        result = channel_anatomical_scores(
            vecs,
            np.array(["r0"] * 5),
            np.array([f"ch{i}" for i in range(5)]),
            {},
            "dynamic",
        )
        assert result.centroid_spearman is None

    def test_with_windows_dynamic(self):
        """Per-window Spearman is computed for dynamic mode."""
        n_channels = 10
        positions = self._make_positions(n_channels)

        vecs = []
        rec_ids = []
        ch_ids = []
        win_ids = []
        for win in range(2):
            for i in range(n_channels):
                angle = 2 * math.pi * i / n_channels
                emb = np.array([math.cos(angle), math.sin(angle), 0.0, 0.0])
                noise = np.random.randn(4) * 0.01
                vecs.append(emb + noise)
                rec_ids.append("r0")
                ch_ids.append(f"ch{i}")
                win_ids.append(f"w{win}")

        vecs_arr = _l2_normalize(np.array(vecs))
        result = channel_anatomical_scores_with_windows(
            vecs_arr,
            np.array(rec_ids),
            np.array(ch_ids),
            np.array(win_ids),
            positions,
            "dynamic",
            min_positioned=9,
        )

        assert result.centroid_spearman is not None
        assert result.window_spearman is not None
        assert result.n_eligible_windows == 2

    def test_static_mode_no_window_scores(self):
        """Static mode: per-window scores are not computed."""
        n_channels = 10
        positions = self._make_positions(n_channels)
        vecs = _l2_normalize(np.random.randn(n_channels, 4))
        rec = np.array(["r0"] * n_channels)
        ch = np.array([f"ch{i}" for i in range(n_channels)])
        win = np.array(["w0"] * n_channels)

        result = channel_anatomical_scores_with_windows(
            vecs,
            rec,
            ch,
            win,
            positions,
            "static",
            min_positioned=9,
        )

        assert result.window_spearman is None
        assert result.n_eligible_windows == 0


class TestSpearmanDistanceCorrelation:
    def test_perfectly_correlated(self):
        """Representations whose cosine distance mirrors physical distance."""
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))
        positions = np.array(
            [
                [1.0, 0.0, 0.0],
                [math.sqrt(0.5), math.sqrt(0.5), 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        rho = _spearman_distance_correlation(vecs, positions)
        assert rho is not None
        assert rho > 0.5

    def test_constant_cosine_distance_returns_none(self):
        """All pairs equidistant in cosine space → undefined correlation."""
        v = _l2_normalize(np.array([[1.0, 0.0]]))
        vecs = np.tile(v, (3, 1))
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )

        rho = _spearman_distance_correlation(vecs, positions)
        assert rho is None

    def test_single_pair(self):
        """Fewer than 2 vectors returns None."""
        rho = _spearman_distance_correlation(
            np.array([[1.0, 0.0]]), np.array([[0.0, 0.0, 0.0]])
        )
        assert rho is None

    def test_is_invariant_to_coordinate_radius(self):
        vecs = _l2_normalize(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))
        unit_positions = np.array(
            [
                [1.0, 0.0, 0.0],
                [math.sqrt(0.5), math.sqrt(0.5), 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        scaled_positions = unit_positions * np.array([[2.0], [0.5], [3.0]])

        rho_unit = _spearman_distance_correlation(vecs, unit_positions)
        rho_scaled = _spearman_distance_correlation(vecs, scaled_positions)

        assert rho_unit == pytest.approx(rho_scaled)


# ---------------------------------------------------------------------------
# Electrode name normalization tests
# ---------------------------------------------------------------------------


class TestElectrodeNormalization:
    def test_strips_namespace(self):
        assert normalize_electrode_name("dataset_a/sub-01/Fp1") == "fp1"

    def test_lowercases(self):
        assert normalize_electrode_name("FP1") == "fp1"

    def test_strips_whitespace(self):
        assert normalize_electrode_name("  T3 ") == "t7"

    def test_vetted_alias_t3(self):
        assert normalize_electrode_name("T3") == "t7"

    def test_vetted_alias_t4(self):
        assert normalize_electrode_name("T4") == "t8"

    def test_vetted_alias_t5(self):
        assert normalize_electrode_name("T5") == "p7"

    def test_vetted_alias_t6(self):
        assert normalize_electrode_name("T6") == "p8"

    def test_no_alias_passthrough(self):
        assert normalize_electrode_name("Cz") == "cz"

    def test_ecog_passthrough(self):
        assert normalize_electrode_name("ECoG-LAH1") == "ecog-lah1"

    def test_bipolar_passthrough(self):
        assert normalize_electrode_name("Fp1-F3") == "fp1-f3"

    def test_collapses_internal_whitespace(self):
        assert normalize_electrode_name("F  3") == "f3"


class TestElectrodePositions3D:
    def test_returns_dict(self):
        positions = get_electrode_positions_3d()
        if not positions:
            pytest.skip("MNE not installed")
        assert isinstance(positions, dict)

    def test_known_electrode_present(self):
        positions = get_electrode_positions_3d()
        if not positions:
            pytest.skip("MNE not installed")
        assert "fp1" in positions
        assert "cz" in positions
        assert positions["cz"].shape == (3,)

    def test_aliases_present(self):
        positions = get_electrode_positions_3d()
        if not positions:
            pytest.skip("MNE not installed")
        if "t7" in positions:
            assert "t3" in positions
            np.testing.assert_array_equal(positions["t3"], positions["t7"])


class TestResolveChannelPositions:
    def test_resolves_known_channels(self):
        positions = {
            "fp1": np.array([0.0, 0.1, 0.0]),
            "cz": np.array([0.0, 0.0, 0.1]),
        }
        channels = np.array(["fp1", "cz", "unknown_ch"])
        resolved, n_res, n_unres = resolve_channel_positions(
            channels, positions
        )

        assert n_res == 2
        assert n_unres == 1
        assert "fp1" in resolved
        assert "unknown_ch" not in resolved


# ---------------------------------------------------------------------------
# Aggregate metric computation tests
# ---------------------------------------------------------------------------


class TestComputeChannelMetrics:
    def test_returns_all_keys(self):
        vecs = _l2_normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.01],
                    [0.0, 1.0],
                    [0.01, 1.0],
                ]
            )
        )
        rec = np.array(["r0", "r0", "r0", "r0"])
        ch = np.array(["c0", "c0", "c1", "c1"])

        metrics = compute_channel_metrics(vecs, rec, ch, "dynamic")

        assert "temporal_consistency" in metrics
        assert "separability" in metrics
        assert "canonical_consistency" in metrics
        assert "anatomy" in metrics

    def test_static_mode_marks_appropriately(self):
        vecs = _l2_normalize(np.random.randn(4, 3))
        rec = np.array(["r0"] * 4)
        ch = np.array(["c0", "c0", "c1", "c1"])

        metrics = compute_channel_metrics(vecs, rec, ch, "static")

        tc = metrics["temporal_consistency"]
        assert isinstance(tc, TemporalConsistencyResult)
        assert tc.is_static is True

        sep = metrics["separability"]
        assert isinstance(sep, SeparabilityResult)
        assert sep.unavailable_reason is not None


# ---------------------------------------------------------------------------
# W&B formatting tests
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_channel_metrics_keys(self):
        metrics = {
            "temporal_consistency": TemporalConsistencyResult(
                score=0.9,
                n_recordings=2,
                n_channels=10,
                n_observations=40,
                is_static=False,
            ),
            "separability": SeparabilityResult(
                accuracy=0.85,
                margin=0.1,
                n_recordings=2,
                n_channels=10,
                n_observations=40,
            ),
            "canonical_consistency": CanonicalConsistencyResult(
                accuracy=0.7,
                margin=0.05,
                n_electrodes=5,
                n_centroids=15,
                n_excluded_electrodes=2,
            ),
            "anatomy": AnatomyResult(
                centroid_spearman=0.6,
                window_spearman=0.55,
                centroid_iqr=0.1,
                window_iqr=0.15,
                n_eligible_recordings=3,
                n_eligible_windows=10,
                n_resolved_channels=30,
                n_undefined=0,
            ),
        }
        logged = format_channel_metrics_for_logging(metrics)

        assert "val/embedding_viz/channel/temporal_consistency" in logged
        assert logged["val/embedding_viz/channel/temporal_consistency"] == 0.9
        assert "val/embedding_viz/channel/within_recording_accuracy" in logged
        assert "val/embedding_viz/channel/within_recording_margin" in logged
        assert "val/embedding_viz/channel/canonical_accuracy" in logged
        assert "val/embedding_viz/channel/canonical_margin" in logged
        assert "val/embedding_viz/channel/anatomy_centroid_spearman" in logged
        assert "val/embedding_viz/channel/anatomy_window_spearman" in logged
        assert (
            "val/embedding_viz/channel/temporal_consistency/n_observations"
            in logged
        )
        assert (
            "val/embedding_viz/channel/within_recording/n_observations"
            in logged
        )

    def test_none_scores_omitted(self):
        metrics = {
            "temporal_consistency": TemporalConsistencyResult(
                score=None,
                n_recordings=0,
                n_channels=0,
                n_observations=0,
                is_static=False,
            ),
            "separability": SeparabilityResult(
                accuracy=None,
                margin=None,
                n_recordings=0,
                n_channels=0,
                n_observations=0,
                unavailable_reason="static mode",
            ),
            "canonical_consistency": CanonicalConsistencyResult(
                accuracy=None,
                margin=None,
                n_electrodes=0,
                n_centroids=0,
                n_excluded_electrodes=0,
            ),
            "anatomy": AnatomyResult(
                centroid_spearman=None,
                window_spearman=None,
                centroid_iqr=None,
                window_iqr=None,
                n_eligible_recordings=0,
                n_eligible_windows=0,
                n_resolved_channels=0,
                n_undefined=0,
            ),
        }
        logged = format_channel_metrics_for_logging(metrics)

        assert "val/embedding_viz/channel/temporal_consistency" not in logged
        assert (
            "val/embedding_viz/channel/within_recording_accuracy" not in logged
        )
        assert (
            "val/embedding_viz/channel/anatomy_centroid_spearman" not in logged
        )
        assert (
            "val/embedding_viz/channel/temporal_consistency/n_recordings"
            in logged
        )

    def test_backbone_silhouette_keys(self):
        silhouettes = {
            "dataset": SilhouetteResult(
                score=0.5,
                n_samples=100,
                n_excluded=10,
                n_groups=3,
                n_excluded_groups=1,
            ),
            "subject": SilhouetteResult(
                score=None,
                n_samples=50,
                n_excluded=60,
                n_groups=1,
                n_excluded_groups=5,
            ),
        }
        logged = format_backbone_silhouettes_for_logging(silhouettes)

        assert "val/embedding_viz/backbone/silhouette/dataset" in logged
        assert logged["val/embedding_viz/backbone/silhouette/dataset"] == 0.5
        assert "val/embedding_viz/backbone/silhouette/subject" not in logged
        assert (
            "val/embedding_viz/backbone/silhouette/subject/n_samples" in logged
        )
        assert (
            "val/embedding_viz/backbone/silhouette/dataset/n_excluded_groups"
            in logged
        )


# ---------------------------------------------------------------------------
# Integration with Phase 3 observation_selector types
# ---------------------------------------------------------------------------


class TestPhase3Integration:
    """Verify that Phase 4 metrics consume Phase 3 outputs correctly."""

    def test_normalization_on_representation_shaped_tensor(self):
        """Simulate backbone_representations shape (N, D_backbone)."""
        backbone = np.random.randn(10, 64).astype(np.float32)
        result = normalize_representations(backbone)

        assert result.vectors.shape == (10, 64)
        norms = np.linalg.norm(result.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_channel_metrics_on_flattened_channel_repr(self):
        """Simulate channel_representations (N_windows, C, D) flattened to (N_obs, D)."""
        n_windows = 4
        n_channels = 10
        d = 16

        channel_repr = np.random.randn(n_windows, n_channels, d).astype(
            np.float32
        )
        flat_vecs = channel_repr.reshape(-1, d)
        norm_result = normalize_representations(flat_vecs)

        rec_ids = np.repeat(["r0", "r0", "r1", "r1"], n_channels)
        ch_ids = np.tile([f"ch{i}" for i in range(n_channels)], n_windows)

        tc = channel_temporal_consistency(
            norm_result.vectors, rec_ids, ch_ids, "dynamic"
        )
        assert isinstance(tc, TemporalConsistencyResult)
        assert tc.n_observations > 0

    def test_silhouette_on_selected_backbone_subset(self):
        """Simulate selecting a subset of backbone vectors (Phase 3 → Phase 4)."""
        all_vectors = np.random.randn(100, 32).astype(np.float32)
        selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        selected = all_vectors[selected_indices]

        norm_result = normalize_representations(selected)
        dist = cosine_distance_matrix(norm_result.vectors)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        result = cosine_silhouette(dist, labels[: norm_result.n_valid])
        assert isinstance(result, SilhouetteResult)
