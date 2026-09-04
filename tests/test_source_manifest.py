"""Unit tests for source-pool and source-selection manifest schemas (WP1).

Covers hash determinism, round-trip serialization, validation gates,
canonical identity helpers, and error paths for malformed manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.data.source_manifest import (
    SOURCE_POOL_SCHEMA,
    SOURCE_SELECTION_SCHEMA,
    SelectionCondition,
    SelectionSummary,
    SourcePool,
    SourcePoolManifest,
    SourceRecordingSelection,
    SourceSelectionManifest,
    canonical_recording_id,
    source_interval_identity,
)
from foundry.data.source_selection import select_class_indices


# ── helpers ──────────────────────────────────────────────────────────────────


def _pool(composition="same_species", **overrides):
    defaults = dict(
        composition=composition,
        source_species=["minipigs"],
        source_subjects=["minipigs:sub-01"],
        source_recordings=["minipigs:sub-01_ses-01"],
        source_subject_count=1,
        source_recording_count=1,
        class_counts={"low_bass": 5, "midrange": 3},
        target_leakage=[],
        source_train_split_hashes={"minipigs:sub-01_ses-01": "abc123"},
    )
    defaults.update(overrides)
    return SourcePool(**defaults)


def _pool_manifest(**overrides):
    pool = _pool()
    base = dict(
        phase0_audit_sha256="audit_hash_placeholder",
        target_species="minipigs",
        target_subject="sub-06",
        eligible_target_recordings=["sub-06_ses-01"],
        pools={"same_species": pool},
        manifest_hash="placeholder",
    )
    base.update(overrides)
    payload = SourcePoolManifest(**base).to_dict()
    payload.pop("manifest_hash")
    real_hash = SourcePoolManifest.compute_hash(payload)
    base["manifest_hash"] = real_hash
    return SourcePoolManifest(**base)


def _recording(species="minipigs", subject="sub-01", recording_id="sub-01_ses-01"):
    cid = canonical_recording_id(species, recording_id)
    return SourceRecordingSelection(
        species=species,
        subject=subject,
        recording_id=recording_id,
        canonical_recording_id=cid,
        raw_channel_count=32,
        supported_channel_count=18,
        train_source_intervals_hash="hash_train",
        train_counts_by_class={"low_bass": 3, "midrange": 2},
        train_selected_interval_ids_hash="hash_selected",
        available_train_windows=5,
        valid_source_intervals_hash="hash_valid",
        valid_selected_interval_ids_hash="hash_valid_sel",
        available_validation_windows=2,
    )


def _selection_manifest(**overrides):
    rec = _recording()
    condition = SelectionCondition(
        source_composition="same_species",
        requested_fraction=1.0,
        subject_count_bin=None,
        source_selection_seed=42,
        class_coverage_policy="all_available",
    )
    summary = SelectionSummary(
        source_subject_count=1,
        source_recording_count=1,
        selected_train_examples=5,
        available_train_windows=5,
        realized_train_windows_per_epoch=5,
        selected_signal_seconds=2.5,
        validation_examples=2,
        available_validation_windows=2,
        represented_class_union=["low_bass", "midrange"],
        represented_class_intersection=["low_bass", "midrange"],
        requested_fraction=1.0,
        realized_fraction=1.0,
        selection_implementation="neurosoft-classwise-permutation-v1",
        sampler_implementation="neurosoft-first-fixed-window-v1",
        window_seconds=0.5,
        batch_size=5,
    )
    base = dict(
        selection_id="test_sel_id",
        family="source_volume",
        phase0_audit_sha256="audit_hash",
        source_pool_manifest="../pools/test.json",
        source_pool_hash="pool_hash",
        target_species="minipigs",
        target_subject="sub-06",
        condition=condition,
        summary=summary,
        subjects=["minipigs:sub-01"],
        recordings=[rec],
        target_leakage=[],
        manifest_hash="placeholder",
    )
    base.update(overrides)
    manifest = SourceSelectionManifest(**base)
    payload = manifest.to_dict()
    payload.pop("manifest_hash")
    real_hash = SourceSelectionManifest.compute_hash(payload)
    base["manifest_hash"] = real_hash
    return SourceSelectionManifest(**base)


# ── canonical identity helpers ───────────────────────────────────────────────


class TestCanonicalRecordingId:
    def test_format(self):
        assert (
            canonical_recording_id("minipigs", "sub-01_ses-01")
            == "minipigs:sub-01_ses-01"
        )

    def test_species_collision_avoidance(self):
        raw = "sub-03_ses-01_task-AcousStim_acq-RH_desc-raw"
        assert canonical_recording_id("minipigs", raw) != canonical_recording_id(
            "monkeys", raw
        )

    def test_empty_species_fails(self):
        with pytest.raises(ValueError, match="species"):
            canonical_recording_id("", "sub-01")

    def test_empty_recording_fails(self):
        with pytest.raises(ValueError, match="recording_id"):
            canonical_recording_id("minipigs", "")


class TestSourceIntervalIdentity:
    def test_deterministic(self):
        a = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "bass")
        b = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "bass")
        assert a == b

    def test_sensitive_to_index(self):
        a = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "bass")
        b = source_interval_identity("minipigs:sub-01", 1, 0.0, 0.5, "bass")
        assert a != b

    def test_sensitive_to_label(self):
        a = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "bass")
        b = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "treble")
        assert a != b

    def test_sensitive_to_recording_id(self):
        a = source_interval_identity("minipigs:sub-01", 0, 0.0, 0.5, "bass")
        b = source_interval_identity("monkeys:sub-01", 0, 0.0, 0.5, "bass")
        assert a != b

    def test_empty_recording_fails(self):
        with pytest.raises(ValueError, match="non-empty"):
            source_interval_identity("", 0, 0.0, 0.5, "bass")


# ── source-pool manifest ────────────────────────────────────────────────────


class TestSourcePoolManifest:
    def test_round_trip_json(self):
        manifest = _pool_manifest()
        text = manifest.to_json()
        loaded = SourcePoolManifest.from_json(text)
        assert loaded.manifest_hash == manifest.manifest_hash
        loaded.validate_hash()

    def test_hash_determinism(self):
        a = _pool_manifest()
        b = _pool_manifest()
        assert a.manifest_hash == b.manifest_hash

    def test_hash_sensitive_to_target_subject(self):
        a = _pool_manifest()
        b = _pool_manifest(target_subject="sub-07")
        assert a.manifest_hash != b.manifest_hash

    def test_hash_sensitive_to_pool_contents(self):
        a = _pool_manifest()
        pool2 = _pool(class_counts={"low_bass": 999})
        b = _pool_manifest(pools={"same_species": pool2})
        assert a.manifest_hash != b.manifest_hash

    def test_hash_sensitive_to_audit_sha(self):
        a = _pool_manifest()
        b = _pool_manifest(phase0_audit_sha256="different_audit")
        assert a.manifest_hash != b.manifest_hash

    def test_validate_hash_rejects_tampered(self):
        manifest = _pool_manifest()
        d = manifest.to_dict()
        d["target_subject"] = "sub-99"
        tampered = SourcePoolManifest.from_dict(d)
        with pytest.raises(ValueError, match="hash mismatch"):
            tampered.validate_hash()

    def test_validate_no_leakage_clean(self):
        manifest = _pool_manifest()
        manifest.validate_no_leakage()

    def test_validate_no_leakage_fails(self):
        pool = _pool(target_leakage=["minipigs:sub-06_ses-01"])
        manifest = _pool_manifest(pools={"same_species": pool})
        with pytest.raises(ValueError, match="leakage"):
            manifest.validate_no_leakage()

    def test_wrong_schema_version_fails(self):
        d = _pool_manifest().to_dict()
        d["version"] = 99
        with pytest.raises(ValueError, match="version"):
            SourcePoolManifest.load_from_dict_validated(d) if hasattr(
                SourcePoolManifest, "load_from_dict_validated"
            ) else SourcePoolManifest.from_dict(d)

    def test_save_and_load(self, tmp_path):
        manifest = _pool_manifest()
        path = tmp_path / "pool.json"
        manifest.save(path)
        loaded = SourcePoolManifest.load(path)
        assert loaded.manifest_hash == manifest.manifest_hash

    def test_malformed_missing_pools_fails(self):
        with pytest.raises(ValueError):
            SourcePoolManifest.from_dict(
                {"schema": SOURCE_POOL_SCHEMA, "version": 1}
            )


# ── source-selection manifest ───────────────────────────────────────────────


class TestSourceSelectionManifest:
    def test_round_trip_json(self):
        manifest = _selection_manifest()
        text = manifest.to_json()
        loaded = SourceSelectionManifest.from_json(text)
        assert loaded.manifest_hash == manifest.manifest_hash
        loaded.validate_hash()

    def test_hash_determinism(self):
        a = _selection_manifest()
        b = _selection_manifest()
        assert a.manifest_hash == b.manifest_hash

    def test_hash_sensitive_to_selection_seed(self):
        cond_43 = SelectionCondition(
            source_composition="same_species",
            requested_fraction=1.0,
            subject_count_bin=None,
            source_selection_seed=43,
            class_coverage_policy="all_available",
        )
        a = _selection_manifest()
        b = _selection_manifest(condition=cond_43)
        assert a.manifest_hash != b.manifest_hash

    def test_hash_sensitive_to_target(self):
        a = _selection_manifest()
        b = _selection_manifest(target_subject="sub-07")
        assert a.manifest_hash != b.manifest_hash

    def test_hash_sensitive_to_recordings(self):
        rec2 = _recording(subject="sub-02", recording_id="sub-02_ses-01")
        a = _selection_manifest()
        b = _selection_manifest(recordings=[_recording(), rec2])
        assert a.manifest_hash != b.manifest_hash

    def test_validate_hash_rejects_tampered(self):
        manifest = _selection_manifest()
        d = manifest.to_dict()
        d["target_subject"] = "sub-99"
        tampered = SourceSelectionManifest.from_dict(d)
        with pytest.raises(ValueError, match="hash mismatch"):
            tampered.validate_hash()

    def test_validate_no_leakage(self):
        manifest = _selection_manifest()
        manifest.validate_no_leakage()

    def test_validate_no_leakage_fails(self):
        manifest = _selection_manifest(target_leakage=["leaked_recording"])
        with pytest.raises(ValueError, match="leakage"):
            manifest.validate_no_leakage()

    def test_validate_test_policy(self):
        manifest = _selection_manifest()
        manifest.validate_test_policy()

    def test_validate_test_policy_rejects_allowed(self):
        manifest = _selection_manifest(source_test_policy="allowed")
        with pytest.raises(ValueError, match="forbidden"):
            manifest.validate_test_policy()

    def test_validate_summary_consistency(self):
        manifest = _selection_manifest()
        manifest.validate_summary_consistency()

    def test_validate_summary_detects_train_count_mismatch(self):
        bad_summary = SelectionSummary(
            source_subject_count=1,
            source_recording_count=1,
            selected_train_examples=999,
            available_train_windows=5,
            realized_train_windows_per_epoch=0,
            selected_signal_seconds=2.5,
            validation_examples=2,
            available_validation_windows=2,
            represented_class_union=["low_bass", "midrange"],
            represented_class_intersection=["low_bass", "midrange"],
            requested_fraction=1.0,
            realized_fraction=1.0,
            selection_implementation="v1",
            sampler_implementation="v1",
            window_seconds=0.5,
            batch_size=5,
        )
        manifest = _selection_manifest(summary=bad_summary)
        with pytest.raises(ValueError, match="selected_train_examples"):
            manifest.validate_summary_consistency()

    def test_invalid_family_fails(self):
        d = _selection_manifest().to_dict()
        d["family"] = "nonexistent_family"
        with pytest.raises(ValueError, match="family"):
            SourceSelectionManifest.from_dict(d)

    def test_save_and_load(self, tmp_path):
        manifest = _selection_manifest()
        path = tmp_path / "selection.json"
        manifest.save(path)
        loaded = SourceSelectionManifest.load(path)
        assert loaded.manifest_hash == manifest.manifest_hash

    def test_byte_identical_regeneration(self, tmp_path):
        manifest = _selection_manifest()
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        manifest.save(p1)
        manifest.save(p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_malformed_missing_fields_fails(self):
        with pytest.raises(ValueError):
            SourceSelectionManifest.from_dict(
                {"schema": SOURCE_SELECTION_SCHEMA, "version": 1}
            )

    def test_duplicate_recording_ids_get_distinct_hashes(self):
        """The same raw recording ID in two species must produce different canonical IDs."""
        rec_pig = _recording(species="minipigs", recording_id="sub-01_ses-01")
        rec_monkey = _recording(species="monkeys", recording_id="sub-01_ses-01")
        assert rec_pig.canonical_recording_id != rec_monkey.canonical_recording_id


# ── select_class_indices (source selection primitive) ────────────────────────


class TestSelectClassIndices:
    def test_deterministic(self):
        indices = list(range(10))
        a = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        b = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        assert a == b

    def test_full_count_returns_sorted_all(self):
        indices = [3, 1, 4, 0, 2]
        result = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        assert result == sorted(indices)

    def test_zero_count_returns_empty(self):
        result = select_class_indices(
            [0, 1, 2], canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=0
        )
        assert result == []

    def test_prefix_nesting(self):
        """Smaller counts must be strict subsets of larger counts (nesting)."""
        indices = list(range(20))
        small = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        large = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=10
        )
        assert set(small).issubset(set(large))

    def test_different_seeds_differ(self):
        indices = list(range(20))
        a = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        b = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=43, count=5
        )
        assert a != b

    def test_different_recording_ids_differ(self):
        indices = list(range(20))
        a = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-01", class_id=0, seed=42, count=5
        )
        b = select_class_indices(
            indices, canonical_recording_id="minipigs:sub-02", class_id=0, seed=42, count=5
        )
        assert a != b

    def test_invalid_count_fails(self):
        with pytest.raises(ValueError, match="Requested"):
            select_class_indices(
                [0, 1], canonical_recording_id="x", class_id=0, seed=42, count=5
            )

    def test_negative_count_fails(self):
        with pytest.raises(ValueError, match="Requested"):
            select_class_indices(
                [0, 1], canonical_recording_id="x", class_id=0, seed=42, count=-1
            )


# ── committed manifests sanity (runs against the actual v1 tree) ─────────────


MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifests" / "neurosoft_supervised" / "v1"


@pytest.mark.skipif(
    not (MANIFEST_ROOT / "index.json").exists(),
    reason="committed v1 manifests not present",
)
class TestCommittedManifests:
    def test_index_has_no_duplicate_selection_ids(self):
        index = json.loads((MANIFEST_ROOT / "index.json").read_text())
        ids = [entry["selection_id"] for entry in index["entries"]]
        assert len(ids) == len(set(ids)), "duplicate selection IDs in index"

    def test_index_expected_count(self):
        index = json.loads((MANIFEST_ROOT / "index.json").read_text())
        assert len(index["entries"]) >= 512, (
            f"Expected ≥512 manifest entries, got {len(index['entries'])}"
        )

    def test_all_pool_manifests_validate(self):
        for path in sorted(MANIFEST_ROOT.glob("source_pools/**/*.json")):
            manifest = SourcePoolManifest.load(path)
            manifest.validate_no_leakage()

    def test_all_selection_manifests_validate_hash(self):
        families = ["phase3_smoke", "source_volume", "subject_diversity", "species_composition"]
        checked = 0
        for family_dir in families:
            for path in sorted(MANIFEST_ROOT.glob(f"{family_dir}/**/*.json")):
                manifest = SourceSelectionManifest.load(path)
                manifest.validate_no_leakage()
                manifest.validate_test_policy()
                checked += 1
        assert checked > 0, "no selection manifests found to validate"
