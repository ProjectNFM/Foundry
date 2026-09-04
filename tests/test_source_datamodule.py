"""Integration tests for source-manifest DataModule loading and session routing (WP2).

Covers _DatasetNamespaceAnnotation, source_test_policy enforcement,
selection_manifest/training_fraction mutual exclusion, and role validation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundry.data.datamodules.base import (
    NeuralDataModule,
    _DatasetNamespaceAnnotation,
)


# ── _DatasetNamespaceAnnotation ──────────────────────────────────────────────


class _FakeData:
    """Minimal stand-in for a Data object with a session.id."""

    def __init__(self, session_id: str):
        self.session = SimpleNamespace(id=session_id)


class TestDatasetNamespaceAnnotation:
    def test_sets_namespace_for_known_recording(self):
        annotation = _DatasetNamespaceAnnotation(
            {"sub-01_ses-01": "minipigs", "sub-02_ses-01": "monkeys"}
        )
        data = _FakeData("sub-01_ses-01")
        result = annotation(data)
        assert result.dataset_namespace == "minipigs"

    def test_same_raw_id_maps_to_its_species(self):
        annotation = _DatasetNamespaceAnnotation({"sub-01_ses-01": "monkeys"})
        data = _FakeData("sub-01_ses-01")
        result = annotation(data)
        assert result.dataset_namespace == "monkeys"

    def test_unknown_recording_does_not_set_namespace(self):
        annotation = _DatasetNamespaceAnnotation({"known": "minipigs"})
        data = _FakeData("unknown-recording")
        result = annotation(data)
        assert not hasattr(result, "dataset_namespace")

    def test_does_not_mutate_session_id(self):
        annotation = _DatasetNamespaceAnnotation({"sub-01_ses-01": "minipigs"})
        data = _FakeData("sub-01_ses-01")
        annotation(data)
        assert data.session.id == "sub-01_ses-01"


# ── DataModule role/config validation ────────────────────────────────────────


class _StubDataset:
    """Minimal dataset class for DataModule construction."""
    pass


class TestSourcePretrainingRoleValidation:
    """Verify mutual exclusion and test-policy rules at the DataModule constructor level."""

    def test_selection_manifest_and_training_fraction_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            NeuralDataModule(
                dataset_class=_StubDataset,
                root="/tmp/fake",
                selection_manifest="path/to/manifest.json",
                training_fraction=0.5,
            )

    def test_source_pretraining_requires_selection_manifest(self):
        with pytest.raises(ValueError, match="selection_manifest"):
            NeuralDataModule(
                dataset_class=_StubDataset,
                root="/tmp/fake",
                role="source_pretraining",
            )

    def test_source_pretraining_defaults_test_policy_to_forbidden(self):
        dm = NeuralDataModule(
            dataset_class=_StubDataset,
            root="/tmp/fake",
            role="source_pretraining",
            selection_manifest="path/to/manifest.json",
        )
        assert dm.source_test_policy == "forbidden"

    def test_explicit_forbidden_policy_preserved(self):
        dm = NeuralDataModule(
            dataset_class=_StubDataset,
            root="/tmp/fake",
            source_test_policy="forbidden",
            role="source_pretraining",
            selection_manifest="path/to/manifest.json",
        )
        assert dm.source_test_policy == "forbidden"

    def test_old_target_fraction_config_unchanged(self):
        """A config with training_fraction and no source manifest works as before."""
        dm = NeuralDataModule(
            dataset_class=_StubDataset,
            root="/tmp/fake",
            training_fraction=0.5,
        )
        assert dm.training_fraction == 0.5
        assert dm.selection_manifest_path is None
        assert dm.role is None
