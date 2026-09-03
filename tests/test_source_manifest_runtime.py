"""Runtime safety checks for Phase 3 source and transfer manifests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.fraction_manifest import _canonical_hash
from foundry.data.source_manifest import source_interval_identity
from foundry.training.checkpoint_manifest import (
    CheckpointManifestError,
    verify_checkpoint_integrity,
)
from main import _validate_manifest_target


class _Intervals:
    def __init__(self) -> None:
        self.start = np.array([0.0, 0.5])
        self.end = np.array([0.5, 1.0])
        self.behavior_labels = np.array(["low_bass", "midrange"])

    def __len__(self) -> int:
        return len(self.start)


class _Dataset:
    def __init__(self, intervals: _Intervals) -> None:
        self._intervals = intervals

    def get_sampling_intervals(self, *, split: str):
        return {"sub-01_ses-01": self._intervals}


def _recording(*, selected_indices: list[int] = [0, 1]):
    intervals = _Intervals()
    canonical_id = "minipigs:sub-01_ses-01"
    ids = [
        source_interval_identity(canonical_id, i, start, end, label)
        for i, (start, end, label) in enumerate(
            zip(intervals.start, intervals.end, intervals.behavior_labels)
        )
    ]
    return SimpleNamespace(
        recording_id="sub-01_ses-01",
        canonical_recording_id=canonical_id,
        train_source_intervals_hash=_canonical_hash(ids),
        valid_source_intervals_hash=_canonical_hash(ids),
        train_selected_indices=selected_indices,
        train_selected_interval_ids=[ids[i] for i in selected_indices if i < len(ids)],
        valid_interval_ids=ids,
    )


def test_source_runtime_verification_accepts_exact_live_intervals():
    datamodule = SimpleNamespace(dataset=_Dataset(_Intervals()))
    manifest = SimpleNamespace(recordings=[_recording()])

    NeuralDataModule._verify_source_manifest_intervals(
        datamodule, manifest, source_interval_identity
    )


def test_source_runtime_verification_rejects_out_of_range_index():
    datamodule = SimpleNamespace(dataset=_Dataset(_Intervals()))
    manifest = SimpleNamespace(recordings=[_recording(selected_indices=[3])])

    with pytest.raises(RuntimeError, match="out-of-range"):
        NeuralDataModule._verify_source_manifest_intervals(
            datamodule, manifest, source_interval_identity
        )


def test_source_runtime_verification_rejects_split_drift():
    datamodule = SimpleNamespace(dataset=_Dataset(_Intervals()))
    recording = _recording()
    recording.train_source_intervals_hash = "tampered"
    manifest = SimpleNamespace(recordings=[recording])

    with pytest.raises(RuntimeError, match="split hash mismatch"):
        NeuralDataModule._verify_source_manifest_intervals(
            datamodule, manifest, source_interval_identity
        )


class NeurosoftMinipigsDataset:
    pass


def _downstream_datamodule(recording_id: str):
    return SimpleNamespace(
        dataset_class=NeurosoftMinipigsDataset,
        dataset=SimpleNamespace(recording_ids=[recording_id]),
    )


def test_manifest_transfer_requires_exact_excluded_target():
    manifest = {
        "trained_on": {
            "excluded_target": {"species": "minipigs", "subject": "sub-01"}
        }
    }
    datamodule = _downstream_datamodule("sub-01_ses-01_task-AcousStim")

    _validate_manifest_target(manifest, datamodule)

    manifest["trained_on"]["excluded_target"]["subject"] = "sub-02"
    with pytest.raises(ValueError, match="does not match"):
        _validate_manifest_target(manifest, datamodule)


def test_checkpoint_manifest_cannot_escape_its_checkpoint_root(tmp_path):
    manifest = {"checkpoint": {"path": "../outside.ckpt", "sha256": "x"}}

    with pytest.raises(CheckpointManifestError, match="relative path"):
        verify_checkpoint_integrity(manifest, str(tmp_path))
