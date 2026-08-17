"""Validation-only leave-one-subject-out split tests for NeuroSoft."""

from __future__ import annotations

import pytest

from foundry.data.datasets.neurosoft import _NeurosoftLOSO


class _Recording:
    def __init__(self, name: str):
        self.acoustic_stim_trials = [name]
        self.on_vs_off_trials = [name]


class _FakeNeurosoftDataset(_NeurosoftLOSO):
    """Minimal dataset double; avoids opening external NeuroSoft artifacts."""

    def __init__(self, held_out_subject: str):
        self.split_type = "loso"
        self.held_out_subject = held_out_subject
        self.task_type = "acoustic_stim"
        self.recording_ids = [
            "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw",
            "sub-01_ses-02_task-AcousStim_acq-LH_desc-raw",
            "sub-02_ses-01_task-AcousStim_acq-LH_desc-raw",
        ]
        self._recordings = {rid: _Recording(rid) for rid in self.recording_ids}

    def get_recording(self, recording_id):
        return self._recordings[recording_id]


def _dataset(held_out_subject: str = "sub-02"):
    return _FakeNeurosoftDataset(held_out_subject)


def test_loso_keeps_held_out_subject_out_of_training():
    dataset = _dataset()

    train = dataset.get_sampling_intervals("train")
    valid = dataset.get_sampling_intervals("valid")

    assert all(len(train[rid]) > 0 for rid in train if rid.startswith("sub-01"))
    assert all(
        len(train[rid]) == 0 for rid in train if rid.startswith("sub-02")
    )
    assert all(
        len(valid[rid]) == 0 for rid in valid if rid.startswith("sub-01")
    )
    assert all(len(valid[rid]) > 0 for rid in valid if rid.startswith("sub-02"))


def test_loso_has_no_test_partition_under_validation_only_protocol():
    dataset = _dataset()

    assert all(
        len(interval) == 0
        for interval in dataset.get_sampling_intervals("test").values()
    )


def test_loso_rejects_unknown_held_out_subject():
    dataset = _dataset("sub-99")

    with pytest.raises(ValueError, match="Available subjects"):
        dataset.get_sampling_intervals("train")
