from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
from auditorydecoding.data.neurosoft_dataset import _empty_interval
from torch_brain.data import Data


class _NeurosoftLOSO:
    """Add deterministic held-out-subject validation splits to NeuroSoft.

    The processed NeuroSoft artifacts only expose a small set of fixed
    ``intersubject`` assignments.  ``split_type='loso'`` derives the desired
    leave-one-subject-out partition from the BIDS recording IDs instead.  This
    keeps every recording from the held-out subject out of training and makes
    it the *validation* partition, which is the evaluation protocol used by
    the baseline experiments.
    """

    def __init__(self, *, held_out_subject: str | None = None, **kwargs):
        self.held_out_subject = held_out_subject
        super().__init__(**kwargs)

    def get_sampling_intervals(self, split=None):
        if self.split_type == "loso":
            if split is None:
                return {
                    rid: self.get_recording(rid).domain
                    for rid in self.recording_ids
                }
            if split not in ("train", "valid", "test"):
                raise ValueError(
                    "split must be 'train', 'valid', 'test', or None."
                )
            return self._get_loso_intervals(split)
        return super().get_sampling_intervals(split)

    @staticmethod
    def _subject_from_recording_id(recording_id: str) -> str:
        """Extract the BIDS subject token without loading the recording."""
        subject, separator, _ = str(recording_id).partition("_")
        if not separator or not subject.startswith("sub-"):
            raise ValueError(
                "LOSO requires BIDS-style recording IDs beginning with 'sub-'; "
                f"got {recording_id!r}."
            )
        return subject

    def _get_loso_intervals(self, split: str) -> dict:
        if not self.held_out_subject:
            raise ValueError(
                "held_out_subject is required when split_type='loso'."
            )

        subjects = {
            self._subject_from_recording_id(rid) for rid in self.recording_ids
        }
        if self.held_out_subject not in subjects:
            raise ValueError(
                f"held_out_subject={self.held_out_subject!r} is not in this "
                f"dataset. Available subjects: {sorted(subjects)}."
            )
        if len(subjects) < 2:
            raise ValueError(
                "LOSO requires recordings from at least two subjects."
            )

        result = {}
        for rid in self.recording_ids:
            is_held_out = (
                self._subject_from_recording_id(rid) == self.held_out_subject
            )
            include = (split == "train" and not is_held_out) or (
                split == "valid" and is_held_out
            )
            if not include:
                result[rid] = _empty_interval()
                continue

            data = self.get_recording(rid)
            if self.task_type == "on_vs_off":
                result[rid] = data.on_vs_off_trials
            elif self.task_type == "acoustic_stim":
                result[rid] = data.acoustic_stim_trials
            else:
                raise ValueError(f"Invalid task_type {self.task_type!r}.")
        return result


class NeurosoftMinipigs2026(_NeurosoftLOSO, _AuditoryNeurosoftMinipigs2026):
    """Foundry wrapper for Neurosoft minipig data, including LOSO splits."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)


class NeurosoftMonkeys2026(_NeurosoftLOSO, _AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data, including LOSO splits."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)
