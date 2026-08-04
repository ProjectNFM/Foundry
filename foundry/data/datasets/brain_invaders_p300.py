from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from torch_brain.data import Data, Interval
from torch_brain.datasets.dataset import Dataset

FoldType = Literal["intrasession", "intersubject", "intersession"]
VALID_FOLD_TYPES = get_args(FoldType)


class BrainInvadersP300(Dataset):
    """Foundry wrapper for the Brain Invaders 2014a P300 dataset.

    Translates Foundry's ``fold`` / ``split_type`` conventions to
    torch_brain's ``fold_number`` / ``fold_type``.

    The ``epoch_duration`` parameter controls how trial intervals are extended
    for sampling.  P300 trials are stored as short non-overlapping markers
    (onset → next-event-onset) to satisfy torch_brain's disjoint-interval
    constraint.  At sampling time, each marker is extended to
    ``[onset, onset + epoch_duration]`` so the full ERP window is captured.
    """

    def __init__(
        self,
        *,
        root: str | None = None,
        fold: int = 0,
        split_type: str = "intrasession",
        task_type: str
        | None = None,  # consumed by NeuralDataModule, not forwarded
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        uniquify_channel_ids: bool = True,
        dirname: str = "korczowski_brain_invaders_2014a",
        epoch_duration: float = 1.0,
        **kwargs,
    ):
        if root is None:
            from torch_brain.datasets._utils import get_processed_dir

            root = get_processed_dir()

        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.uniquify_channel_ids = uniquify_channel_ids
        self.epoch_duration = epoch_duration

        if fold is None or not (0 <= fold < 3):
            raise ValueError(
                f"Fold number must be an integer between 0 and 2, got {fold}"
            )
        self.fold_number = fold
        self.fold_type = split_type

        if split_type not in VALID_FOLD_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. Must be one of {VALID_FOLD_TYPES}."
            )

    def _extend_to_epoch_duration(self, intervals: Interval) -> Interval:
        """Extend trial markers to full ``epoch_duration`` windows for sampling.

        HDF5 stores short, non-overlapping trial markers (onset → next onset).
        This method extends each marker to ``[onset, onset + epoch_duration]``
        so the sampler creates windows that capture the full ERP response.
        The returned intervals may overlap, which is fine — the sampler only
        iterates over ``(start, end)`` pairs without checking disjointness.
        """
        if self.epoch_duration is None:
            return intervals

        extended_end = intervals.start + self.epoch_duration
        kwargs = {}
        for key in intervals.keys():
            if key not in ("start", "end"):
                kwargs[key] = getattr(intervals, key)

        return Interval(
            start=intervals.start.copy(),
            end=extended_end,
            **kwargs,
        )

    def _ensure_normalized(self) -> None:
        """Pre-normalize EEG signals in-place in the data cache.

        Called lazily on first data access. Modifies ``_data_objects`` so
        that every subsequent access already contains the normalized signal.
        """
        if getattr(self, "_signals_normalized", False):
            return
        self._signals_normalized = True

        if not hasattr(self, "_data_objects"):
            self._data_objects = {}
            import h5py

            for rid in self.recording_ids:
                fpath = self._filepaths[rid]
                self._data_objects[rid] = Data.from_hdf5(h5py.File(fpath))

        for rid, data in self._data_objects.items():
            if not (hasattr(data, "eeg") and hasattr(data.eeg, "signal")):
                continue
            ch_types = np.array([str(t) for t in data.channels.type])
            eeg_mask = np.isin(np.char.lower(ch_types), ["eeg"])
            sig = np.asarray(data.eeg.signal, dtype=np.float64)
            eeg_sig = sig[:, eeg_mask]
            mean, std = float(eeg_sig.mean()), float(eeg_sig.std())
            if std > 0:
                sig[:, eeg_mask] = (eeg_sig - mean) / std
            data.eeg.signal = sig.astype(np.float32)

    def __getitem__(self, index):
        """Optimized item access that avoids deep-copying the full recording.

        The base class deep-copies the entire cached recording (~57 MB) for
        every sample.  Instead we slice directly from the cache — ``slice()``
        already returns a new ``Data`` object without mutating the original —
        then apply the recording hook to the (small) slice.
        """
        self._ensure_normalized()
        data = self._data_objects[index.recording_id]
        sample = data.slice(index.start, index.end)
        self.get_recording_hook(sample)
        if index._namespace:
            self.apply_namespace(sample, index._namespace + "/")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ):
        """Return trial-level sampling intervals.

        Always returns ``p300_trials`` intervals since P300 trials are
        discrete, non-contiguous events.  Intervals are extended to
        ``epoch_duration`` so the sampler doesn't drop short trials.
        """
        self._ensure_normalized()
        if split is None:
            return {
                rid: self._extend_to_epoch_duration(
                    self.get_recording(rid).p300_trials
                )
                for rid in self.recording_ids
            }

        if split not in ("train", "valid", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intrasession":
            key = f"splits.fold_{self.fold_number}.{split}"
            return {
                rid: self._extend_to_epoch_duration(
                    self.get_recording(rid).get_nested_attribute(key)
                )
                for rid in self.recording_ids
            }

        key = f"splits.SubjectSplit_fold{self.fold_number}"
        fallback_keys = [
            f"splits.{self.fold_type}_fold_{self.fold_number}_assignment",
            f"splits.fold_{self.fold_number}_assignment",
        ]
        result = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            assignment = None
            for k in [key] + fallback_keys:
                try:
                    assignment = str(rec.get_nested_attribute(k))
                    break
                except (AttributeError, KeyError):
                    continue
            if assignment is None:
                raise AttributeError(
                    f"Could not find intersubject split assignment for "
                    f"recording '{rid}'. Tried keys: {[key] + fallback_keys}"
                )
            if assignment == split:
                result[rid] = self._extend_to_epoch_duration(rec.p300_trials)
        return result

    def get_channel_ids(self) -> list[str]:
        all_ids: set[str] = set()
        for rec_id in self.recording_ids:
            rec = self.get_recording(rec_id, "")
            all_ids.update(str(c) for c in np.asarray(rec.channels.id))
        return sorted(all_ids)

    def get_recording_hook(self, data: Data):
        from torch_brain.utils import np_string_prefix

        if self.uniquify_channel_ids:
            data.channels.id = np_string_prefix(
                f"{data.session.id}/", data.channels.id
            )

        super().get_recording_hook(data)

    @classmethod
    def get_required_transforms(cls, task_type: str) -> list:
        if task_type == "p300":
            return [_keep_anchor_trial]
        return []


def _keep_anchor_trial(data: Data) -> Data:
    """Keep only the anchor trial (earliest onset) in a windowed sample.

    Each sampling window is centered on a single stimulus onset, but
    ``epoch_duration`` may be long enough to overlap with subsequent
    stimuli.  Retaining all of them forces the model to produce
    contradictory predictions from the same feature vector (e.g. one
    Target and two NonTargets), which prevents learning entirely.

    This transform keeps only the trial whose onset is closest to the
    start of the window (the anchor trial) so the model sees exactly
    one classification target per window.
    """
    if not hasattr(data, "p300_trials"):
        return data

    trials = data.p300_trials
    n = len(trials.start) if hasattr(trials, "start") else 0
    if n <= 1:
        return data

    idx = int(np.argmin(np.asarray(trials.start)))
    extra = {}
    for key in trials.keys():
        if key in ("start", "end"):
            continue
        extra[key] = np.asarray(getattr(trials, key))[idx : idx + 1]

    data.p300_trials = Interval(
        start=trials.start[idx : idx + 1],
        end=trials.end[idx : idx + 1],
        **extra,
    )
    return data
