from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from torch_brain.data import Data
from torch_brain.datasets.dataset import Dataset

FoldType = Literal["intrasession", "intersubject", "intersession"]
VALID_FOLD_TYPES = get_args(FoldType)


class BrainInvadersP300(Dataset):
    """Foundry wrapper for the Brain Invaders 2014a P300 dataset.

    Translates Foundry's ``fold`` / ``split_type`` conventions to
    torch_brain's ``fold_number`` / ``fold_type``.
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

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ):
        """Return trial-level sampling intervals.

        Always returns ``p300_trials`` intervals since P300 trials are
        discrete, non-contiguous events.
        """
        if split is None:
            return {
                rid: self.get_recording(rid).p300_trials
                for rid in self.recording_ids
            }

        if split not in ("train", "valid", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intrasession":
            key = f"splits.fold_{self.fold_number}.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
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
                result[rid] = rec.p300_trials
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
        return []
