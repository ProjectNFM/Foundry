from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.data.dataset import np
from torch_brain.dataset import Dataset
from .mixins import EEGDatasetMixin


class KorczowskiBrainInvaders2014a(EEGDatasetMixin, Dataset):
    """Brain Invaders 2014a P300 Dataset (Korczowski et al., 2014).

    EEG recordings from 71 subjects performing a visual P300 Brain-Computer Interface task
    using 16 active dry electrodes across up to 3 sessions.

    Args:
        root: Path to the root directory containing processed dataset folders.
        recording_ids: Optional list of recording IDs to include. If None, all recordings are used.
        transform: Optional transform to apply to each data sample.
        uniquify_channel_ids: If True, prefix channel IDs with session ID to ensure uniqueness.
        split_type: Which k-fold split to use. Options: "fold_0" through "fold_4", or None.
            If None, returns full domain without splits.
        dirname: Directory name within root containing the dataset files.
        **kwargs: Additional arguments passed to Dataset.__init__.
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        fold_number: Optional[Literal[0, 1, 2]] = 0,
        fold_type: Literal["intra-subject", "inter-subject"] = "inter-subject",
        dirname: str = "korczowski_brain_invaders_2014a",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.eeg_dataset_mixin_uniquify_channel_ids = uniquify_channel_ids
        self.fold_number = fold_number
        self.fold_type = fold_type

        if fold_type not in ["intra-subject", "inter-subject"]:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. "
                "Must be one of ['intra-subject', 'inter-subject']."
            )

        if fold_number not in [0, 1, 2]:
            raise ValueError(
                f"Invalid fold_number '{fold_number}'. "
                "Must be one of [0, 1, 2] or None."
            )

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        """Get sampling intervals for the dataset.

        Args:
            split: Which split to return. Options: "train", "valid", "test", or None.
                If None, returns the full domain.

        Returns:
            Dictionary mapping recording IDs to their sampling intervals.
        """
        if self.fold_number is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if self.fold_type == "intra-subject":
            key = f"splits.{self.fold_type}.fold_{self.fold_number}.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }
        elif self.fold_type == "inter-subject":
            key = f"splits.SubjectSplit_fold{self.fold_number}"
            res = {}
            for rid in self.recording_ids:
                recording = self.get_recording(rid)
                if recording.get_nested_attribute(key) == split:
                    res[rid] = recording.get_nested_attribute("p300_trials")
            return res
