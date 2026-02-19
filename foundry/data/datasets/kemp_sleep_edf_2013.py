from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset
from .mixins import EEGDatasetMixin, ModalityMixin
from .modalities import SLEEP_STAGE_5CLASS


class KempSleepEDF2013(ModalityMixin, EEGDatasetMixin, Dataset):
    MODALITIES = {"sleep_stage_5class": SLEEP_STAGE_5CLASS}

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        fold_number: Optional[Literal[0, 1, 2, 3, 4]] = 0,
        dirname: str = "kemp_sleep_edf_2013",
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

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if self.fold_number is None:
            return {
                rid: self.get_recording(rid).domain
                for rid in self.recording_ids
            }

        if self.fold_number not in [0, 1, 2, 3, 4]:
            raise ValueError(
                f"Invalid fold_number '{self.fold_number}'. Must be one of [0, 1, 2, 3, 4] or None."
            )

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        key = f"splits.fold_{self.fold_number}.{split}"
        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }
