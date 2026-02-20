from typing import Callable, Optional, Literal
import numpy as np
from pathlib import Path

from torch_brain.dataset import Dataset
from .mixins import EEGDatasetMixin, ModalityMixin
from .modalities import (
    MOTOR_IMAGERY_5CLASS,
    MOTOR_IMAGERY_LEFT_RIGHT,
    MOTOR_IMAGERY_RIGHT_FEET,
)


class SchalkWolpawPhysionet2009(ModalityMixin, EEGDatasetMixin, Dataset):
    MODALITIES = {
        "motor_imagery_5class": MOTOR_IMAGERY_5CLASS,
        "motor_imagery_left_right": MOTOR_IMAGERY_LEFT_RIGHT,
        "motor_imagery_right_feet": MOTOR_IMAGERY_RIGHT_FEET,
    }
    """PhysioNet Motor Imagery Dataset (Schalk & Wolpaw, 2009).

    EEG motor imagery data from 109 volunteers performing motor imagery tasks.
    Includes multiple task configurations and k-fold cross-validation splits.

    Args:
        root: Path to the root directory containing processed dataset folders.
        recording_ids: Optional list of recording IDs to include. If None, all recordings are used.
        transform: Optional transform to apply to each data sample.
        uniquify_channel_ids: If True, prefix channel IDs with session ID to ensure uniqueness.
        task_type: Task configuration to use for sampling intervals. Options:
            - "MotorImagery": All 5 classes (left_hand, right_hand, hands, feet, rest)
            - "LeftRightImagery": Binary classification (left_hand, right_hand)
            - "RightHandFeetImagery": Binary classification (right_hand, feet)
            - None: Use full domain (no task-specific splits)
        fold_number: Which k-fold split to use. Options: 0 through 4, or None.
            If None, returns full domain without splits.
        fold_type: Type of fold to use. Options: "intra-subject", "inter-subject"
        dirname: Directory name within root containing the dataset files.
        **kwargs: Additional arguments passed to Dataset.__init__.
    """

    TASK_CONFIGS = {
        "MotorImagery": ["left_hand", "right_hand", "hands", "feet", "rest"],
        "LeftRightImagery": ["left_hand", "right_hand"],
        "RightHandFeetImagery": ["right_hand", "feet"],
    }

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        task_type: Optional[
            Literal["MotorImagery", "LeftRightImagery", "RightHandFeetImagery"]
        ] = "MotorImagery",
        fold_number: Optional[Literal[0, 1, 2]] = 0,
        fold_type: Literal["intra-subject", "inter-subject"] = "inter-subject",
        dirname: str = "schalk_wolpaw_physionet_2009",
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
        self.task_type = task_type
        self.fold_number = fold_number
        self.fold_type = fold_type

        if task_type is not None and task_type not in self.TASK_CONFIGS:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Must be one of {list(self.TASK_CONFIGS.keys())} or None."
            )

        if fold_type not in ["intra-subject", "inter-subject"]:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. Must be one of ['intra-subject', 'inter-subject']."
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
        if self.task_type is None or self.fold_number is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intra-subject":
            key = f"splits.{self.task_type}.fold_{self.fold_number}.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }
        elif self.fold_type == "inter-subject":
            key = f"splits.SubjectSplit_fold{self.fold_number}"
            res = {}
            for rid in self.recording_ids:
                recording = self.get_recording(rid)

                movements = recording.get_nested_attribute(
                    "motor_imagery_trials.movements"
                )
                valid_movements = self.TASK_CONFIGS[self.task_type]
                mask = np.isin(movements, valid_movements)

                if recording.get_nested_attribute(key) == split:
                    res[rid] = recording.get_nested_attribute(
                        "motor_imagery_trials"
                    ).select_by_mask(mask)
            return res
