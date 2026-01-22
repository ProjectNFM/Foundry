from typing import Callable, Optional, Literal
from pathlib import Path

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
        split_type: Optional[Literal["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]] = "fold_0",
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
        self.split_type = split_type
    
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
        if self.split_type is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}
        
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )
        
        key = f"splits.{self.split_type}.{split}"
        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }
