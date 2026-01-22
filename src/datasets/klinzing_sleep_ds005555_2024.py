from typing import Callable, Optional
from pathlib import Path

from torch_brain.dataset import Dataset
from .mixins import EEGDatasetMixin


class KlinzingSleepDS0055552024(EEGDatasetMixin, Dataset):
    """Bitbrain Open Access Sleep (BOAS) Dataset from OpenNeuro (Klinzing et al., 2024).
    
    Simultaneous recordings from a clinical PSG system (Micromed) and a wearable EEG headband 
    (Bitbrain) across 128 nights. Includes expert-consensus sleep stage labels.
    
    The dataset contains two acquisition types:
    - PSG recordings: Full polysomnography with multiple EEG, EOG, EMG, and physiological channels
    - Headband recordings: Wearable device with 2 frontal EEG channels and motion/PPG sensors
    
    Args:
        root: Path to the root directory containing processed dataset folders.
        recording_ids: Optional list of recording IDs to include. If None, all recordings are used.
        transform: Optional transform to apply to each data sample.
        uniquify_channel_ids: If True, prefix channel IDs with session ID to ensure uniqueness.
        dirname: Directory name within root containing the dataset files.
        **kwargs: Additional arguments passed to Dataset.__init__.
    """
    
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        dirname: str = "klinzing_sleep_ds005555_2024",
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
