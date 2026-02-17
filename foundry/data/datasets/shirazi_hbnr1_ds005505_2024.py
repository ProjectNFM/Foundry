from typing import Callable, Optional
from pathlib import Path

from torch_brain.dataset import Dataset
from .mixins import NeuralDatasetMixin, ModalityMixin


class ShiraziHbnr1DS0055052024(ModalityMixin, NeuralDatasetMixin, Dataset):
    MODALITIES = {}
    """Healthy Brain Network (HBN) EEG Dataset from OpenNeuro (Shirazi et al., 2024).

    High-density EEG recordings from participants performing various passive and active tasks
    including resting state, movie watching, and cognitive tasks. Uses 129-channel EEG system
    (128 electrodes E1-E128 plus Cz reference).

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
        dirname: str = "shirazi_hbnr1_ds005505_2024",
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
