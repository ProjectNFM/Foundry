from typing import Callable, Optional, Dict, Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from torch_brain.data import collate
from torch_brain.dataset import Dataset as TorchBrainDataset


class EEGDataModule(LightningDataModule):
    """Base DataModule for EEG datasets.

    Handles train/val/test splits, applies transforms, and creates DataLoaders
    with proper tokenization.
    """

    def __init__(
        self,
        dataset: TorchBrainDataset,
        model: torch.nn.Module,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
    ):
        """Initialize EEGDataModule.

        Args:
            dataset: Dataset instance with get_sampling_intervals method.
                Transform should be applied directly to the dataset.
            model: Model instance with tokenize method.
            transform: Optional transform to apply before tokenization (e.g., Patching).
                This should be passed to the dataset constructor.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for DataLoaders.
            pin_memory: Whether to pin memory in DataLoaders.
            shuffle_train: Whether to shuffle training data.
        """
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

        self.train_intervals: Optional[Dict[str, Any]] = None
        self.val_intervals: Optional[Dict[str, Any]] = None
        self.test_intervals: Optional[Dict[str, Any]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train/val/test sampling intervals."""
        if stage == "test" or stage is None:
            self.train_intervals = self.dataset.get_sampling_intervals(
                split="train"
            )
            self.val_intervals = self.dataset.get_sampling_intervals(
                split="valid"
            )

        if stage == "test" or stage is None:
            self.test_intervals = self.dataset.get_sampling_intervals(
                split="test"
            )

    def _collate_fn(self, batch):
        """Collate function that applies tokenization."""
        processed_batch = []
        for item in batch:
            tokenized = self.model.tokenize(item)
            processed_batch.append(tokenized)
        return collate(processed_batch)

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_intervals is None:
            raise RuntimeError(
                "setup() must be called before train_dataloader()"
            )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_intervals is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_intervals is None:
            raise RuntimeError(
                "setup() must be called before test_dataloader()"
            )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
