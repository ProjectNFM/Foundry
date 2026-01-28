from typing import Callable, Optional, Literal

import torch
from torch.utils.data import DataLoader

from torch_brain.data import RandomFixedWindowSampler

from foundry.data.datasets.schalk_wolpaw_physionet_2009 import (
    SchalkWolpawPhysionet2009,
)
from .base import EEGDataModule


class PhysionetDataModule(EEGDataModule):
    """PyTorch Lightning DataModule for Physionet Motor Imagery Dataset.

    Extends EEGDataModule with Physionet-specific configuration including
    task type selection and fold configuration.
    """

    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        transform: Optional[Callable] = None,
        task_type: Optional[
            Literal["MotorImagery", "LeftRightImagery", "RightHandFeetImagery"]
        ] = "MotorImagery",
        fold_number: Optional[Literal[0, 1, 2]] = 0,
        fold_type: Literal["intra-subject", "inter-subject"] = "inter-subject",
        recording_ids: Optional[list[str]] = None,
        uniquify_channel_ids: bool = True,
        dirname: str = "schalk_wolpaw_physionet_2009",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        window_duration: Optional[float] = None,
        **kwargs,
    ):
        """Initialize PhysionetDataModule.

        Args:
            root: Path to the root directory containing processed dataset folders.
            model: Model instance with tokenize method.
            transform: Optional transform to apply before tokenization (e.g., Patching).
            task_type: Task configuration to use for sampling intervals. Options:
                - "MotorImagery": All 5 classes (left_hand, right_hand, hands, feet, rest)
                - "LeftRightImagery": Binary classification (left_hand, right_hand)
                - "RightHandFeetImagery": Binary classification (right_hand, feet)
                - None: Use full domain (no task-specific splits)
            fold_number: Which k-fold split to use. Options: 0 through 2, or None.
            fold_type: Type of fold to use. Options: "intra-subject", "inter-subject"
            recording_ids: Optional list of recording IDs to include.
            uniquify_channel_ids: If True, prefix channel IDs with session ID.
            dirname: Directory name within root containing the dataset files.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for DataLoaders.
            pin_memory: Whether to pin memory in DataLoaders.
            shuffle_train: Whether to shuffle training data.
            window_duration: Duration of windows for RandomFixedWindowSampler in seconds.
                If None, uses model.sequence_length.
            **kwargs: Additional arguments passed to dataset initialization.
        """
        dataset = SchalkWolpawPhysionet2009(
            root=root,
            recording_ids=recording_ids,
            transform=transform,
            uniquify_channel_ids=uniquify_channel_ids,
            task_type=task_type,
            fold_number=fold_number,
            fold_type=fold_type,
            dirname=dirname,
            **kwargs,
        )

        super().__init__(
            dataset=dataset,
            model=model,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
        )

        self.window_duration = (
            window_duration
            if window_duration is not None
            else getattr(model, "sequence_length", 1.0)
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with RandomFixedWindowSampler."""
        train_intervals = self.dataset.get_sampling_intervals(split="train")
        sampler = RandomFixedWindowSampler(
            dataset=self.dataset,
            intervals=train_intervals,
            window_duration=self.window_duration,
            batch_size=self.batch_size,
        )

        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader with RandomFixedWindowSampler."""
        val_intervals = self.dataset.get_sampling_intervals(split="valid")
        sampler = RandomFixedWindowSampler(
            dataset=self.dataset,
            intervals=val_intervals,
            window_duration=self.window_duration,
            batch_size=self.batch_size,
        )

        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader with RandomFixedWindowSampler."""
        test_intervals = self.dataset.get_sampling_intervals(split="test")
        sampler = RandomFixedWindowSampler(
            dataset=self.dataset,
            intervals=test_intervals,
            window_duration=self.window_duration,
            batch_size=self.batch_size,
        )

        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Collate function that applies tokenization."""
        return super()._collate_fn(batch)
