from typing import Callable, Optional, Literal

import torch
from torch.utils.data import DataLoader
from torch_brain.data import collate

from torch_brain.data.sampler import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose

from foundry.data.datasets.schalk_wolpaw_physionet_2009 import (
    SchalkWolpawPhysionet2009,
)


class PhysionetDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for Physionet Motor Imagery Dataset.

    Extends EEGDataModule with Physionet-specific configuration including
    task type selection and fold configuration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        window_length: Optional[float] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
        # Dataset specific args
        recording_ids: Optional[list[str]] = None,
        uniquify_channel_ids: bool = True,
        task_type: Optional[
            Literal["MotorImagery", "LeftRightImagery", "RightHandFeetImagery"]
        ] = "MotorImagery",
        fold_number: Optional[Literal[0, 1, 2]] = 0,
        fold_type: Literal["intra-subject", "inter-subject"] = "inter-subject",
        dirname: str = "schalk_wolpaw_physionet_2009",
    ):
        """Initialize PhysionetDataModule.

        Args:
            model: Model instance with tokenize method.
            root: Root directory of the data.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for DataLoaders.
            pin_memory: Whether to pin memory in DataLoaders.
            shuffle_train: Whether to shuffle training data.
            window_length: Length of windows for RandomFixedWindowSampler in seconds.
            transform: Optional transform to apply to each data sample before tokenization.
            seed: Random seed for sampling.
            recording_ids: Optional list of recording IDs to include.
            uniquify_channel_ids: If True, prefix channel IDs with session ID.
            task_type: Task configuration for sampling intervals.
            fold_number: Which k-fold split to use.
            fold_type: Type of fold to use.
            dirname: Directory name within root containing the dataset files.
        """
        super().__init__()
        self.model = model
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.window_length = window_length
        self.transform = transform
        self.seed = seed

        # Dataset specific args
        self.recording_ids = recording_ids
        self.uniquify_channel_ids = uniquify_channel_ids
        self.task_type = task_type
        self.fold_number = fold_number
        self.fold_type = fold_type
        self.dirname = dirname

        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup the DataModule.

        Args:
            stage: Stage to setup the DataModule for. Can be 'fit', 'test', 'validate'.
        """
        if self.dataset is None:
            transforms = Compose(
                [
                    self.transform,
                    self.model.tokenize,
                ]
            )
            self.dataset = SchalkWolpawPhysionet2009(
                root=self.root,
                recording_ids=self.recording_ids,
                transform=transforms,
                uniquify_channel_ids=self.uniquify_channel_ids,
                task_type=self.task_type,
                fold_number=self.fold_number,
                fold_type=self.fold_type,
                dirname=self.dirname,
            )

            if self.model.session_emb.is_lazy():
                session_ids = self.dataset.get_recording_ids()
                self.model.session_emb.initialize_vocab(session_ids)

            if self.model.channel_emb.is_lazy():
                channel_ids = self.dataset.get_channel_ids()
                self.model.channel_emb.initialize_vocab(channel_ids)

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with RandomFixedWindowSampler."""
        train_intervals = self.dataset.get_sampling_intervals(split="train")
        sampler = RandomFixedWindowSampler(
            sampling_intervals=train_intervals,
            window_length=self.window_length,
            drop_short=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader with RandomFixedWindowSampler."""
        val_intervals = self.dataset.get_sampling_intervals(split="valid")
        sampler = RandomFixedWindowSampler(
            sampling_intervals=val_intervals,
            window_length=self.window_length,
            drop_short=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader with RandomFixedWindowSampler."""
        test_intervals = self.dataset.get_sampling_intervals(split="test")
        sampler = RandomFixedWindowSampler(
            sampling_intervals=test_intervals,
            window_length=self.window_length,
            drop_short=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
