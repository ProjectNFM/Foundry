"""Generic base DataModule for neural data with flexible composition.

This module provides a base LightningDataModule that works with any dataset
and optional tokenization/vocab initialization. It decouples data loading from
model-specific preprocessing.
"""

from typing import Callable, Optional, Literal

import torch
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose


class NeuralDataModule(LightningDataModule):
    """Generic LightningDataModule for neural datasets with optional tokenization.

    This base module handles data loading, sampling, and batching for any dataset
    that has `get_sampling_intervals()` and optionally `get_channel_ids()` and
    `get_recording_ids()` methods. Model-specific preprocessing (tokenization)
    is applied as a transform, making the datamodule reusable.

    Subclasses should define ``TASK_TO_READOUT`` to map their task_type values
    to the corresponding readout spec names used by the model.

    Usage:
        dm = NeuralDataModule(
            dataset_class=MyDataset,
            root="./data/",
            batch_size=32,
            window_length=10.0,
            tokenizer=model.tokenize,  # optional
            dataset_kwargs={"dirname": "my_dataset"},
        )
        trainer.fit(module, dm)
    """

    TASK_TO_READOUT: dict[str, list[str]] = {}

    @classmethod
    def get_readout_specs_for_task(cls, task_type: str) -> list[str]:
        if not cls.TASK_TO_READOUT:
            raise ValueError(
                f"{cls.__name__} does not define a TASK_TO_READOUT mapping"
            )
        if task_type not in cls.TASK_TO_READOUT:
            raise ValueError(
                f"Unknown task_type '{task_type}' for {cls.__name__}. "
                f"Available: {list(cls.TASK_TO_READOUT.keys())}"
            )
        return cls.TASK_TO_READOUT[task_type]

    def __init__(
        self,
        dataset_class,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        window_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        dataset_kwargs: Optional[dict] = None,
    ):
        """Initialize NeuralDataModule.

        Args:
            dataset_class: Dataset class to instantiate (must have get_sampling_intervals method).
            root: Root directory of the data.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for DataLoaders.
            pin_memory: Whether to pin memory in DataLoaders.
            window_length: Length of windows for RandomFixedWindowSampler in seconds.
            transforms: Optional list of transforms to apply to each sample.
            tokenizer: Optional tokenizer (e.g., model.tokenize) to apply after transforms.
            seed: Random seed for sampling.
            dataset_kwargs: Optional dict of kwargs to pass to dataset_class constructor.
        """
        super().__init__()
        self.dataset_class = dataset_class
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.window_length = window_length
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs or {}

        # Build transform pipeline
        transform_list = transforms or []
        if tokenizer is not None:
            transform_list = list(transform_list) + [tokenizer]

        self.transform = transform_list if transform_list else None
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup the DataModule.

        Args:
            stage: Stage to setup the DataModule for ('fit', 'test', 'validate').
        """
        if self.dataset is not None:
            return

        # Build transform
        if self.transform:
            transform = Compose(self.transform)
        else:
            transform = None

        # Instantiate dataset
        self.dataset = self.dataset_class(
            root=self.root,
            transform=transform,
            **self.dataset_kwargs,
        )

    def _create_dataloader(
        self, split: Literal["train", "valid", "test"]
    ) -> DataLoader:
        """Create a DataLoader for a given split.

        Args:
            split: One of 'train', 'valid', or 'test'.

        Returns:
            DataLoader for the split.
        """
        sampling_intervals = self.dataset.get_sampling_intervals(split=split)

        sampler = RandomFixedWindowSampler(
            sampling_intervals=sampling_intervals,
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
            drop_last=(split == "train"),  # Only drop last for training
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return self._create_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return self._create_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return self._create_dataloader("test")
