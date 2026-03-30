"""Generic base DataModule for neural data with flexible composition.

This module provides a base LightningDataModule that works with any dataset
and optional tokenization/vocab initialization. It decouples data loading from
model-specific preprocessing.
"""

import logging
from collections import Counter
from typing import Callable, Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose

logger = logging.getLogger(__name__)


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
    READOUT_CLASS_NAMES: dict[str, list[str]] = {}

    @classmethod
    def get_class_names_for_task(cls, task_type: str) -> dict[str, list[str]]:
        readout_names = cls.get_readout_specs_for_task(task_type)
        return {
            name: cls.READOUT_CLASS_NAMES[name]
            for name in readout_names
            if name in cls.READOUT_CLASS_NAMES
        }

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
        task_type: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.window_length = window_length
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs or {}
        self.task_type = task_type

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

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        """Compute inverse-frequency class weights from the training split.

        Scans training intervals to count label occurrences per readout, then
        returns weights proportional to ``total / (num_classes * class_count)``
        (the same formula used by scikit-learn's ``compute_class_weight``),
        raised to the power of ``smoothing``.

        A smoothing of 1.0 gives standard inverse-frequency weights. Values
        below 1 (e.g. 0.5 for sqrt-inverse-frequency) dampen the correction
        and reduce over-prediction of minority classes, which often improves
        macro-F1 on imbalanced datasets. A smoothing of 0.0 produces uniform
        weights.

        Must be called after :meth:`setup`.
        """
        from torch_brain.registry import MODALITY_REGISTRY
        from foundry.data.datasets.modalities import MappedCrossEntropyLoss

        if self.dataset is None:
            raise RuntimeError("Call setup() before compute_class_weights()")
        if self.task_type is None:
            raise ValueError(
                "task_type must be set to compute class weights automatically"
            )

        readout_names = self.TASK_TO_READOUT.get(self.task_type, [])
        if not readout_names:
            return {}

        import foundry.data.datasets.modalities  # noqa: F401

        train_intervals = self.dataset.get_sampling_intervals(split="train")

        class_weights: dict[str, list[float]] = {}
        for readout_name in readout_names:
            spec = MODALITY_REGISTRY[readout_name]
            value_field = spec.value_key.split(".")[-1]

            counts: Counter = Counter()
            for rid, intervals in train_intervals.items():
                if not hasattr(intervals, value_field):
                    continue
                values = getattr(intervals, value_field)
                unique_labels = np.unique(values)
                for v in unique_labels:
                    selected = intervals.select_by_mask(values == v)
                    counts[int(v)] += sum(selected.end - selected.start)

            if isinstance(spec.loss_fn, MappedCrossEntropyLoss):
                key_map = dict(
                    zip(
                        spec.loss_fn._keys.tolist(),
                        spec.loss_fn._values.tolist(),
                    )
                )
                mapped_counts: Counter = Counter()
                for label, count in counts.items():
                    if label in key_map:
                        mapped_counts[key_map[label]] += count
                counts = mapped_counts

            total = sum(counts.values())
            num_classes = spec.dim
            weights = [
                (total / (num_classes * max(counts.get(i, 0), 1))) ** smoothing
                for i in range(num_classes)
            ]
            class_weights[readout_name] = weights
            logger.info(
                "Class weights for %s (smoothing=%.2f): %s (counts: %s)",
                readout_name,
                smoothing,
                [f"{w:.3f}" for w in weights],
                dict(sorted(counts.items())),
            )

        return class_weights

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
