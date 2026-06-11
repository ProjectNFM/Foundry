"""Generic base DataModule for neural data with flexible composition.

This module provides a base LightningDataModule that works with any dataset
and optional tokenization/vocab initialization. It decouples data loading from
model-specific preprocessing.
"""

import logging
from typing import Callable, Literal, Optional

import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose

from foundry.tasks.class_weights import compute_class_weights_for_tasks

logger = logging.getLogger(__name__)


def normalize_data_config(data_cfg: DictConfig) -> None:
    """Merge top-level dataset params into ``dataset_kwargs`` (in-place).

    Experiment configs override ``data.split_type`` / ``data.task_type`` at
    the top level while base data configs may leave mandatory placeholders
    inside ``dataset_kwargs``. Hydra's recursive ``instantiate`` fails on
    those placeholders, so resolve them here before instantiation.
    """
    merges = ("task_type", "split_type", "fold", "recording_ids")
    if "dataset_kwargs" not in data_cfg:
        OmegaConf.update(data_cfg, "dataset_kwargs", {}, force_add=True)

    with open_dict(data_cfg):
        with open_dict(data_cfg.dataset_kwargs):
            for key in merges:
                if key in data_cfg and not OmegaConf.is_missing(data_cfg, key):
                    data_cfg.dataset_kwargs[key] = data_cfg[key]

            for key in list(data_cfg.dataset_kwargs.keys()):
                if OmegaConf.is_missing(data_cfg.dataset_kwargs, key):
                    del data_cfg.dataset_kwargs[key]


class NeuralDataModule(LightningDataModule):
    """Generic LightningDataModule for neural datasets with optional tokenization.

    This base module handles data loading, sampling, and batching for any dataset
    that has `get_sampling_intervals()` and optionally `get_channel_ids()` and
    `get_recording_ids()` methods. Model-specific preprocessing (tokenization)
    is applied as a transform, making the datamodule reusable.

    Usage:
        dm = NeuralDataModule(
            dataset_class=MyDataset,
            root="./data/",
            batch_size=32,
            sequence_length=10.0,
            tokenizer=model.tokenize,  # optional
            dataset_kwargs={"dirname": "my_dataset"},
        )
        trainer.fit(module, dm)
    """

    def __init__(
        self,
        dataset_class,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sequence_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        dataset_kwargs: Optional[dict] = None,
        task_type: Optional[str] = None,
        split_type: Optional[str] = None,
        fold: Optional[int] = None,
        recording_ids: Optional[list[str]] = None,
    ):
        super().__init__()
        if isinstance(dataset_class, str):
            dataset_class = get_class(dataset_class)
        self.dataset_class = dataset_class
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_length = sequence_length
        self.seed = seed
        self.dataset_kwargs = dict(dataset_kwargs or {})

        for key, val in (
            ("task_type", task_type),
            ("split_type", split_type),
            ("fold", fold),
            ("recording_ids", recording_ids),
        ):
            if val is not None:
                self.dataset_kwargs[key] = val

        self.task_type = self.dataset_kwargs.get("task_type")

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

        transform_list = list(self.transform) if self.transform else []
        if self.task_type is not None and hasattr(
            self.dataset_class, "get_required_transforms"
        ):
            required = self.dataset_class.get_required_transforms(
                self.task_type
            )
            transform_list = list(required) + transform_list

        transform = Compose(transform_list) if transform_list else None

        self.dataset = self.dataset_class(
            root=self.root,
            transform=transform,
            **self.dataset_kwargs,
        )

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        if self.dataset is None:
            raise RuntimeError("Call setup() before compute_class_weights()")
        if self.task_type is None:
            raise ValueError(
                "task_type must be set to compute class weights automatically"
            )

        task_configs = self.dataset_class.get_tasks_for_experiment(
            self.task_type
        )
        return compute_class_weights_for_tasks(
            task_configs, self.dataset, split="train", smoothing=smoothing
        )

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())

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
            window_length=self.sequence_length,
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
