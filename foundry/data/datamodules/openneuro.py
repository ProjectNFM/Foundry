from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.dataset import NestedDataset
from torch_brain.transforms import Compose

from foundry.data.datamodules.base import NeuralDataModule

OpenNeuroSplitType = Literal["intrasession", "intersubject", "intersession"]
SplitAssignmentType = Literal["train", "valid", "test"]


class OpenNeuroDataModule(NeuralDataModule):
    """DataModule for one or more processed OpenNeuro datasets."""

    def __init__(
        self,
        root: str,
        dataset_dirs: list[str],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sequence_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        split_type: OpenNeuroSplitType = "intrasession",
        recording_ids: Optional[list[str]] = None,
        keep_files_open: bool = True,
        task_type: Optional[str] = None,
    ):
        if not dataset_dirs:
            raise ValueError("dataset_dirs must contain at least one dataset.")

        dataset_kwargs = {
            "split_type": split_type,
            "recording_ids": recording_ids,
            "keep_files_open": keep_files_open,
        }

        super().__init__(
            dataset_class=OpenNeuroDataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sequence_length=sequence_length,
            transforms=transforms,
            tokenizer=tokenizer,
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            task_type=task_type,
        )

        self.dataset_dirs = dataset_dirs

    def _group_recording_ids(self) -> dict[str, Optional[list[str]]]:
        recording_ids = self.dataset_kwargs.get("recording_ids")
        if recording_ids is None:
            return {dataset_dir: None for dataset_dir in self.dataset_dirs}

        if len(self.dataset_dirs) == 1:
            return {self.dataset_dirs[0]: recording_ids}

        dataset_ids: dict[str, list[str]] = {
            dataset_dir: [] for dataset_dir in self.dataset_dirs
        }
        valid_prefixes = set(self.dataset_dirs)
        for recording_id in recording_ids:
            if "/" not in recording_id:
                raise ValueError(
                    "When using multiple dataset_dirs, recording_ids must be "
                    "namespaced as '<dataset_dir>/<recording_id>'."
                )
            dataset_name, raw_recording_id = recording_id.split("/", 1)
            if dataset_name not in valid_prefixes:
                raise ValueError(
                    f"Unknown dataset prefix '{dataset_name}' in recording_ids. "
                    f"Expected one of {sorted(valid_prefixes)}."
                )
            dataset_ids[dataset_name].append(raw_recording_id)
        return dataset_ids

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            return

        transform = Compose(self.transform) if self.transform else None
        dataset_kwargs = dict(self.dataset_kwargs)
        split_type = dataset_kwargs.pop("split_type")
        dataset_kwargs.pop("recording_ids", None)
        recording_ids_by_dataset = self._group_recording_ids()

        if len(self.dataset_dirs) == 1:
            dataset_dir = self.dataset_dirs[0]
            self.dataset = OpenNeuroDataset(
                root=self.root,
                dataset_dir=dataset_dir,
                split_type=split_type,
                recording_ids=recording_ids_by_dataset[dataset_dir],
                transform=transform,
                **dataset_kwargs,
            )
            return

        datasets = {}
        for dataset_dir in self.dataset_dirs:
            datasets[dataset_dir] = OpenNeuroDataset(
                root=self.root,
                dataset_dir=dataset_dir,
                split_type=split_type,
                recording_ids=recording_ids_by_dataset[dataset_dir],
                transform=None,
                **dataset_kwargs,
            )

        # Apply transforms at the nested level so they run consistently once.
        self.dataset = NestedDataset(datasets=datasets, transform=transform)

    def _create_dataloader(self, split: SplitAssignmentType) -> DataLoader:
        sampling_intervals = self.dataset.get_sampling_intervals(
            split_assignment=split
        )

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
            drop_last=(split == "train"),
        )

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        if isinstance(self.dataset, NestedDataset):
            channel_ids: set[str] = set()
            for dataset in self.dataset.datasets.values():
                channel_ids.update(dataset.get_channel_ids())
            return sorted(channel_ids)
        return sorted(self.dataset.get_channel_ids())
