import logging
from collections import Counter
from typing import Callable, Literal, Optional

import numpy as np
import torch
from brainsets.datasets import Neuroprobe2025
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.transforms import Compose

from foundry.data.datamodules.base import NeuralDataModule

logger = logging.getLogger(__name__)


def normalize_neuroprobe_recording(data):
    """Normalize Neuroprobe sample fields for existing tokenizers."""
    if hasattr(data, "seeg_data") and not hasattr(data, "seeg"):
        data.seeg = data.seeg_data

    if hasattr(data, "seeg") and hasattr(data.seeg, "data"):
        data.seeg.signal = data.seeg.data

    if not hasattr(data, "channels"):
        return data

    if not hasattr(data.channels, "type"):
        data.channels.type = np.full(
            len(data.channels.id), "seeg", dtype=object
        )
        return data

    channel_types = np.asarray(data.channels.type)
    if channel_types.dtype.kind not in {"U", "S", "O"}:
        data.channels.type = np.full(len(channel_types), "seeg", dtype=object)
        return data

    lowered = np.char.lower(channel_types.astype(str))
    valid_modalities = {"eeg", "ecog", "seeg", "ieeg"}
    if np.isin(lowered, list(valid_modalities)).all():
        data.channels.type = lowered.astype(str)
        return data

    mapped = np.where(np.isin(lowered, list(valid_modalities)), lowered, "seeg")
    data.channels.type = mapped.astype(str)
    return data


class NeuroprobeDataModule(NeuralDataModule):
    """DataModule for Neuroprobe benchmark-style train/val/test splits.

    Neuroprobe2025 resolves split membership at dataset construction time,
    so this module keeps one dataset instance per split.
    """

    TASK_TO_READOUT = {
        "speech_binary": ["neuroprobe_speech_binary"],
    }

    READOUT_CLASS_NAMES: dict[str, list[str]] = {
        "neuroprobe_speech_binary": ["NoSpeech", "Speech"],
    }

    _SPLIT_TO_DATASET_SPLIT: dict[str, Literal["train", "val", "test"]] = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sequence_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        task_type: Literal["speech_binary"] = "speech_binary",
        subset_tier: Literal["full", "lite", "nano"] = "full",
        test_subject: int = 1,
        test_session: int = 0,
        label_mode: Literal["binary", "multiclass"] = "binary",
        task: str = "speech",
        regime: Literal["SS-SM", "SS-DM", "DS-DM"] = "SS-SM",
        fold: int = 0,
        recording_ids: Optional[list[str]] = None,
        dirname: str = "neuroprobe_2025",
        uniquify_channel_ids_with_subject: bool = True,
        uniquify_channel_ids_with_session: bool = False,
    ):
        self._use_explicit_recordings = recording_ids is not None
        if self._use_explicit_recordings:
            dataset_kwargs = {
                "recording_ids": recording_ids,
                "dirname": dirname,
                "uniquify_channel_ids_with_subject": (
                    uniquify_channel_ids_with_subject
                ),
                "uniquify_channel_ids_with_session": (
                    uniquify_channel_ids_with_session
                ),
            }
        else:
            dataset_kwargs = {
                "subset_tier": subset_tier,
                "test_subject": test_subject,
                "test_session": test_session,
                "label_mode": label_mode,
                "task": task,
                "regime": regime,
                "fold": fold,
                "dirname": dirname,
                "uniquify_channel_ids_with_subject": (
                    uniquify_channel_ids_with_subject
                ),
                "uniquify_channel_ids_with_session": (
                    uniquify_channel_ids_with_session
                ),
            }

        normalized_transforms = [normalize_neuroprobe_recording]
        if transforms:
            normalized_transforms.extend(transforms)

        super().__init__(
            dataset_class=Neuroprobe2025,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sequence_length=sequence_length,
            transforms=normalized_transforms,
            tokenizer=tokenizer,
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            task_type=task_type,
        )
        self.datasets: dict[str, Neuroprobe2025] = {}

    def setup(self, stage: Optional[str] = None):
        if self.transform:
            transform = Compose(self.transform)
        else:
            transform = None

        if self._use_explicit_recordings:
            required_splits = ("train", "valid", "test")
        elif stage in (None, "fit"):
            required_splits = ("train", "valid")
        elif stage == "validate":
            required_splits = ("valid",)
        elif stage in ("test", "predict"):
            required_splits = ("test",)
        else:
            required_splits = ("train", "valid", "test")

        for split in required_splits:
            if split in self.datasets:
                continue

            kwargs = dict(self.dataset_kwargs)
            if not self._use_explicit_recordings:
                kwargs["split"] = self._SPLIT_TO_DATASET_SPLIT[split]

            self.datasets[split] = self.dataset_class(
                root=self.root,
                transform=transform,
                **kwargs,
            )

        if "train" in self.datasets:
            self.dataset = self.datasets["train"]
        elif self.datasets:
            self.dataset = next(iter(self.datasets.values()))

    def _create_dataloader(
        self, split: Literal["train", "valid", "test"]
    ) -> DataLoader:
        if self.sequence_length is None:
            raise ValueError(
                "sequence_length must be set for NeuroprobeDataModule"
            )

        if split not in self.datasets:
            stage = "fit" if split in ("train", "valid") else "test"
            self.setup(stage=stage)

        dataset = self.datasets[split]
        sampling_intervals = dataset.get_sampling_intervals()

        sampler = RandomFixedWindowSampler(
            sampling_intervals=sampling_intervals,
            window_length=self.sequence_length,
            drop_short=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=(split == "train"),
        )

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        from torch_brain.registry import MODALITY_REGISTRY
        from foundry.data.datasets.modalities import MappedCrossEntropyLoss

        if self.task_type is None:
            raise ValueError(
                "task_type must be set to compute class weights automatically"
            )
        if "train" not in self.datasets:
            self.setup(stage="fit")

        readout_names = self.TASK_TO_READOUT.get(self.task_type, [])
        if not readout_names:
            return {}

        train_intervals = self.datasets["train"].get_sampling_intervals()

        class_weights: dict[str, list[float]] = {}
        for readout_name in readout_names:
            spec = MODALITY_REGISTRY[readout_name]
            value_field = spec.value_key.split(".")[-1]

            counts: Counter = Counter()
            for intervals in train_intervals.values():
                if not hasattr(intervals, value_field):
                    continue
                values = getattr(intervals, value_field)
                unique_labels = np.unique(values)
                for value in unique_labels:
                    selected = intervals.select_by_mask(values == value)
                    counts[int(value)] += sum(selected.end - selected.start)

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
                [f"{weight:.3f}" for weight in weights],
                dict(sorted(counts.items())),
            )

        return class_weights

    def get_recording_ids(self) -> list[str]:
        if not self.datasets:
            self.setup(stage="fit")

        all_ids: set[str] = set()
        for dataset in self.datasets.values():
            all_ids.update(dataset.recording_ids)
        return sorted(all_ids)

    def get_channel_ids(self) -> list[str]:
        if not self.datasets:
            self.setup(stage="fit")

        channel_ids: set[str] = set()
        for dataset in self.datasets.values():
            channel_ids.update(dataset.get_channel_ids())
        return sorted(channel_ids)
