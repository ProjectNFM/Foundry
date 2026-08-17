"""Generic base DataModule for neural data with flexible composition.

This module provides a base LightningDataModule that works with any dataset
and optional tokenization/vocab initialization. It decouples data loading from
model-specific preprocessing.
"""

import gc
import logging
import math
from typing import TYPE_CHECKING, Callable, Literal, Optional, Type

import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose

from foundry.data.samplers import (
    DeterministicSamplerWrapper,
    DistributedSamplerWrapper,
    FastRandomFixedWindowSampler,
    VariableLengthBatchSampler,
)
from foundry.tasks.class_weights import compute_class_weights_for_tasks
from foundry.tasks.classification_mapping import (
    filter_intervals_by_mapping,
    validate_task_mappings,
)

if TYPE_CHECKING:
    from foundry.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


_INTERPOLATION_ONLY_KEYS = ("subject", "held_out_subject")


def _disable_gc_in_worker(worker_id: int) -> None:
    """Disable cyclic GC in DataLoader workers to prevent CUDA tensor cleanup.

    When workers are forked from a process with an active CUDA context, they
    inherit references to GPU tensors (e.g. via model bound methods used as
    dataset transforms). Python's cyclic GC may try to free these tensors,
    triggering ``cudaErrorInitializationError`` because CUDA cannot be used
    in forked child processes. Disabling cyclic GC in workers is safe because
    workers only perform CPU-side data loading and do not create new CUDA
    reference cycles.
    """
    gc.disable()


def normalize_data_config(data_cfg: DictConfig) -> None:
    """Merge top-level dataset params into ``dataset_kwargs`` (in-place).

    Experiment configs override ``data.split_type`` / ``data.task_type`` at
    the top level while base data configs may leave mandatory placeholders
    inside ``dataset_kwargs``. Hydra's recursive ``instantiate`` fails on
    those placeholders, so resolve them here before instantiation.

    Keys listed in ``_INTERPOLATION_ONLY_KEYS`` (e.g. ``subject``) are used
    only for config interpolation (such as resolver arguments) and are
    stripped before instantiation so they are not passed to the datamodule
    constructor.
    """
    merges = (
        "task_type",
        "split_type",
        "fold",
        "recording_ids",
        "held_out_subject",
    )
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

        strip_keys = [k for k in _INTERPOLATION_ONLY_KEYS if k in data_cfg]
        if strip_keys:
            if "dataset_kwargs" in data_cfg:
                for dk in list(data_cfg.dataset_kwargs.keys()):
                    val = data_cfg.dataset_kwargs[dk]
                    if isinstance(val, (list, tuple)):
                        OmegaConf.update(
                            data_cfg, f"dataset_kwargs.{dk}", list(val)
                        )
            for key in strip_keys:
                del data_cfg[key]


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
        task_configs: Optional[dict[str, "TaskConfig"]] = None,
        sampler_class: Optional[Type[RandomFixedWindowSampler]] = None,
        session_pct: Optional[dict[str, float]] = None,
        window_lengths: Optional[list[float]] = None,
    ):
        """Initialize the data module.

        Args:
            dataset_class: Dataset class (or importable string) to instantiate.
            root: Root directory for the dataset files.
            batch_size: Samples per batch.
            num_workers: Number of data-loading worker processes.
            pin_memory: Whether to pin GPU memory in the DataLoader.
            sequence_length: Duration of each sampling window in seconds.
                When ``window_lengths`` is provided this is ignored for
                sampling (the maximum window length is used instead).
            transforms: Optional list of transforms applied before tokenization.
            tokenizer: Optional tokenizer callable (e.g. ``model.tokenize``)
                appended to the transform pipeline.
            seed: Random seed for reproducible sampling.
            dataset_kwargs: Extra keyword arguments forwarded to the dataset
                constructor (e.g. ``dirname``, ``split_type``).
            task_type: Convenience shortcut merged into ``dataset_kwargs``.
            split_type: Convenience shortcut merged into ``dataset_kwargs``.
            fold: Cross-validation fold index merged into ``dataset_kwargs``.
            recording_ids: Explicit list of recording IDs to use.
            task_configs: Per-task :class:`TaskConfig` dicts used for class
                mapping validation and class weight computation.
            sampler_class: Sampler class for windowed sampling. Defaults to
                :class:`FastRandomFixedWindowSampler`.
            session_pct: Per-split fraction of sessions to keep, e.g.
                ``{"train": 0.5, "valid": 1.0}``.
            window_lengths: Optional list of window durations in seconds for
                multi-length training.  When provided, a
                :class:`VariableLengthBatchSampler` is used instead of the
                regular sampler.  Each batch randomly selects one length so
                all samples within a batch share the same duration.
        """
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
        self.sampler_class: Type[RandomFixedWindowSampler] = (
            sampler_class
            if sampler_class is not None
            else FastRandomFixedWindowSampler
        )
        self.window_lengths = sorted(window_lengths) if window_lengths else None
        self._task_configs = task_configs

        for key, val in (
            ("task_type", task_type),
            ("split_type", split_type),
            ("fold", fold),
            ("recording_ids", recording_ids),
        ):
            if val is not None:
                self.dataset_kwargs[key] = val

        self.task_type = self.dataset_kwargs.get("task_type")

        raw_pct = session_pct or self.dataset_kwargs.pop("session_pct", None)
        self._session_pct: dict[str, float] = {}
        if raw_pct is not None:
            for split_name in ("train", "valid", "test"):
                pct = float(raw_pct.get(split_name, 1.0))
                if not 0.0 < pct <= 1.0:
                    raise ValueError(
                        f"session_pct.{split_name} must be in (0, 1], got {pct}"
                    )
                self._session_pct[split_name] = pct

        self._tokenizer = tokenizer

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

        if self._task_configs:
            validate_task_mappings(self._task_configs, self.dataset)

    def set_tokenizer(self, tokenizer: Optional[Callable]) -> None:
        """Replace the tokenizer in the transform pipeline.

        Can be called before or after :meth:`setup`.  When the dataset
        already exists, its transform is rebuilt in-place.
        """
        old = self._tokenizer
        self._tokenizer = tokenizer

        base = [t for t in (self.transform or []) if t is not old]
        if tokenizer is not None:
            base = list(base) + [tokenizer]
        self.transform = base if base else None

        if self.dataset is not None:
            transform_list = list(self.transform) if self.transform else []
            if self.task_type is not None and hasattr(
                self.dataset_class, "get_required_transforms"
            ):
                required = self.dataset_class.get_required_transforms(
                    self.task_type
                )
                transform_list = list(required) + transform_list
            self.dataset.transform = (
                Compose(transform_list) if transform_list else None
            )

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        """Compute inverse-frequency class weights for classification tasks.

        Args:
            smoothing: Smoothing factor for the weight computation.

        Returns:
            Dict mapping task name to a list of per-class weight floats.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
            ValueError: If ``task_configs`` was not provided at init.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before compute_class_weights()")
        if not self._task_configs:
            raise ValueError(
                "task_configs must be provided to compute class weights"
            )

        return compute_class_weights_for_tasks(
            self._task_configs, self.dataset, split="train", smoothing=smoothing
        )

    def get_recording_ids(self) -> list[str]:
        """Return sorted list of all recording IDs in the dataset."""
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        """Return sorted list of unique channel IDs across the dataset."""
        return sorted(set(self.dataset.get_channel_ids()))

    def _filter_intervals(self, sampling_intervals):
        """Remove intervals whose labels are excluded by task class mappings.

        Args:
            sampling_intervals: Dict mapping recording ID to interval lists.

        Returns:
            Filtered copy of *sampling_intervals* with unmapped intervals
            removed.  Returned unchanged when no task configs have class
            mappings.
        """
        if not self._task_configs:
            return sampling_intervals
        for name, cfg in self._task_configs.items():
            if cfg.class_mapping is None or cfg.target_extractor is None:
                continue
            value_field = cfg.target_extractor["value_key"].split(".")[-1]
            sampling_intervals = {
                rid: filter_intervals_by_mapping(
                    intervals, cfg.class_mapping, value_field
                )
                for rid, intervals in sampling_intervals.items()
            }
        return sampling_intervals

    _SPLIT_SEED_OFFSETS: dict[str, int] = {"train": 0, "valid": 1, "test": 2}

    def _subsample_sessions(
        self,
        sampling_intervals: dict,
        split: Literal["train", "valid", "test"],
    ) -> dict:
        """Deterministically keep a fraction of recordings for *split*.

        Uses a seeded shuffle so the subset is reproducible but independent
        across splits. Always keeps at least one recording.
        """
        pct = self._session_pct.get(split, 1.0)
        if pct >= 1.0:
            return sampling_intervals

        rids = sorted(sampling_intervals.keys())
        n_keep = max(1, math.ceil(len(rids) * pct))
        if n_keep >= len(rids):
            return sampling_intervals

        rng = torch.Generator().manual_seed(
            self.seed + self._SPLIT_SEED_OFFSETS[split] + 1000
        )
        perm = torch.randperm(len(rids), generator=rng).tolist()
        keep = set(rids[i] for i in perm[:n_keep])

        logger.info(
            "session_pct[%s]=%.2f: keeping %d / %d recordings",
            split,
            pct,
            n_keep,
            len(rids),
        )

        return {
            rid: ivl for rid, ivl in sampling_intervals.items() if rid in keep
        }

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
        sampling_intervals = self._filter_intervals(sampling_intervals)
        if self._session_pct:
            sampling_intervals = self._subsample_sessions(
                sampling_intervals, split
            )

        split_seed = self.seed + self._SPLIT_SEED_OFFSETS[split]
        gen = torch.Generator().manual_seed(split_seed)
        trainer = getattr(self, "_trainer", None)
        world_size = int(getattr(trainer, "world_size", 1))
        global_rank = int(getattr(trainer, "global_rank", 0))

        if self.window_lengths is not None:
            batch_sampler = VariableLengthBatchSampler(
                sampling_intervals=sampling_intervals,
                window_lengths=self.window_lengths,
                batch_size=self.batch_size,
                drop_last=(split == "train"),
                generator=gen,
            )
            if split != "train":
                batch_sampler = DeterministicSamplerWrapper(
                    batch_sampler, split_seed
                )
            if world_size > 1:
                batch_sampler = DistributedSamplerWrapper(
                    batch_sampler,
                    num_replicas=world_size,
                    rank=global_rank,
                    drop_last=(split == "train"),
                )
            return DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
                worker_init_fn=_disable_gc_in_worker
                if self.num_workers > 0
                else None,
            )

        sampler = self.sampler_class(
            sampling_intervals=sampling_intervals,
            window_length=self.sequence_length,
            drop_short=True,
            generator=gen,
        )
        if split != "train":
            sampler = DeterministicSamplerWrapper(sampler, split_seed)
        if world_size > 1:
            sampler = DistributedSamplerWrapper(
                sampler,
                num_replicas=world_size,
                rank=global_rank,
                drop_last=(split == "train"),
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
            worker_init_fn=_disable_gc_in_worker
            if self.num_workers > 0
            else None,
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
