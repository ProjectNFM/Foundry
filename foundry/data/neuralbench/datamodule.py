"""NeuralBenchDataModule: LightningDataModule for NeuralBench tasks.

Replaces :class:`~foundry.data.datamodules.base.NeuralDataModule` for
NeuralBench tasks.  Uses NeuralSet's native split assignments and
pre-windowed epochs with index-based sampling (not window-based).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Optional

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_brain.batching import collate

from foundry.data.neuralbench.adapter import NeuralSetAdapter, P3_LABEL_MAP

logger = logging.getLogger(__name__)


def _require_neuralbench() -> None:
    try:
        import neuralbench  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "neuralbench is required for NeuralBench integration.  "
            "Install with: uv sync --group neuralbench"
        ) from exc


class NeuralBenchDataModule(LightningDataModule):
    """DataModule that uses NeuralSet to serve NeuralBench tasks.

    Manages the NeuralBench data pipeline: loads the effective task
    configuration, builds NeuralSet studies/segmenters, wraps each split's
    dataset with :class:`NeuralSetAdapter`, and provides standard Lightning
    dataloaders.

    The resulting Data objects are compatible with Foundry's tokenizers
    and collation (``torch_brain.batching.collate``).

    Args:
        task: NeuralBench task ID (e.g. ``"p3"``).
        dataset: NeuralBench dataset name (e.g. ``"korczowski2014a"``).
        cache_dir: NeuralSet data cache path.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin GPU memory in DataLoaders.
        tokenizer: Optional tokenizer callable (set later via
            :meth:`set_tokenizer`).
        task_configs: Per-task :class:`TaskConfig` dicts; set by
            ``main.py`` after instantiation.
        label_map: Mapping from one-hot argmax index to string label.
        label_attr: Interval attribute name for labels.
        interval_name: Interval name on the Data object.
        session_prefix: Prefix for synthetic session IDs.
    """

    def __init__(
        self,
        task: str,
        dataset: str,
        cache_dir: str = "",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer: Optional[Callable] = None,
        task_configs: Optional[dict] = None,
        label_map: Optional[dict[int, str]] = None,
        label_attr: str = "targets",
        interval_name: str = "p300_trials",
        session_prefix: Optional[str] = None,
    ):
        super().__init__()
        self.nb_task = task
        self.nb_dataset_name = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._tokenizer = tokenizer
        self._task_configs = task_configs
        self.label_map = label_map if label_map is not None else P3_LABEL_MAP
        self.label_attr = label_attr
        self.interval_name = interval_name
        self.session_prefix = (
            session_prefix if session_prefix is not None else f"nb/{task}"
        )

        self._train_adapter: NeuralSetAdapter | None = None
        self._val_adapter: NeuralSetAdapter | None = None
        self._test_adapter: NeuralSetAdapter | None = None
        self._session_ids: list[str] = []
        self._channel_ids: list[str] = []
        self._channel_names: list[str] = []
        self._num_channels: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        if self._train_adapter is not None:
            return

        _require_neuralbench()

        from exca import ConfDict
        from neuralbench.data import Data as NBData
        from neuralbench.experiment_config import prepare_task_configs
        from neuralbench.registry import DEFAULTS_DIR, load_yaml_config

        default_config = load_yaml_config(DEFAULTS_DIR / "config.yaml")
        grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))

        configs = prepare_task_configs(
            ConfDict(default_config),
            grid,
            device="eeg",
            task_name=self.nb_task,
            use_task_grid=False,
            debug=False,
            force=False,
            prepare=False,
            download=False,
            models=[None],
            datasets=[self.nb_dataset_name],
            quiet=True,
        )
        cfg = configs[0]

        if self.cache_dir:
            cfg["data"]["study"]["source"]["path"] = self.cache_dir

        nb_data = NBData(**cfg["data"])
        loaders = nb_data.prepare()

        channel_names = list(nb_data.neuro._channels.keys())
        sampling_rate = float(nb_data.neuro.frequency)

        self._channel_names = channel_names
        self._num_channels = len(channel_names)

        logger.info(
            "NeuralBench %s/%s: %d channels @ %.0f Hz, splits=%s",
            self.nb_task,
            self.nb_dataset_name,
            self._num_channels,
            sampling_rate,
            list(loaders.keys()),
        )

        split_name_map = {
            "train": "_train_adapter",
            "val": "_val_adapter",
            "test": "_test_adapter",
        }
        for split_name, attr_name in split_name_map.items():
            if split_name not in loaders:
                continue
            adapter = NeuralSetAdapter(
                nb_dataset=loaders[split_name].dataset,
                channel_names=channel_names,
                sampling_rate=sampling_rate,
                split=split_name,
                label_map=self.label_map,
                label_attr=self.label_attr,
                interval_name=self.interval_name,
                session_prefix=self.session_prefix,
                transform=self._tokenizer,
            )
            setattr(self, attr_name, adapter)
            logger.info("  %s: %d samples", split_name, len(adapter))

        self._collect_metadata(loaders)

    def _collect_metadata(self, loaders: dict) -> None:
        """Collect session and channel IDs from trigger metadata."""
        session_id_set: set[str] = set()

        for split_name, loader in loaders.items():
            ds = loader.dataset
            triggers = getattr(ds, "triggers", None)
            if triggers is not None and "subject" in triggers.columns:
                for subj in triggers["subject"].unique():
                    session_id_set.add(f"{self.session_prefix}/{subj}")
            else:
                logger.warning(
                    "Split '%s' has no trigger metadata; scanning samples "
                    "for subject IDs (this may be slow).",
                    split_name,
                )
                adapter = getattr(
                    self,
                    {
                        "train": "_train_adapter",
                        "val": "_val_adapter",
                        "test": "_test_adapter",
                    }.get(split_name, ""),
                    None,
                )
                if adapter is not None:
                    session_id_set.update(_scan_session_ids(adapter))

        self._session_ids = sorted(session_id_set)
        self._channel_ids = sorted(
            f"{sid}/{ch}"
            for sid in self._session_ids
            for ch in self._channel_names
        )

        logger.info(
            "  Metadata: %d sessions, %d unique channel IDs",
            len(self._session_ids),
            len(self._channel_ids),
        )

    # ------------------------------------------------------------------
    # Metadata interface (for VocabInitializerCallback and main.py)
    # ------------------------------------------------------------------

    def get_recording_ids(self) -> list[str]:
        """Return all session IDs across splits."""
        return list(self._session_ids)

    def get_channel_ids(self) -> list[str]:
        """Return all unique channel IDs across sessions."""
        return list(self._channel_ids)

    def get_session_configs(self) -> dict[str, int]:
        """Return ``{session_id: num_channels}`` mapping."""
        return {sid: self._num_channels for sid in self._session_ids}

    def get_num_channels(self) -> int:
        """Return the number of EEG channels."""
        return self._num_channels

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        """Compute inverse-frequency class weights from the training labels.

        Mirrors :meth:`NeuralDataModule.compute_class_weights` but counts
        labels directly from the NeuralBench training adapter instead of
        using the H5 dataset interface.
        """
        if self._train_adapter is None:
            raise RuntimeError("Call setup() before compute_class_weights()")
        if not self._task_configs:
            raise ValueError(
                "task_configs must be provided to compute class weights"
            )

        counts: Counter = Counter()
        ds = self._train_adapter.nb_dataset
        for i in range(len(ds)):
            sample = ds[i]
            target = sample["target"] if isinstance(sample, dict) else sample.data["target"]
            target_np = target.numpy() if hasattr(target, "numpy") else np.asarray(target)
            class_idx = int(np.argmax(target_np.flatten()))
            counts[class_idx] += 1

        weights: dict[str, list[float]] = {}
        total = sum(counts.values())
        for name, cfg in self._task_configs.items():
            if cfg.kind not in ("binary", "multiclass"):
                continue
            num_classes = cfg.output_dim
            task_weights = [
                (total / (num_classes * max(counts.get(i, 0), 1))) ** smoothing
                for i in range(num_classes)
            ]
            weights[name] = task_weights
            logger.info(
                "Class weights for %s (smoothing=%.2f): %s (counts: %s)",
                name,
                smoothing,
                [f"{w:.3f}" for w in task_weights],
                dict(sorted(counts.items())),
            )
        return weights

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def set_tokenizer(self, tokenizer: Optional[Callable]) -> None:
        """Replace the tokenizer transform on all split adapters."""
        self._tokenizer = tokenizer
        for adapter in self._iter_adapters():
            adapter.transform = tokenizer

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self._train_adapter is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        return self._make_dataloader(self._train_adapter, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_adapter is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        return self._make_dataloader(self._val_adapter, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_adapter is None:
            raise RuntimeError("Call setup() before test_dataloader()")
        return self._make_dataloader(self._test_adapter, shuffle=False)

    def _make_dataloader(
        self, adapter: NeuralSetAdapter, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            adapter,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_adapters(self):
        for adapter in (
            self._train_adapter,
            self._val_adapter,
            self._test_adapter,
        ):
            if adapter is not None:
                yield adapter


def _scan_session_ids(adapter: NeuralSetAdapter) -> set[str]:
    """Scan adapter samples to collect unique session IDs (slow fallback)."""
    ids: set[str] = set()
    seen_subjects: set[str] = set()
    stale_count = 0
    for i in range(len(adapter)):
        _, sample_data = adapter._get_sample_data(i)
        subject_key = adapter._subject_key_fn(
            adapter.nb_dataset[i], sample_data
        )
        if subject_key not in seen_subjects:
            seen_subjects.add(subject_key)
            ids.add(f"{adapter.session_prefix}/{subject_key}")
            stale_count = 0
        else:
            stale_count += 1
        if stale_count > 500:
            break
    return ids
