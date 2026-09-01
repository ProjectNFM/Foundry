"""Generic base DataModule for neural data with flexible composition.

This module provides a base LightningDataModule that works with any dataset
and optional tokenization/vocab initialization. It decouples data loading from
model-specific preprocessing.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Literal, Optional, Type

import numpy as np
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler
from lightning import LightningDataModule
from torch_brain.transforms import Compose

from foundry.data.fraction_manifest import (
    FractionManifest,
    FractionManifestBuilder,
    _canonical_hash,
)
from foundry.data.samplers import (
    FastRandomFixedWindowSampler,
    VariableLengthBatchSampler,
)
from foundry.tasks.class_weights import compute_class_weights_for_tasks
from foundry.tasks.classification_mapping import (
    filter_intervals_by_mapping,
    validate_task_mappings,
)

if TYPE_CHECKING:
    from foundry.data.normalization import RecordingChannelStats
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
        training_fraction: float | None = None,
        training_fraction_seed: int | None = None,
        training_fraction_task: str | None = None,
        training_fraction_min_class_support: int = 3,
        training_fraction_min_present_classes: int = 6,
        audit_json: str | None = None,
        audit_species: str | None = None,
        input_normalization: Optional[dict] = None,
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
            training_fraction: If set, select this fraction of training
                intervals using class-aware nested sampling.
            training_fraction_seed: Seed for fraction permutations.
                Defaults to ``self.seed``.
            training_fraction_task: Task name whose class mapping drives
                the fraction selection.  Required when multiple tasks have
                class mappings.
            training_fraction_min_class_support: Minimum per-class support
                for a fraction to be available.
            training_fraction_min_present_classes: Minimum number of present
                classes for a fraction to be available.
            audit_json: Optional verified Phase 0 audit artifact. When set,
                runtime split hashes must match its recording entry before
                fraction manifests are accepted.
            audit_species: Species key used to disambiguate duplicated BIDS
                recording IDs in the Phase 0 audit.
            input_normalization: Optional normalization configuration dict.
                When ``mode`` is ``"recording_train_channel_zscore"`` or
                ``"recording_train_global_zscore"``, frozen train-only
                statistics are fitted per recording and applied to all splits.
                Keys: ``mode``, ``supported_modalities``,
                ``scale_floor``, ``accumulator_dtype``.
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

        self.training_fraction = training_fraction
        self.training_fraction_seed = training_fraction_seed
        self.training_fraction_task = training_fraction_task
        self.training_fraction_min_class_support = (
            training_fraction_min_class_support
        )
        self.training_fraction_min_present_classes = (
            training_fraction_min_present_classes
        )
        self.audit_json = audit_json
        self.audit_species = audit_species

        self._fraction_manifests: dict[str, FractionManifest] = {}
        self._fraction_split_hashes: dict[str, dict[str, str]] = {}
        self._fraction_split_class_counts: dict[
            str, dict[str, dict[str, int]]
        ] = {}
        self._fraction_audit_records: dict[str, dict] = {}
        self._fraction_audit_artifact_sha256: str | None = None

        self._user_transforms: list[Callable] = list(transforms or [])
        self._tokenizer = tokenizer
        self._standardizer: Optional[Callable] = None
        self._input_normalization_cfg = input_normalization
        self._normalization_stats: Optional[
            dict[str, "RecordingChannelStats"]
        ] = None

        self.transform = self._build_transform_list()
        self.dataset = None

    # -- Transform pipeline management -----------------------------------------

    def _build_transform_list(self) -> list[Callable] | None:
        """Assemble the non-required transform pipeline from components.

        Order: ``user_transforms -> [standardizer] -> [tokenizer]``.
        Required dataset transforms are prepended separately in
        :meth:`setup` / :meth:`_rebuild_dataset_transform`.
        """
        parts = list(self._user_transforms)
        if self._standardizer is not None:
            parts.append(self._standardizer)
        if self._tokenizer is not None:
            parts.append(self._tokenizer)
        return parts if parts else None

    def _rebuild_dataset_transform(self) -> None:
        """Rebuild the dataset's composed transform from current components."""
        if self.dataset is None:
            return
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

    def setup(self, stage: Optional[str] = None):
        """Setup the DataModule.

        Args:
            stage: Stage to setup the DataModule for ('fit', 'test', 'validate').
        """
        if self.dataset is None:
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

        self._maybe_fit_normalization()

    def set_tokenizer(self, tokenizer: Optional[Callable]) -> None:
        """Replace the tokenizer in the transform pipeline.

        Can be called before or after :meth:`setup`.  When the dataset
        already exists, its transform is rebuilt in-place.
        """
        self._tokenizer = tokenizer
        self.transform = self._build_transform_list()
        self._rebuild_dataset_transform()

    # -- Input normalization ---------------------------------------------------

    def _maybe_fit_normalization(self) -> None:
        """Fit train-only recording statistics if normalization is configured."""
        if self._input_normalization_cfg is None:
            return
        mode = self._input_normalization_cfg.get("mode", "disabled")
        if mode == "disabled":
            return
        if mode not in {
            "recording_train_channel_zscore",
            "recording_train_global_zscore",
        }:
            raise ValueError(
                f"Unsupported input_normalization.mode={mode!r}; "
                "expected 'disabled', 'recording_train_channel_zscore', or "
                "'recording_train_global_zscore'"
            )
        if self._standardizer is not None:
            return
        # Hyperparameter discovery creates a temporary datamodule before task
        # configs are attached.  A fraction manifest cannot be resolved there,
        # so fitting at that point would use the wrong train population.
        if self.training_fraction is not None and not self._task_configs:
            logger.debug(
                "Deferring normalization until training-fraction task configs "
                "are attached"
            )
            return
        self.prepare_training_fraction_manifests()
        self._fit_and_insert_normalization()

    def _fit_and_insert_normalization(self) -> None:
        """Fit configured recording statistics and insert standardizer."""
        from foundry.data.normalization import (
            fit_recording_global_stats,
            fit_recording_stats,
        )
        from foundry.data.transforms import RecordingChannelStandardize

        cfg = self._input_normalization_cfg
        mode = cfg["mode"]
        supported_modalities = frozenset(
            str(modality).lower()
            for modality in cfg.get(
                "supported_modalities", ["eeg", "ecog", "seeg", "ieeg"]
            )
        )
        from foundry.data.utils import NEURAL_MODALITIES

        if (
            not supported_modalities
            or not supported_modalities <= NEURAL_MODALITIES
        ):
            raise ValueError(
                "input_normalization.supported_modalities must be a non-empty "
                f"subset of {sorted(NEURAL_MODALITIES)}"
            )
        scale_floor = float(cfg.get("scale_floor", 1e-8))
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError(
                "input_normalization.scale_floor must be a finite positive value"
            )
        accumulator_dtype_name = str(cfg.get("accumulator_dtype", "float64"))
        if accumulator_dtype_name != "float64":
            raise ValueError(
                "input_normalization.accumulator_dtype must be 'float64'"
            )

        train_intervals = self._effective_sampling_intervals("train")

        stats_by_recording: dict[str, RecordingChannelStats] = {}
        for rid, intervals in train_intervals.items():
            if len(intervals) == 0:
                continue
            fit_stats = (
                fit_recording_global_stats
                if mode == "recording_train_global_zscore"
                else fit_recording_stats
            )
            stats = fit_stats(
                self.dataset,
                rid,
                intervals,
                supported_modalities=supported_modalities,
                scale_floor=scale_floor,
                accumulator_dtype=np.float64,
            )
            stats_by_recording[rid] = stats

        all_rids: set[str] = set()
        for split in ("train", "valid", "test"):
            ivls = self._effective_sampling_intervals(split)
            for rid, intervals in ivls.items():
                if len(intervals) > 0:
                    all_rids.add(rid)

        missing = all_rids - set(stats_by_recording)
        if missing:
            raise RuntimeError(
                f"Recording(s) {sorted(missing)} appear in validation/test "
                f"splits but have no train intervals. Within-recording "
                f"normalization requires train-derived statistics for every "
                f"recording."
            )

        if not stats_by_recording:
            raise RuntimeError(
                "No recordings with non-empty train intervals found; "
                "cannot fit normalization statistics"
            )

        self._standardizer = RecordingChannelStandardize(
            stats_by_recording=stats_by_recording,
            supported_modalities=supported_modalities,
            scale_floor=scale_floor,
        )
        self._normalization_stats = stats_by_recording

        self.transform = self._build_transform_list()
        self._rebuild_dataset_transform()

        logger.info(
            "Fitted input normalization for %d recording(s) "
            "(mode=%s, scale_floor=%.1e)",
            len(stats_by_recording),
            mode,
            scale_floor,
        )

    @property
    def normalization_stats(
        self,
    ) -> Optional[dict[str, "RecordingChannelStats"]]:
        """Read-only view of fitted normalization statistics, if any."""
        return (
            dict(self._normalization_stats)
            if self._normalization_stats
            else None
        )

    @property
    def input_normalization_config(self) -> Optional[dict]:
        """Read-only view of the input normalization configuration."""
        if self._input_normalization_cfg is None:
            return None
        return dict(self._input_normalization_cfg)

    def write_normalization_artifacts(
        self, output_dir: str, *, git_sha: str | None = None
    ) -> dict[str, str] | None:
        """Write frozen normalization stats and return their artifact metadata."""
        if not self._normalization_stats:
            return None
        from foundry.data.normalization import save_normalization_stats

        train_intervals = self._effective_sampling_intervals("train")
        interval_payload = [
            {
                "recording_id": recording_id,
                "start": float(start).hex(),
                "end": float(end).hex(),
            }
            for recording_id, intervals in sorted(train_intervals.items())
            for start, end in zip(intervals.start, intervals.end)
        ]
        train_interval_hash = hashlib.sha256(
            json.dumps(interval_payload, separators=(",", ":")).encode()
        ).hexdigest()
        provenance = {
            "git_sha": git_sha,
            "train_interval_hash": train_interval_hash,
            "fraction_manifest_hashes": {
                recording_id: manifest.manifest_hash
                for recording_id, manifest in self._fraction_manifests.items()
            },
            "phase0_audit_artifact_sha256": self._fraction_audit_artifact_sha256,
        }
        stats_path, manifest_path = save_normalization_stats(
            self._normalization_stats,
            output_dir,
            self._input_normalization_cfg or {},
            provenance=provenance,
        )
        metadata = {
            "stats_path": str(stats_path),
            "manifest_path": str(manifest_path),
            "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "train_interval_hash": train_interval_hash,
        }
        logger.info(
            "Wrote input-normalization artifacts to %s (sha256=%s…)",
            stats_path.parent,
            metadata["stats_sha256"][:16],
        )
        return metadata

    # -- Class weights ---------------------------------------------------------

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

    # -- Training fraction manifests -------------------------------------------

    def _resolve_fraction_task_config(self) -> "TaskConfig":
        """Resolve the single task config used for fraction selection."""
        if not self._task_configs:
            raise ValueError(
                "task_configs must be provided for training_fraction"
            )
        candidates = {
            name: cfg
            for name, cfg in self._task_configs.items()
            if cfg.class_mapping is not None
        }
        if not candidates:
            raise ValueError(
                "No task config with a class_mapping found for "
                "training_fraction"
            )
        if self.training_fraction_task:
            if self.training_fraction_task not in candidates:
                raise ValueError(
                    f"training_fraction_task={self.training_fraction_task!r} "
                    f"not found among tasks with class mappings: "
                    f"{sorted(candidates)}"
                )
            return candidates[self.training_fraction_task]
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        raise ValueError(
            f"Multiple tasks have class mappings ({sorted(candidates)}); "
            f"set training_fraction_task to disambiguate"
        )

    @staticmethod
    def _compute_split_hash(recording_id: str, intervals) -> str:
        """Compute the Phase 0 canonical hash for one split's intervals.

        This deliberately hashes the raw interval dictionaries, matching
        :func:`tools.audit_neurosoft_sessions._interval_hash`.  Hashing the
        derived interval-identity digests would create a different algorithm
        and cannot verify the frozen audit artifact.
        """
        if len(intervals) == 0:
            return _canonical_hash([])
        starts = np.asarray(intervals.start)
        ends = np.asarray(intervals.end)
        labels = (
            np.asarray(intervals.behavior_labels)
            if hasattr(intervals, "behavior_labels")
            else np.array([""] * len(intervals))
        )
        payload = [
            {
                "recording_id": recording_id,
                "index": index,
                "start": float(start).hex(),
                "end": float(end).hex(),
                "label": str(label),
            }
            for index, (start, end, label) in enumerate(
                zip(starts, ends, labels)
            )
        ]
        return _canonical_hash(payload)

    @staticmethod
    def _split_class_counts(
        intervals, task_config: "TaskConfig"
    ) -> dict[str, int]:
        """Count mapped target classes in one split using the task mapping."""
        class_names = task_config.class_mapping.class_names
        counts = {name: 0 for name in class_names}
        if not hasattr(intervals, "behavior_labels") or not len(intervals):
            return counts
        mapped, _ = task_config.class_mapping.filter_and_remap(
            np.asarray(intervals.behavior_labels)
        )
        for class_id, class_name in enumerate(class_names):
            counts[class_name] = int((mapped == class_id).sum())
        return counts

    def _load_fraction_audit(self) -> dict | None:
        """Load the configured Phase 0 audit through its hash-verifying loader."""
        if self.audit_json is None:
            return None
        if not self.audit_species:
            raise ValueError(
                "audit_species is required when audit_json is configured to "
                "disambiguate recording IDs across species"
            )
        from foundry.config_resolvers import _load_neurosoft_audit

        audit = _load_neurosoft_audit(self.audit_json)
        self._fraction_audit_artifact_sha256 = audit["artifact_sha256"]
        return audit

    def _audit_recording(self, audit: dict, recording_id: str) -> dict:
        """Return the unique audited recording for this species and ID."""
        matches = [
            record
            for record in audit["recordings"]
            if record.get("recording_id") == recording_id
            and record.get("species") == self.audit_species
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Phase 0 audit must contain exactly one recording for "
                f"species={self.audit_species!r}, recording_id={recording_id!r}; "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _verify_audit_split_hashes(
        recording_id: str,
        actual: dict[str, str],
        audit_record: dict,
    ) -> None:
        """Fail before training when runtime split identities drift from Phase 0."""
        expected = audit_record.get("split_hashes")
        if not isinstance(expected, dict):
            raise RuntimeError(
                f"Phase 0 audit recording {recording_id!r} lacks split_hashes"
            )
        mismatches = {
            split: {
                "expected": expected.get(split),
                "actual": actual.get(split),
            }
            for split in ("train", "valid", "test")
            if expected.get(split) != actual.get(split)
        }
        if mismatches:
            raise RuntimeError(
                "Runtime split hashes differ from the verified Phase 0 audit "
                f"for recording {recording_id!r}: {mismatches}"
            )

    def prepare_training_fraction_manifests(self) -> None:
        """Build and cache fraction manifests for each recording.

        Must be called after :meth:`setup` and task config attachment.
        Fails loudly if any requested cell is unavailable.

        Idempotent: repeated calls reuse cached manifests.
        """
        if self.training_fraction is None:
            return
        if self.dataset is None:
            raise RuntimeError(
                "Call setup() before prepare_training_fraction_manifests()"
            )
        if self._fraction_manifests:
            return

        fraction_seed = (
            self.training_fraction_seed
            if self.training_fraction_seed is not None
            else self.seed
        )
        task_cfg = self._resolve_fraction_task_config()
        audit = self._load_fraction_audit()

        train_intervals = self.dataset.get_sampling_intervals(split="train")
        valid_intervals = self.dataset.get_sampling_intervals(split="valid")
        test_intervals = self.dataset.get_sampling_intervals(split="test")

        for rid, intervals in train_intervals.items():
            value_field = task_cfg.target_extractor["value_key"].split(".")[-1]
            if not hasattr(intervals, value_field):
                raise ValueError(
                    f"Training intervals for {rid} lack attribute "
                    f"{value_field!r}; cannot build fraction manifest"
                )

            builder = FractionManifestBuilder(
                recording_id=rid,
                train_intervals=intervals,
                class_mapping=task_cfg.class_mapping,
                seed=fraction_seed,
                min_class_support=self.training_fraction_min_class_support,
                min_present_classes=self.training_fraction_min_present_classes,
            )
            manifest = builder.build_fraction(self.training_fraction)

            if not manifest.available:
                raise RuntimeError(
                    f"Fraction {self.training_fraction} is unavailable for "
                    f"recording {rid}: {manifest.failure_reason}. "
                    f"This cell should not be in the sweep."
                )

            actual_split_hashes = {
                "train": self._compute_split_hash(rid, intervals),
                "valid": self._compute_split_hash(
                    rid, valid_intervals.get(rid, [])
                ),
                "test": self._compute_split_hash(
                    rid, test_intervals.get(rid, [])
                ),
            }
            self._fraction_split_hashes[rid] = actual_split_hashes
            self._fraction_split_class_counts[rid] = {
                "train": self._split_class_counts(intervals, task_cfg),
                "valid": self._split_class_counts(
                    valid_intervals.get(rid, []), task_cfg
                ),
                "test": self._split_class_counts(
                    test_intervals.get(rid, []), task_cfg
                ),
            }
            if audit is not None:
                audit_record = self._audit_recording(audit, rid)
                self._verify_audit_split_hashes(
                    rid, actual_split_hashes, audit_record
                )
                self._fraction_audit_records[rid] = deepcopy(audit_record)

            self._fraction_manifests[rid] = manifest

        logger.info(
            "Prepared fraction manifests for %d recordings "
            "(fraction=%.2f, seed=%d)",
            len(self._fraction_manifests),
            self.training_fraction,
            fraction_seed,
        )

    @property
    def fraction_manifests(self) -> dict[str, FractionManifest]:
        """Read-only view of cached fraction manifests by recording ID."""
        return dict(self._fraction_manifests)

    @property
    def fraction_split_hashes(self) -> dict[str, dict[str, str]]:
        """Read-only view of computed train/valid/test split hashes."""
        return dict(self._fraction_split_hashes)

    @property
    def fraction_split_class_counts(
        self,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Read-only view of mapped runtime class counts for every split."""
        return deepcopy(self._fraction_split_class_counts)

    @property
    def fraction_audit_records(self) -> dict[str, dict]:
        """Read-only view of the verified Phase 0 recording entries."""
        return deepcopy(self._fraction_audit_records)

    @property
    def fraction_audit_artifact_sha256(self) -> str | None:
        """SHA-256 of the verified Phase 0 artifact, if auditing is enabled."""
        return self._fraction_audit_artifact_sha256

    def _apply_fraction_selection(
        self,
        sampling_intervals: dict,
    ) -> dict:
        """Select only cached fraction indices from raw training intervals."""
        if not self._fraction_manifests:
            return sampling_intervals

        result = {}
        for rid, intervals in sampling_intervals.items():
            manifest = self._fraction_manifests.get(rid)
            if manifest is None:
                result[rid] = intervals
                continue
            n = len(intervals)
            mask = np.zeros(n, dtype=bool)
            for idx in manifest.selected_indices:
                if idx < n:
                    mask[idx] = True
            result[rid] = intervals.select_by_mask(mask)
        return result

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

    def _effective_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"],
    ) -> dict:
        """Return exactly the intervals that the split's loader will sample."""
        sampling_intervals = self.dataset.get_sampling_intervals(split=split)
        if split == "train" and self._fraction_manifests:
            sampling_intervals = self._apply_fraction_selection(
                sampling_intervals
            )
        sampling_intervals = self._filter_intervals(sampling_intervals)
        if self._session_pct:
            sampling_intervals = self._subsample_sessions(
                sampling_intervals, split
            )
        return sampling_intervals

    def _create_dataloader(
        self, split: Literal["train", "valid", "test"]
    ) -> DataLoader:
        """Create a DataLoader for a given split.

        Args:
            split: One of 'train', 'valid', or 'test'.

        Returns:
            DataLoader for the split.
        """
        sampling_intervals = self._effective_sampling_intervals(split)

        split_seed = self.seed + self._SPLIT_SEED_OFFSETS[split]
        gen = torch.Generator().manual_seed(split_seed)

        if self.window_lengths is not None:
            batch_sampler = VariableLengthBatchSampler(
                sampling_intervals=sampling_intervals,
                window_lengths=self.window_lengths,
                batch_size=self.batch_size,
                drop_last=(split == "train"),
                generator=gen,
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
