from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026,
    NeurosoftMonkeys2026,
)
from auditorydecoding.data.neurosoft_pipeline import (
    ON_VS_OFF_TO_ID,
    STIM_FREQUENCY_TO_ID,
)
from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.readout_specs import clone_readout_spec
from foundry.data.datasets.modalities import MappedCrossEntropyLoss

import torch
from torch.utils.data import DataLoader
from torch_brain.registry import MODALITY_REGISTRY

from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.data import collate

from temporaldata import Interval

import logging
import numpy as np

from typing import Optional, Callable, Literal, Type

logger = logging.getLogger(__name__)

# Predefined frequency groupings
FREQ_GROUPINGS = {
    "3band": {
        "stim_500Hz": "low",
        "stim_800Hz": "low",
        "stim_1000Hz": "medium",
        "stim_2000Hz": "medium",
        "stim_5000Hz": "high",
        "stim_8000Hz": "high",
    },
    # All 15 possible ways to pair 6 items into 3 unordered pairs,
    # mapping each pair to a unique class ("class1", "class2", "class3").
    # Each mapping below is a distinct pairing arrangement.
    "3band_rnd_a": {  # (500,1000) (800,5000) (2000,8000)
        "stim_500Hz": "class1", "stim_1000Hz": "class1",
        "stim_800Hz": "class2", "stim_5000Hz": "class2",
        "stim_2000Hz": "class3", "stim_8000Hz": "class3",
    },
    "3band_rnd_b": {  # (500,1000) (800,8000) (2000,5000)
        "stim_500Hz": "class1", "stim_1000Hz": "class1",
        "stim_800Hz": "class2", "stim_8000Hz": "class2",
        "stim_2000Hz": "class3", "stim_5000Hz": "class3",
    },
    "3band_rnd_c": {  # (500,2000) (800,5000) (1000,8000)
        "stim_500Hz": "class1", "stim_2000Hz": "class1",
        "stim_800Hz": "class2", "stim_5000Hz": "class2",
        "stim_1000Hz": "class3", "stim_8000Hz": "class3",
    },
    "3band_rnd_d": {  # (500,2000) (800,8000) (1000,5000)
        "stim_500Hz": "class1", "stim_2000Hz": "class1",
        "stim_800Hz": "class2", "stim_8000Hz": "class2",
        "stim_1000Hz": "class3", "stim_5000Hz": "class3",
    },
    "3band_rnd_e": {  # (500,5000) (800,1000) (2000,8000)
        "stim_500Hz": "class1", "stim_5000Hz": "class1",
        "stim_800Hz": "class2", "stim_1000Hz": "class2",
        "stim_2000Hz": "class3", "stim_8000Hz": "class3",
    },
    "3band_rnd_f": {  # (500,5000) (800,2000) (1000,8000)
        "stim_500Hz": "class1", "stim_5000Hz": "class1",
        "stim_800Hz": "class2", "stim_2000Hz": "class2",
        "stim_1000Hz": "class3", "stim_8000Hz": "class3",
    },
    "3band_rnd_g": {  # (500,8000) (800,1000) (2000,5000)
        "stim_500Hz": "class1", "stim_8000Hz": "class1",
        "stim_800Hz": "class2", "stim_1000Hz": "class2",
        "stim_2000Hz": "class3", "stim_5000Hz": "class3",
    },
    "3band_rnd_h": {  # (500,8000) (800,2000) (1000,5000)
        "stim_500Hz": "class1", "stim_8000Hz": "class1",
        "stim_800Hz": "class2", "stim_2000Hz": "class2",
        "stim_1000Hz": "class3", "stim_5000Hz": "class3",
    },
    "2band": {
        "stim_500Hz": "low",
        "stim_800Hz": "low",
        "stim_5000Hz": "high",
        "stim_8000Hz": "high",
    },
    "2band_rnd_a": {
        "stim_500Hz": "class1",
        "stim_800Hz": "class2",
        "stim_5000Hz": "class1",
        "stim_8000Hz": "class2",
    },
    "2band_rnd_b": {
        "stim_500Hz": "class1",
        "stim_800Hz": "class2",
        "stim_5000Hz": "class2",
        "stim_8000Hz": "class1",
    },
}

# Default grouping (3-band) - used when explicitly requested or as a fallback
FREQ_TO_LABEL = FREQ_GROUPINGS["3band"]

LABEL_TO_ID = {
  "low": 0,
  "medium": 1,
  "high": 2,
}


def format_acoustic_stim_display_names(names: list[str]) -> list[str]:
    """Strip stim_ prefix from frequency names for display in confusion matrices.
    
    Args:
        names: List of frequency strings (e.g., ["stim_500Hz", "stim_800Hz"])
    
    Returns:
        List with stim_ prefix removed (e.g., ["500Hz", "800Hz"])
    """
    return [n.removeprefix("stim_") if n.startswith("stim_") else n for n in names]


def build_acoustic_stim_label_mapping(
    classes: Optional[list[str]],
    freq_grouping: Optional[dict[str, str]] = None,
) -> tuple[dict[int, int], list[str]]:
    """Build label mapping and effective class names for acoustic stimulus task.

    Remaps frequencies to class indices and optionally groups into bands.

    Args:
        classes: List of frequency strings (e.g. ["stim_500Hz", "stim_1000Hz"])
                 or None for all frequencies.
        freq_grouping: Dict mapping frequency names to group names (e.g., {"stim_500Hz": "low"}).
                      If None, raw mode — dense remap raw_id -> 0..N-1.

    Returns:
        Tuple of (raw_freq_id -> class_id mapping dict, effective class names list).
        - If grouping: ({raw_id: band_id, ...}, ["band1", "band2", ...])
        - If raw mode: ({raw_id: 0..N-1, ...}, [selected_frequency_names])
        E.g. with grouping: ({0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}, ["low", "medium", "high"])
        E.g. raw: ({0: 0, 2: 1, 5: 2}, ["stim_500Hz", "stim_1000Hz", "stim_8000Hz"])
    """
    if classes is None:
        # No filtering: return no mapping (signals full registry spec)
        return {}, []

    # Check if no grouping (freq_grouping is None)
    if freq_grouping is None:
        # Raw mode: dense remap raw_id -> 0..N-1, use frequency names as labels
        selected_freqs = []
        for class_name in classes:
            if class_name in STIM_FREQUENCY_TO_ID:
                selected_freqs.append(class_name)
        
        if not selected_freqs:
            raise ValueError(
                f"No valid frequencies found in classes: {classes}. "
                f"Available: {list(STIM_FREQUENCY_TO_ID.keys())}"
            )
        
        # Sort by raw ID for consistent ordering
        selected_freqs_sorted = sorted(
            selected_freqs, 
            key=lambda f: STIM_FREQUENCY_TO_ID[f]
        )
        
        # Build dense remap: raw_id -> 0..N-1
        mapping = {
            STIM_FREQUENCY_TO_ID[f]: i
            for i, f in enumerate(selected_freqs_sorted)
        }
        return mapping, selected_freqs_sorted

    # Use provided grouping dict
    grouping = freq_grouping

    # Get raw frequency IDs from the selected classes
    selected_raw_ids = set()
    selected_bands = set()

    for class_name in classes:
        if class_name in STIM_FREQUENCY_TO_ID:
            raw_id = STIM_FREQUENCY_TO_ID[class_name]
            selected_raw_ids.add(raw_id)
            if class_name in grouping:
                band = grouping[class_name]
                selected_bands.add(band)

    if not selected_raw_ids:
        raise ValueError(
            f"No valid frequencies found in classes: {classes}. "
            f"Available: {list(STIM_FREQUENCY_TO_ID.keys())}"
        )

    # Build ordered list of bands using LABEL_TO_ID order for consistency
    selected_band_order = [band for band in sorted(selected_bands)]
    # Sort by LABEL_TO_ID values to maintain a consistent order
    selected_band_order.sort(key=lambda b: LABEL_TO_ID.get(b, float('inf')))

    # Build mapping from raw ID -> band ID
    mapping = {}
    for freq_str, raw_id in STIM_FREQUENCY_TO_ID.items():
        if raw_id in selected_raw_ids and freq_str in grouping:
            band = grouping[freq_str]
            band_id = selected_band_order.index(band)
            mapping[raw_id] = band_id

    return mapping, selected_band_order


class NeurosoftDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "on_vs_off": ["neurosoft_on_vs_off"],
        "acoustic_stim": ["neurosoft_acoustic_stim"],
    }

    READOUT_CLASS_NAMES: dict[str, list[str]] = {
        "on_vs_off": list(ON_VS_OFF_TO_ID.keys()),
        "acoustic_stim": [
            k
            for k, _ in sorted(STIM_FREQUENCY_TO_ID.items(), key=lambda x: x[1])
        ],
    }

    def __init__(
        self,
        dataset_class: Type[NeurosoftDataset],
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sequence_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        split_type: Optional[
            Literal[
                "intersubject",
                "intersession",
                "intrasession",
                "intrasession-block",
                "intrasession-causal",
            ]
        ] = None,
        task_type: Optional[
            Literal["on_vs_off", "acoustic_stim"]
        ] = "on_vs_off",
        fold_number: Optional[int] = 0,
        recording_ids: Optional[list[str]] = None,
        classes: Optional[list[str]] = None,
        freq_grouping: Optional[str | dict[str, str]] = None,
    ):
        dataset_kwargs = {
            "recording_ids": recording_ids,
            "split_type": split_type,
            "task_type": task_type,
            "fold_num": fold_number,
        }
        super().__init__(
            dataset_class=dataset_class,
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

        self.classes = classes
        
        # Resolve freq_grouping
        if freq_grouping is None:
            self.freq_grouping = None  # No grouping
        elif isinstance(freq_grouping, str):
            if freq_grouping not in FREQ_GROUPINGS:
                raise ValueError(
                    f"Unknown freq_grouping '{freq_grouping}'. "
                    f"Available: {list(FREQ_GROUPINGS.keys())}"
                )
            self.freq_grouping = FREQ_GROUPINGS[freq_grouping]
        elif isinstance(freq_grouping, dict):
            self.freq_grouping = freq_grouping
        else:
            raise TypeError(
                f"freq_grouping must be str, dict, or None, got {type(freq_grouping)}"
            )

    def get_effective_readout_specs(self) -> dict[str, any]:
        """Get readout specs with effective dims/losses for class filtering.

        For acoustic_stim with classes set, remaps frequencies to dense indices
        and returns a cloned spec with mapped loss function.
        """
        specs = {}
        readout_names = self.get_readout_specs_for_task(self.task_type)

        for readout_name in readout_names:
            base_spec = MODALITY_REGISTRY[readout_name]

            # For acoustic_stim with class filtering, apply remapping
            if (
                readout_name == "neurosoft_acoustic_stim"
                and self.classes is not None
            ):
                mapping, effective_names = build_acoustic_stim_label_mapping(
                    self.classes,
                    freq_grouping=self.freq_grouping,
                )
                effective_dim = len(effective_names)
                loss_fn = MappedCrossEntropyLoss(mapping)
                specs[readout_name] = clone_readout_spec(
                    base_spec, dim=effective_dim, loss_fn=loss_fn
                )
                logger.info(
                    "Effective readout %s: dim=%d loss=%s (data.classes=%s)",
                    readout_name,
                    effective_dim,
                    type(loss_fn).__name__,
                    self.classes,
                )
            else:
                # Return base spec unchanged
                specs[readout_name] = base_spec

        return specs

    def get_effective_class_names_for_task(
        self, task_type: str
    ) -> dict[str, list[str]]:
        """Get class names aligned with effective dimensions.

        For acoustic_stim with classes set, returns remapped names with stim_
        prefix removed for confusion matrix display. For full 26-class runs,
        also strips the prefix for consistency.
        """
        readout_names = self.TASK_TO_READOUT.get(task_type, [])
        class_names = {}

        for readout_name in readout_names:
            if readout_name == "neurosoft_acoustic_stim":
                if self.classes is not None:
                    # Filtered mode
                    mapping, effective_names = build_acoustic_stim_label_mapping(
                        self.classes,
                        freq_grouping=self.freq_grouping,
                    )
                    # Apply display formatting (strip stim_ prefix for confusion matrix)
                    if effective_names:
                        # If names are frequency strings, strip stim_ prefix
                        # (band names like "low" won't be affected)
                        display_names = format_acoustic_stim_display_names(effective_names)
                        class_names[readout_name] = display_names
                else:
                    # Full 26-class mode
                    if task_type in self.READOUT_CLASS_NAMES:
                        default_names = self.READOUT_CLASS_NAMES[task_type]
                        # Also strip stim_ for consistency across runs
                        class_names[readout_name] = format_acoustic_stim_display_names(default_names)
            else:
                # Other readouts (e.g., on_vs_off)
                if task_type in self.READOUT_CLASS_NAMES:
                    class_names[readout_name] = self.READOUT_CLASS_NAMES[task_type]

        return class_names

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())

    def validate_binary_class_coverage(
        self,
        split: Literal["train", "valid", "test"] = "train",
    ) -> None:
        """Fail fast when a binary acoustic_stim run lacks required frequencies.

        For ``acoustic_stim`` with exactly two entries in ``data.classes``, each
        recording must contain trials for both stimulation frequencies in the
        given split (after class filtering). Raises :class:`ValueError` otherwise.
        """
        if self.task_type != "acoustic_stim":
            return
        if self.classes is None or len(self.classes) != 2:
            return
        if self.dataset is None:
            raise RuntimeError(
                "Call setup() before validate_binary_class_coverage()"
            )

        required = set(self.classes)
        sampling_intervals = self.get_sampling_intervals(split)
        failures: list[tuple[str, list[str]]] = []

        for rid in self.dataset.recording_ids:
            intervals = sampling_intervals.get(rid)
            if intervals is None or len(intervals) == 0:
                failures.append((rid, sorted(required)))
                continue

            if not hasattr(intervals, "behavior_labels"):
                failures.append((rid, sorted(required)))
                continue

            present = set(np.unique(intervals.behavior_labels).tolist())
            missing = sorted(required - present)
            if missing:
                failures.append((rid, missing))

        if failures:
            lines = [
                "Binary classification requires both stimulation frequencies "
                f"in the '{split}' split, but some recordings are missing "
                f"trials (data.classes={list(self.classes)}):"
            ]
            for rid, missing in failures:
                lines.append(f"  - {rid}: missing {missing}")
            raise ValueError("\n".join(lines))

    def get_sampling_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict[str, Interval]:
        sampling_intervals = self.dataset.get_sampling_intervals(split)

        # Filter sampling intervals by classes if set
        # Note: keep behavior_ids as raw frequency IDs; MappedCrossEntropyLoss will remap them
        if self.classes is not None:
            for rid, intervals in sampling_intervals.items():
                classes_mask = np.isin(intervals.behavior_labels, self.classes)
                sampling_intervals[rid] = intervals.select_by_mask(classes_mask)

        return sampling_intervals

    def _create_dataloader(
        self, split: Literal["train", "valid", "test"]
    ) -> DataLoader:
        sampling_intervals = self.get_sampling_intervals(split)
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


class NeurosoftMinipigs2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMinipigs2026, **kwargs)


class NeurosoftMonkeys2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMonkeys2026, **kwargs)
