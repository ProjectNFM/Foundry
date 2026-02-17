from typing import TYPE_CHECKING, Optional, Literal

import numpy as np
from temporaldata import Data

from torch_brain.utils import np_string_prefix

if TYPE_CHECKING:
    from torch_brain.registry import ModalitySpec


class NeuralDatasetMixin:
    """
    Mixin class for dataset subclasses containing neural data.

    This mixin is modality-agnostic and works with any neural data source
    (EEG, iEEG, fMRI, PET, etc.). It provides methods for retrieving channel
    and session identifiers.

    Provides:
        - ``get_channel_ids()`` for retrieving IDs of all included channels.
        - ``get_recording_ids()`` for retrieving IDs of all recordings/sessions.
        - If the class attribute ``neural_dataset_mixin_uniquify_channel_ids`` is set to ``True``,
          channel IDs will be made unique across recordings by prefixing each channel ID with the
          corresponding ``session.id``. This helps avoid collisions when combining data from
          multiple sessions. (default: ``False``)
    """

    neural_dataset_mixin_uniquify_channel_ids: bool = False

    def get_recording_hook(self, data: Data):
        if self.neural_dataset_mixin_uniquify_channel_ids:
            data.channels.id = np_string_prefix(
                f"{data.session.id}/",
                data.channels.id.astype(str),
            )
        super().get_recording_hook(data)

    def get_channel_ids(self) -> list[str]:
        """Return a sorted list of all channel IDs across all recordings in the dataset."""
        ans = [
            self.get_recording(rid).channels.id for rid in self.recording_ids
        ]
        return np.sort(np.concatenate(ans)).tolist()

    def get_recording_ids(self) -> list[str]:
        """Return a sorted list of all recording IDs in the dataset."""
        return np.sort(np.array(self.recording_ids)).tolist()


class FoldedDatasetMixin:
    """Mixin for datasets that support fold-based cross-validation splits.
    
    Provides shared logic for intra-subject and inter-subject fold handling,
    reducing duplication across dataset implementations.
    """

    TASK_CONFIGS: dict[str, list[str]] = {}  # Override in subclass if needed

    def get_sampling_intervals_intra_subject(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        fold_number: Optional[int] = None,
        task_type: Optional[str] = None,
    ):
        """Get sampling intervals for intra-subject fold splits.
        
        Args:
            split: Which split to return ("train", "valid", "test")
            fold_number: Which fold to use
            task_type: Task configuration name (optional, if dataset supports task configs)
            
        Returns:
            Dictionary mapping recording IDs to their sampling intervals
        """
        if fold_number is None:
            return {
                rid: self.get_recording(rid).domain
                for rid in self.recording_ids
            }

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if task_type:
            key = f"splits.{task_type}.fold_{fold_number}.{split}"
        else:
            key = f"splits.fold_{fold_number}.{split}"

        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }

    def get_sampling_intervals_inter_subject(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        fold_number: Optional[int] = None,
        task_type: Optional[str] = None,
        movement_field: Optional[str] = None,
    ):
        """Get sampling intervals for inter-subject fold splits.
        
        Args:
            split: Which split to return ("train", "valid", "test")
            fold_number: Which fold to use
            task_type: Task configuration name (optional, for filtering movements)
            movement_field: Field name containing movement/task labels (optional)
            
        Returns:
            Dictionary mapping recording IDs to their sampling intervals
        """
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        key = f"splits.SubjectSplit_fold{fold_number}"
        res = {}

        for rid in self.recording_ids:
            recording = self.get_recording(rid)
            
            # Check if recording belongs to this split
            if recording.get_nested_attribute(key) != split:
                continue

            # If task_type is specified, filter by valid movements/tasks
            if task_type and movement_field and self.TASK_CONFIGS:
                valid_movements = self.TASK_CONFIGS.get(task_type, [])
                movements = recording.get_nested_attribute(movement_field)
                mask = np.isin(movements, valid_movements)
                intervals = recording.get_nested_attribute(movement_field)
                res[rid] = intervals.select_by_mask(mask)
            else:
                res[rid] = recording.domain

        return res


# Backward compatibility alias
class EEGDatasetMixin(NeuralDatasetMixin):
    """Deprecated: Use NeuralDatasetMixin instead."""

    @property
    def eeg_dataset_mixin_uniquify_channel_ids(self):
        return self.neural_dataset_mixin_uniquify_channel_ids

    @eeg_dataset_mixin_uniquify_channel_ids.setter
    def eeg_dataset_mixin_uniquify_channel_ids(self, value):
        self.neural_dataset_mixin_uniquify_channel_ids = value


class ModalityMixin:
    """Mixin for datasets that provide modality specifications."""

    MODALITIES: dict[str, int] = {}

    @classmethod
    def get_modality_specs(cls) -> dict[str, "ModalitySpec"]:
        """Return ModalitySpec objects for this dataset's modalities keyed by name."""
        from torch_brain.registry import MODALITY_REGISTRY

        return {name: MODALITY_REGISTRY[name] for name in cls.MODALITIES.keys()}

    @classmethod
    def get_modality_names(cls) -> list[str]:
        """Return names of available modalities."""
        return list(cls.MODALITIES.keys())

    @classmethod
    def get_modality(cls, name: str) -> "ModalitySpec":
        """Get a single modality spec by name."""
        if name not in cls.MODALITIES:
            available = cls.get_modality_names()
            raise ValueError(
                f"Unknown modality '{name}'. Available: {available}"
            )
        from torch_brain.registry import MODALITY_REGISTRY

        return MODALITY_REGISTRY[name]

    @classmethod
    def get_modalities(
        cls, names: list[str] | None = None
    ) -> list["ModalitySpec"]:
        """Get modality specs as list. If names is None, returns all available."""
        if names is None:
            names = list(cls.MODALITIES.keys())
        return [cls.get_modality(name) for name in names]

    def get_recording_hook(self, data: Data):
        """Set multitask_readout config based on dataset's supported modalities."""
        if not hasattr(data, "config") or data.config is None:
            data.config = {}
        data.config["multitask_readout"] = [
            {"readout_id": name} for name in self.MODALITIES.keys()
        ]
        super().get_recording_hook(data)


def combine_modalities(
    *dataset_tasks: tuple[type, list[str]],
) -> list["ModalitySpec"]:
    """Combine modalities from multiple datasets.

    Args:
        *dataset_tasks: Variable number of (dataset_class, modality_names) tuples

    Returns:
        List of ModalitySpec objects from all specified datasets

    Example:
        combine_modalities(
            (SchalkWolpawPhysionet2009, ["motor_imagery_5class"]),
            (KempSleepEDF2013, ["sleep_stage_5class"]),
        )
    """
    specs = []
    for dataset_cls, task_names in dataset_tasks:
        specs.extend(dataset_cls.get_modalities(task_names))
    return specs
