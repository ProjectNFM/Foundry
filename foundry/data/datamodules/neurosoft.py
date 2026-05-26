from typing import Callable, Literal, Optional, Type

import numpy as np
from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026,
    NeurosoftMonkeys2026,
)
from auditorydecoding.data.neurosoft_pipeline import (
    ON_VS_OFF_TO_ID,
    STIM_FREQUENCY_TO_ID,
)
from torch_brain.dataset.dataset import DatasetIndex

from foundry.data.datamodules.base import NeuralDataModule


LOGFREQ_TASK = "acoustic_stim_logfreq"


def _neurosoft_dataset_task_type(task_type: Optional[str]) -> Optional[str]:
    if task_type == LOGFREQ_TASK:
        return "acoustic_stim"
    return task_type


def _stim_label_to_frequency_hz(label: object) -> float | None:
    if isinstance(label, bytes):
        label = label.decode()
    label = str(label)

    if label == "stim_wn":
        return None
    if label.startswith("stim_") and label.endswith("Hz"):
        return float(label.removeprefix("stim_").removesuffix("Hz"))
    raise ValueError(f"Cannot derive frequency from acoustic label {label!r}")


def _tone_mask_and_frequencies(
    behavior_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = []
    frequencies = []
    for label in behavior_labels:
        frequency_hz = _stim_label_to_frequency_hz(label)
        is_tone = frequency_hz is not None
        mask.append(is_tone)
        if is_tone:
            frequencies.append(frequency_hz)
    return (
        np.asarray(mask, dtype=bool),
        np.asarray(frequencies, dtype=np.float32),
    )


def _all_tone_log_frequencies() -> np.ndarray:
    frequencies = [
        frequency_hz
        for label in STIM_FREQUENCY_TO_ID
        if (frequency_hz := _stim_label_to_frequency_hz(label)) is not None
    ]
    return np.log(np.asarray(frequencies, dtype=np.float32))


_LOGFREQ_VALUES = _all_tone_log_frequencies()
LOGFREQ_NORMALIZE_MEAN = float(_LOGFREQ_VALUES.mean())
LOGFREQ_NORMALIZE_STD = float(_LOGFREQ_VALUES.std())


def filter_acoustic_stim_tone_intervals(
    sampling_intervals: dict,
    dataset: Optional[NeurosoftDataset] = None,
) -> dict:
    """Remove white-noise acoustic trials from sampler intervals."""
    filtered = {}
    tone_ids = {
        label_id
        for label, label_id in STIM_FREQUENCY_TO_ID.items()
        if label != "stim_wn"
    }

    for recording_id, intervals in sampling_intervals.items():
        if hasattr(intervals, "behavior_labels"):
            mask, _ = _tone_mask_and_frequencies(intervals.behavior_labels)
        elif hasattr(intervals, "behavior_ids"):
            mask = np.isin(intervals.behavior_ids, list(tone_ids))
        else:
            if dataset is None:
                filtered[recording_id] = intervals
                continue
            trials = dataset.get_recording(recording_id).acoustic_stim_trials
            trials = trials.select_by_interval(intervals)
            mask, _ = _tone_mask_and_frequencies(trials.behavior_labels)
            filtered[recording_id] = trials.select_by_mask(mask)
            continue
        filtered[recording_id] = intervals.select_by_mask(mask)

    return filtered


class AddNeurosoftLogFrequencyTargets:
    """Attach ln(frequency Hz) targets to Neurosoft acoustic tone trials."""

    def __call__(self, data):
        if not hasattr(data, "acoustic_stim_trials"):
            raise ValueError("Data is missing acoustic_stim_trials")

        trials = data.acoustic_stim_trials
        if not hasattr(trials, "behavior_labels"):
            raise ValueError(
                "acoustic_stim_trials must include behavior_labels"
            )

        mask, frequencies_hz = _tone_mask_and_frequencies(
            np.asarray(trials.behavior_labels)
        )
        trials = trials.select_by_mask(mask)
        log_frequencies = np.log(frequencies_hz).astype(np.float32)
        trials.log_frequency_values = log_frequencies.reshape(-1, 1)
        data.acoustic_stim_trials = trials
        data.config = dict(getattr(data, "config", {}) or {})
        data.config["multitask_readout"] = [
            {
                "readout_id": "neurosoft_acoustic_stim_logfreq",
                "normalize_mean": LOGFREQ_NORMALIZE_MEAN,
                "normalize_std": LOGFREQ_NORMALIZE_STD,
            }
        ]
        return data


def _prepend_logfreq_transform(
    transforms: Optional[list[Callable]],
    task_type: Optional[str],
) -> Optional[list[Callable]]:
    if task_type != LOGFREQ_TASK:
        return transforms
    return [AddNeurosoftLogFrequencyTargets(), *(transforms or [])]


class NeurosoftMinipigsMonkeys2026:
    """Combined Neurosoft minipig + monkey dataset with namespaced ids."""

    SOURCES = {
        "minipigs": NeurosoftMinipigs2026,
        "monkeys": NeurosoftMonkeys2026,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        fold_num: Optional[int] = 0,
        split_type: Optional[str] = None,
        task_type: Optional[str] = "on_vs_off",
        minipigs_recording_ids: Optional[list[str]] = None,
        monkeys_recording_ids: Optional[list[str]] = None,
    ):
        dataset_task_type = _neurosoft_dataset_task_type(task_type)
        recording_ids = {
            "minipigs": minipigs_recording_ids,
            "monkeys": monkeys_recording_ids,
        }
        self.datasets = {
            source: dataset_class(
                root=root,
                transform=transform,
                fold_num=fold_num,
                split_type=split_type,
                task_type=dataset_task_type,
                recording_ids=recording_ids[source],
            )
            for source, dataset_class in self.SOURCES.items()
        }
        self.recording_ids = [
            self._join_recording_id(source, recording_id)
            for source, dataset in self.datasets.items()
            for recording_id in dataset.recording_ids
        ]

    @staticmethod
    def _join_recording_id(source: str, recording_id: str) -> str:
        return f"{source}/{recording_id}"

    def _split_recording_id(self, recording_id: str) -> tuple[str, str]:
        source, inner_recording_id = recording_id.split("/", 1)
        if source not in self.datasets:
            raise KeyError(f"Unknown Neurosoft source '{source}'")
        return source, inner_recording_id

    def __getitem__(self, index: DatasetIndex):
        source, inner_recording_id = self._split_recording_id(
            index.recording_id
        )
        return self.datasets[source][
            DatasetIndex(
                inner_recording_id,
                index.start,
                index.end,
                _namespace=source,
            )
        ]

    def get_recording(self, recording_id: str, _namespace: str = ""):
        source, inner_recording_id = self._split_recording_id(recording_id)
        namespace = source if not _namespace else f"{_namespace}/{source}"
        return self.datasets[source].get_recording(
            inner_recording_id,
            _namespace=namespace,
        )

    def get_sampling_intervals(self, split=None) -> dict:
        intervals = {}
        for source, dataset in self.datasets.items():
            intervals.update(
                {
                    self._join_recording_id(source, recording_id): interval
                    for recording_id, interval in dataset.get_sampling_intervals(
                        split
                    ).items()
                }
            )
        return intervals

    def get_channel_ids(self) -> list[str]:
        channel_ids = []
        for source, dataset in self.datasets.items():
            channel_ids.extend(
                f"{source}/{channel_id}"
                for channel_id in dataset.get_channel_ids()
            )
        return sorted(set(channel_ids))

    def get_recording_ids(self) -> list[str]:
        return sorted(self.recording_ids)


class NeurosoftDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "on_vs_off": ["neurosoft_on_vs_off"],
        "acoustic_stim": ["neurosoft_acoustic_stim"],
        LOGFREQ_TASK: ["neurosoft_acoustic_stim_logfreq"],
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
            Literal["on_vs_off", "acoustic_stim", "acoustic_stim_logfreq"]
        ] = "on_vs_off",
        fold_number: Optional[int] = 0,
        recording_ids: Optional[list[str]] = None,
    ):
        dataset_task_type = _neurosoft_dataset_task_type(task_type)
        transforms = _prepend_logfreq_transform(transforms, task_type)
        dataset_kwargs = {
            "recording_ids": recording_ids,
            "split_type": split_type,
            "task_type": dataset_task_type,
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

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())

    def _get_sampling_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        sampling_intervals = super()._get_sampling_intervals(split)
        if self.task_type == LOGFREQ_TASK:
            return filter_acoustic_stim_tone_intervals(
                sampling_intervals, dataset=self.dataset
            )
        return sampling_intervals


class NeurosoftMinipigs2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMinipigs2026, **kwargs)


class NeurosoftMonkeys2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMonkeys2026, **kwargs)


class NeurosoftMinipigsMonkeys2026DataModule(NeurosoftDataModule):
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
            Literal["on_vs_off", "acoustic_stim", "acoustic_stim_logfreq"]
        ] = "on_vs_off",
        fold_number: Optional[int] = 0,
        minipigs_recording_ids: Optional[list[str]] = None,
        monkeys_recording_ids: Optional[list[str]] = None,
    ):
        dataset_task_type = _neurosoft_dataset_task_type(task_type)
        transforms = _prepend_logfreq_transform(transforms, task_type)
        dataset_kwargs = {
            "minipigs_recording_ids": minipigs_recording_ids,
            "monkeys_recording_ids": monkeys_recording_ids,
            "split_type": split_type,
            "task_type": dataset_task_type,
            "fold_num": fold_number,
        }
        NeuralDataModule.__init__(
            self,
            dataset_class=NeurosoftMinipigsMonkeys2026,
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
