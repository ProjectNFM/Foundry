from typing import Callable, Literal, Optional, Type
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


class AddNeurosoftSourceId:
    """Attach the Neurosoft source label to each sample before tokenization."""

    def __init__(self, source_id: str):
        self.source_id = source_id

    def __call__(self, data):
        data.source_id = self.source_id
        return data


def _prepend_neurosoft_transforms(
    transforms: Optional[list[Callable]],
    source_id: Optional[str] = None,
) -> Optional[list[Callable]]:
    transform_list = list(transforms or [])
    if source_id is not None:
        transform_list.insert(0, AddNeurosoftSourceId(source_id))
    return transform_list or None


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
                task_type=task_type,
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
        sample = self.datasets[source][
            DatasetIndex(
                inner_recording_id,
                index.start,
                index.end,
                _namespace=source,
            )
        ]
        if isinstance(sample, dict):
            sample["source_id"] = source
        return sample

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
        task_type: Optional[Literal["on_vs_off", "acoustic_stim"]] = "on_vs_off",
        fold_number: Optional[int] = 0,
        recording_ids: Optional[list[str]] = None,
        source_id: Optional[str] = None,
    ):
        transforms = _prepend_neurosoft_transforms(transforms, source_id)
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

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())


class NeurosoftMinipigs2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(
            dataset_class=NeurosoftMinipigs2026,
            source_id="minipigs",
            **kwargs,
        )


class NeurosoftMonkeys2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(
            dataset_class=NeurosoftMonkeys2026,
            source_id="monkeys",
            **kwargs,
        )


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
        task_type: Optional[Literal["on_vs_off", "acoustic_stim"]] = "on_vs_off",
        fold_number: Optional[int] = 0,
        minipigs_recording_ids: Optional[list[str]] = None,
        monkeys_recording_ids: Optional[list[str]] = None,
    ):
        transforms = _prepend_neurosoft_transforms(transforms)
        dataset_kwargs = {
            "minipigs_recording_ids": minipigs_recording_ids,
            "monkeys_recording_ids": monkeys_recording_ids,
            "split_type": split_type,
            "task_type": task_type,
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
