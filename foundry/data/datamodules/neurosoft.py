from typing import Optional, Callable, Literal, Type

from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.datasets.neurosoft import (
    NeurosoftMinipigs2026,
    NeurosoftMonkeys2026,
)
from foundry.tasks.class_weights import compute_class_weights_for_tasks
from foundry.tasks.config import TaskConfig


class NeurosoftDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "on_vs_off": ["neurosoft_on_vs_off"],
        "acoustic_stim": ["neurosoft_acoustic_stim"],
    }

    @classmethod
    def get_tasks_for_experiment(cls, task_type: str) -> dict[str, TaskConfig]:
        task_names = cls.TASK_TO_READOUT[task_type]
        return NeurosoftMinipigs2026.get_tasks(task_names)

    def compute_class_weights(
        self, smoothing: float = 1.0
    ) -> dict[str, list[float]]:
        if self.dataset is None:
            raise RuntimeError("Call setup() before compute_class_weights()")
        if self.task_type is None:
            raise ValueError(
                "task_type must be set to compute class weights automatically"
            )

        task_configs = self.get_tasks_for_experiment(self.task_type)
        return compute_class_weights_for_tasks(
            task_configs, self.dataset, split="train", smoothing=smoothing
        )

    def __init__(
        self,
        dataset_class: Type[NeurosoftMinipigs2026] | Type[NeurosoftMonkeys2026],
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

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())


class NeurosoftMinipigs2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMinipigs2026, **kwargs)


class NeurosoftMonkeys2026DataModule(NeurosoftDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=NeurosoftMonkeys2026, **kwargs)
