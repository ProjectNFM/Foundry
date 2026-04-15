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
from typing import Optional, Callable, Literal, Type


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
