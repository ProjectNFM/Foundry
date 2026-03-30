from brainsets.datasets import PetersonBruntonPoseTrajectory2022
from foundry.data.datamodules.base import NeuralDataModule
from typing import Optional, Callable, Literal


class AjileDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "active_vs_inactive": ["ajile_inactive_active"],
        "behavior": ["ajile_active_behavior"],
    }

    READOUT_CLASS_NAMES: dict[str, list[str]] = {
        "ajile_inactive_active": ["Active", "Inactive"],
        "ajile_active_behavior": ["Eat", "Talk", "TV", "Computer/Phone"],
    }

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        window_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        split_type: Optional[
            Literal["intersubject", "intersession", "intrasession"]
        ] = "intersubject",
        task_type: Optional[
            Literal["active_vs_inactive", "behavior", "pose_estimation"]
        ] = "behavior",
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
            dataset_class=PetersonBruntonPoseTrajectory2022,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            window_length=window_length,
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
