from brainsets.datasets import (
    PetersonBruntonPoseTrajectory2022,
    PetersonBruntonSplitType,
    PetersonBruntonTaskType,
)
from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.transforms import PreparePoseTrajectories
from typing import Optional, Callable


class AjileDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "active_vs_inactive": ["ajile_inactive_active"],
        "behavior": ["ajile_active_behavior"],
        "pose_estimation": ["ajile_pose_estimation"],
    }

    READOUT_CLASS_NAMES: dict[str, list[str]] = {
        "ajile_inactive_active": ["Active", "Inactive"],
        "ajile_active_behavior": [
            "Eat",
            "Talk",
            "TV",
            "Computer/Phone",
            "Other Activity",
        ],
    }

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
        split_type: Optional[PetersonBruntonSplitType] = "intersession",
        task_type: Optional[PetersonBruntonTaskType] = "behavior",
        fold_number: Optional[int] = 0,
        recording_ids: Optional[list[str]] = None,
    ):
        transform_list = list(transforms) if transforms is not None else []
        if task_type == "pose_estimation":
            transform_list.insert(0, PreparePoseTrajectories())

        dataset_kwargs = {
            "recording_ids": recording_ids,
            "split_type": split_type,
            "task_type": task_type,
            "fold_number": fold_number,
        }
        super().__init__(
            dataset_class=PetersonBruntonPoseTrajectory2022,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sequence_length=sequence_length,
            transforms=transform_list or None,
            tokenizer=tokenizer,
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            task_type=task_type,
        )

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())
