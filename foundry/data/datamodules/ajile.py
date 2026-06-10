from typing import Callable, Optional

from foundry.data.datasets.peterson_brunton_pose_trajectory_2022 import (
    PetersonBruntonSplitType,
    PetersonBruntonTaskType,
)
from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.datasets.peterson_brunton_pose_trajectory_2022 import (
    PetersonBruntonPoseTrajectory2022,
)
from foundry.data.transforms import PreparePoseTrajectories
from foundry.tasks.class_weights import compute_class_weights_for_tasks
from foundry.tasks.config import TaskConfig


class AjileDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "active_vs_inactive": ["ajile_inactive_active"],
        "behavior": ["ajile_active_behavior"],
        "pose_estimation": ["ajile_pose_estimation"],
    }

    @classmethod
    def get_tasks_for_experiment(cls, task_type: str) -> dict[str, TaskConfig]:
        task_names = cls.TASK_TO_READOUT[task_type]
        return PetersonBruntonPoseTrajectory2022.get_tasks(task_names)

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
