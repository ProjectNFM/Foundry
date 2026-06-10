from typing import Callable, Optional, Literal

from foundry.data.datasets.schalk_wolpaw_physionet_2009 import (
    SchalkWolpawPhysionet2009,
)
from foundry.data.datamodules.base import NeuralDataModule
from foundry.tasks.class_weights import compute_class_weights_for_tasks
from foundry.tasks.config import TaskConfig


class PhysionetDataModule(NeuralDataModule):
    """PyTorch Lightning DataModule for Physionet Motor Imagery Dataset.

    Extends NeuralDataModule with Physionet-specific configuration including
    task type selection and fold configuration. Handles tokenization and vocab
    initialization via callbacks rather than tight coupling to the model.
    """

    TASK_TO_READOUT = {
        "MotorImagery": ["motor_imagery_5class"],
        "LeftRightImagery": ["motor_imagery_left_right"],
        "RightHandFeetImagery": ["motor_imagery_right_feet"],
    }

    @classmethod
    def get_tasks_for_experiment(cls, task_type: str) -> dict[str, TaskConfig]:
        task_names = cls.TASK_TO_READOUT[task_type]
        return SchalkWolpawPhysionet2009.get_tasks(task_names)

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
        # Dataset specific args
        recording_ids: Optional[list[str]] = None,
        uniquify_channel_ids: bool = True,
        task_type: Optional[
            Literal["MotorImagery", "LeftRightImagery", "RightHandFeetImagery"]
        ] = "MotorImagery",
        fold_number: Optional[Literal[0, 1, 2]] = 0,
        fold_type: Literal["intra-subject", "inter-subject"] = "inter-subject",
        dirname: str = "schalk_wolpaw_physionet_2009",
    ):
        """Initialize PhysionetDataModule.

        Args:
            root: Root directory of the data.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for DataLoaders.
            pin_memory: Whether to pin memory in DataLoaders.
            sequence_length: Length of windows for RandomFixedWindowSampler in seconds.
            transforms: Optional list of transforms to apply to each sample before tokenization.
            tokenizer: Optional tokenizer (e.g., model.tokenize) to apply as final transform.
            seed: Random seed for sampling.
            recording_ids: Optional list of recording IDs to include.
            uniquify_channel_ids: If True, prefix channel IDs with session ID.
            task_type: Task configuration for sampling intervals.
            fold_number: Which k-fold split to use.
            fold_type: Type of fold to use.
            dirname: Directory name within root containing the dataset files.
        """
        dataset_kwargs = {
            "recording_ids": recording_ids,
            "uniquify_channel_ids": uniquify_channel_ids,
            "task_type": task_type,
            "fold_number": fold_number,
            "fold_type": fold_type,
            "dirname": dirname,
        }

        super().__init__(
            dataset_class=SchalkWolpawPhysionet2009,
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
