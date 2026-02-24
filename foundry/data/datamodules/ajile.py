from brainsets.datasets import PetersonBruntonPoseTrajectory2022
from foundry.data.datamodules.base import NeuralDataModule
from typing import Optional, Callable, Literal


class AjileDataModule(NeuralDataModule):
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
    ):
        dataset_kwargs = {
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
        )
