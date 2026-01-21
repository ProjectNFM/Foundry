from typing import Callable
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.transforms import Compose


class PoYoTokenizerDataModule(LightningDataModule):
    """DataModule for POYO Tokenizer.

    This class is used to load the data for the POYO Tokenizer.
    It assumes that the brainsets are stored in the data_root_dir.
    """

    def __init__(
        self,
        data_root_dir: str,
        batch_size: int,
        config: DictConfig,
        num_workers: int,
        context_length_s: float,
        seed: int,
        pretokenizer_transform: Callable = None,
    ):
        """Initialize the DataModule.

        Args:
            data_root_dir: Root directory of the data.
            batch_size: Batch size.
            config: Configuration for the dataset.
            num_workers: Number of workers for the dataloaders.
            context_length_s: Context length in seconds.
            seed: Random seed.
            tokenizer: Tokenizer instance to use for data transformation.
            pretokenizer_transform: Optional transform to apply before tokenization.
                If None, uses NormalizePerChannel() by default.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_length_s = context_length_s
        self.seed = seed
        self.config = config
        self.tokenizer = tokenizer
        # Use provided pretokenizer_transform or default to NormalizePerChannel
        self.pretokenizer_transform = (
            pretokenizer_transform
            if pretokenizer_transform is not None
            else NormalizePerChannel()
        )

    def setup(self, stage: str):
        """Setup the DataModule.

        Args:
            stage: Stage to setup the DataModule for. Can be 'fit', 'test'.
        """
        if stage == "fit":
            # Compose pretokenizer transform with tokenization
            transform = Compose([self.pretokenizer_transform, self.tokenizer.tokenize])

            self.train_dataset = Dataset(
                root=self.data_root_dir,
                config=self.config,
                split="train",
                transform=transform,
            )
            self.val_dataset = Dataset(
                root=self.data_root_dir,
                config=self.config,
                split="valid",
                transform=transform,
            )
        else:
            raise NotImplementedError(
                f"Invalid stage: {stage}, only 'fit' is supported."
            )

    def train_dataloader(self):
        """Return the training dataloader.

        Returns:
            DataLoader: Training dataloader, a torch.utils.data.DataLoader object with a RandomFixedWindowSampler.
        """
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.context_length_s,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Return the validation dataloader.

        Returns:
            DataLoader: Validation dataloader, a torch.utils.data.DataLoader object with a RandomFixedWindowSampler.
        """
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.context_length_s,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        """Return the test dataloader.

        This method is not implemented yet.
        """
        raise NotImplementedError("Test dataloader not implemented.")
