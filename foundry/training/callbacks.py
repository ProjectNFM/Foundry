"""Lightning callbacks for Foundry model training."""

from typing import Optional
import lightning as L
from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from foundry.core import VocabManager


class VocabInitializerCallback(L.Callback):
    """Callback to initialize model vocabularies from the datamodule.

    This callback handles the initialization of lazy vocabularies (e.g., session and
    channel embeddings) before training begins. It decouples vocab setup from the
    datamodule, allowing models to be reused with different datasets.

    Usage:
        trainer = Trainer(callbacks=[VocabInitializerCallback()])
    """

    def on_fit_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Initialize vocabularies at the start of training.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module being trained.
        """
        # Check if model implements vocab initialization
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        if not isinstance(model, VocabManager):
            return

        if not model.has_lazy_vocabs():
            return

        # Get datamodule
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError(
                "VocabInitializerCallback requires a datamodule. "
                "Call trainer.fit(module, datamodule=dm) or set trainer.datamodule."
            )

        # Get dataset from datamodule
        if not hasattr(datamodule, "dataset") or datamodule.dataset is None:
            raise RuntimeError(
                "VocabInitializerCallback requires datamodule.dataset to be set during setup(). "
                "Ensure setup() is called before training."
            )

        dataset = datamodule.dataset

        # Initialize vocabularies
        vocab_info = {}  
        if hasattr(dataset, "get_recording_ids"):
            vocab_info["session_ids"] = dataset.get_recording_ids()

        if hasattr(dataset, "get_channel_ids"):
            vocab_info["channel_ids"] = dataset.get_channel_ids()

        model.initialize_vocabs(vocab_info)
