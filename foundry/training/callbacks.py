"""Lightning callbacks for Foundry model training."""

import lightning as L
from lightning import Trainer

from foundry.core import VocabManager


class VocabInitializerCallback(L.Callback):
    """Callback to initialize model vocabularies from the datamodule.

    This callback handles the initialization of lazy vocabularies (e.g., session and
    channel embeddings) before training begins. It decouples vocab setup from the
    datamodule, allowing models to be reused with different datasets.

    Usage:
        trainer = Trainer(callbacks=[VocabInitializerCallback()])
    """

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
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

        # Initialize vocabularies from datamodule or its underlying dataset
        vocab_info = {}
        dataset = getattr(datamodule, "dataset", None)

        for method_name, key in [
            ("get_recording_ids", "session_ids"),
            ("get_channel_ids", "channel_ids"),
        ]:
            if hasattr(datamodule, method_name):
                vocab_info[key] = getattr(datamodule, method_name)()
            elif dataset is not None and hasattr(dataset, method_name):
                vocab_info[key] = getattr(dataset, method_name)()

        model.initialize_vocabs(vocab_info)
