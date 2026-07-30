"""Lifecycle callbacks for vocabulary initialization and sampler seeding."""

from __future__ import annotations

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
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        if not isinstance(model, VocabManager):
            return

        if not model.has_lazy_vocabs():
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError(
                "VocabInitializerCallback requires a datamodule. "
                "Call trainer.fit(module, datamodule=dm) or set trainer.datamodule."
            )

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


class DeterministicSamplerCallback(L.Callback):
    """Re-seed the train sampler at the start of every epoch.

    When debugging with ``limit_train_batches=1``, the
    :class:`RandomFixedWindowSampler` normally produces different windows
    each epoch because its internal ``torch.Generator`` advances.  This
    callback resets the generator so every epoch yields identical batches.
    """

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        loader = trainer.train_dataloader
        if loader is None:
            return
        sampler = getattr(loader, "sampler", None)
        gen = getattr(sampler, "generator", None)
        if gen is not None:
            seed = getattr(trainer.datamodule, "seed", 42)
            gen.manual_seed(seed)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        loaders = trainer.val_dataloaders
        if loaders is None:
            return
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for loader in loaders:
            sampler = getattr(loader, "sampler", None)
            gen = getattr(sampler, "generator", None)
            if gen is not None:
                seed = getattr(trainer.datamodule, "seed", 42) + 1
                gen.manual_seed(seed)
