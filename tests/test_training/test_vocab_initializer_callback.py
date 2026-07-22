"""Tests for VocabInitializerCallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning as L
import pytest
import torch.nn as nn
from lightning import Trainer

from foundry.training.callbacks import VocabInitializerCallback


class _LazyVocabModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._channel_lazy = True
        self._session_lazy = True
        self.channel_ids: list[str] | None = None
        self.session_ids: list[str] | None = None

    def initialize_vocabs(self, vocab_info: dict) -> None:
        if "channel_ids" in vocab_info:
            self.channel_ids = vocab_info["channel_ids"]
            self._channel_lazy = False
        if "session_ids" in vocab_info:
            self.session_ids = vocab_info["session_ids"]
            self._session_lazy = False

    def has_lazy_vocabs(self) -> bool:
        return self._channel_lazy or self._session_lazy


class _VocabModule(L.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class _DatamoduleWithVocabMethods:
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset

    def get_recording_ids(self) -> list[str]:
        return ["sess_a", "sess_b"]

    def get_channel_ids(self) -> list[str]:
        return ["ch_1", "ch_2"]


def _run_callback(
    pl_module: L.LightningModule, datamodule: object | None
) -> None:
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.datamodule = datamodule
    VocabInitializerCallback().on_fit_start(trainer, pl_module)


class TestVocabInitializerCallback:
    def test_initializes_vocabs_from_datamodule(self):
        model = _LazyVocabModel()
        module = _VocabModule(model)
        dm = _DatamoduleWithVocabMethods()

        _run_callback(module, dm)

        assert model.session_ids == ["sess_a", "sess_b"]
        assert model.channel_ids == ["ch_1", "ch_2"]
        assert not model.has_lazy_vocabs()

    def test_skips_non_vocab_manager(self):
        model = nn.Linear(2, 2)
        module = _VocabModule(model)
        dm = _DatamoduleWithVocabMethods()

        _run_callback(module, dm)

    def test_skips_when_vocabs_already_initialized(self):
        model = _LazyVocabModel()
        model._channel_lazy = False
        model._session_lazy = False
        module = _VocabModule(model)
        dm = _DatamoduleWithVocabMethods()

        _run_callback(module, dm)

        assert model.channel_ids is None
        assert model.session_ids is None

    def test_requires_datamodule(self):
        model = _LazyVocabModel()
        module = _VocabModule(model)

        with pytest.raises(RuntimeError, match="requires a datamodule"):
            _run_callback(module, None)

    def test_does_not_swallow_vocab_method_errors(self):
        model = _LazyVocabModel()
        module = _VocabModule(model)
        dm = MagicMock()
        dm.dataset = None
        dm.get_recording_ids.return_value = ["sess_a"]
        dm.get_channel_ids.side_effect = RuntimeError("channel lookup failed")

        with pytest.raises(RuntimeError, match="channel lookup failed"):
            _run_callback(module, dm)
