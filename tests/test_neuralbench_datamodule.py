"""Offline integration tests for ``NeuralBenchDataModule``."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from foundry.data.neuralbench.datamodule import NeuralBenchDataModule


class _Dataset:
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples
        self.triggers = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class _Loader:
    def __init__(self, samples: list[dict]) -> None:
        self.dataset = _Dataset(samples)


class _NBData:
    loaders: dict[str, _Loader]

    def __init__(self, **_kwargs) -> None:
        self.neuro = type(
            "Neuro", (), {"_channels": {"Cz": None, "Pz": None}, "frequency": 4}
        )()

    def prepare(self) -> dict[str, _Loader]:
        return self.loaders


def _sample(subject: int, target: tuple[int, int] = (1, 0)) -> dict:
    return {
        "neuro": np.arange(8, dtype=np.float32).reshape(1, 2, 4),
        "target": np.array([target]),
        "subject_id": np.array([[subject]]),
    }


def _install_mock_neuralbench(monkeypatch) -> None:
    """Install only the imports used by setup; no NeuralBench runtime needed."""
    exca = ModuleType("exca")
    exca.ConfDict = lambda value: value
    monkeypatch.setitem(sys.modules, "exca", exca)

    data = ModuleType("neuralbench.data")
    data.Data = _NBData
    monkeypatch.setitem(sys.modules, "neuralbench.data", data)

    experiment_config = ModuleType("neuralbench.experiment_config")
    experiment_config.prepare_task_configs = lambda *_args, **_kwargs: [
        {"data": {"study": {"source": {"path": "original"}}}}
    ]
    monkeypatch.setitem(
        sys.modules, "neuralbench.experiment_config", experiment_config
    )

    registry = ModuleType("neuralbench.registry")
    registry.DEFAULTS_DIR = Path("mock-defaults")
    registry.load_yaml_config = lambda _path: {}
    monkeypatch.setitem(sys.modules, "neuralbench.registry", registry)


def test_setup_uses_mocked_splits_and_exposes_metadata(monkeypatch):
    _NBData.loaders = {
        "train": _Loader([_sample(1), _sample(1, (0, 1))]),
        "val": _Loader([_sample(2)]),
        "test": _Loader([_sample(3)]),
    }
    _install_mock_neuralbench(monkeypatch)
    monkeypatch.setattr(
        "foundry.data.neuralbench.datamodule._require_neuralbench", lambda: None
    )

    dm = NeuralBenchDataModule(
        task="p3", dataset="mock", cache_dir="mock-cache", batch_size=2
    )
    dm.setup("fit")

    assert len(dm._train_adapter) == 2
    assert len(dm._val_adapter) == 1
    assert len(dm._test_adapter) == 1
    assert dm.get_recording_ids() == ["nb/p3/sub-1", "nb/p3/sub-2", "nb/p3/sub-3"]
    assert dm.get_channel_ids() == [
        "nb/p3/sub-1/Cz",
        "nb/p3/sub-1/Pz",
        "nb/p3/sub-2/Cz",
        "nb/p3/sub-2/Pz",
        "nb/p3/sub-3/Cz",
        "nb/p3/sub-3/Pz",
    ]
    assert dm.get_session_configs() == {
        "nb/p3/sub-1": 2,
        "nb/p3/sub-2": 2,
        "nb/p3/sub-3": 2,
    }
    assert dm.get_num_channels() == 2


def test_tokenizer_replacement_is_used_before_collation(monkeypatch):
    _NBData.loaders = {"train": _Loader([_sample(1), _sample(2)])}
    _install_mock_neuralbench(monkeypatch)
    monkeypatch.setattr(
        "foundry.data.neuralbench.datamodule._require_neuralbench", lambda: None
    )

    def tokenizer(data):
        return {"tokenizer_marker": torch.tensor([data.eeg.signal.sum()])}

    dm = NeuralBenchDataModule(task="p3", dataset="mock", batch_size=2)
    dm.setup("fit")
    dm.set_tokenizer(tokenizer)

    sample = dm._train_adapter[0]
    assert sample["tokenizer_marker"].shape == (1,)
    batch = next(iter(dm.train_dataloader()))
    assert "tokenizer_marker" in batch
    assert batch["tokenizer_marker"].shape == (2, 1)
