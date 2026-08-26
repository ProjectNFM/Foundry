"""Offline integration tests for ``NeuralBenchDataModule``."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from foundry.data.neuralbench.adapter import NeuralSetAdapter
from foundry.data.neuralbench.datamodule import (
    NeuralBenchDataModule,
    _build_label_map_from_encoder,
    _stratified_subset_indices,
)
from foundry.tasks.config import TaskConfig


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
        self.target = type(
            "Target", (), {"_label_to_ind": {"NonTarget": 0, "Target": 1}}
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


def test_label_map_preserves_numeric_neuralbench_labels():
    target = type("Target", (), {"_label_to_ind": {1: 0, 2: 1}})()

    assert _build_label_map_from_encoder(target) == {0: 1, 1: 2}


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
    assert dm.get_recording_ids() == [
        "nb/p3/sub-1",
        "nb/p3/sub-2",
        "nb/p3/sub-3",
    ]
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
    assert dm.train_dataloader().drop_last is False


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


def test_class_weights_follow_foundry_output_class_order():
    # NeuralBench LabelEncoder sorts source labels as N1, N2, N3, R, W.
    # Foundry's task order is Wake, N1, N2, N3, REM, so source indices must
    # be remapped before the loss-weight vector is assembled.
    label_map = {0: "N1", 1: "N2", 2: "N3", 3: "R", 4: "W"}
    samples = []
    for source_idx, count in {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}.items():
        target = [0] * 5
        target[source_idx] = 1
        samples.extend(
            {
                "neuro": np.zeros((1, 2, 4), dtype=np.float32),
                "target": np.array([target]),
                "subject_id": np.array([[1]]),
            }
            for _ in range(count)
        )
    dm = NeuralBenchDataModule(task="sleep_stage", dataset="mock")
    dm.label_map = label_map
    dm._train_adapter = NeuralSetAdapter(
        _Dataset(samples),
        channel_names=["Cz", "Pz"],
        sampling_rate=1.0,
        split="train",
        label_map=label_map,
    )
    task = TaskConfig.from_yaml("configs/tasks/neuralbench/sleep_stage.yaml")
    dm._task_configs = {task.name: task}

    weights = dm.compute_class_weights()[task.name]

    # Output order: Wake (1), N1 (2), N2 (1), N3 (1), REM (1).
    assert weights == [1.2, 0.6, 1.2, 1.2, 1.2]


def test_stratified_subset_is_exact_reproducible_and_retains_classes():
    samples = [
        _sample(index, (1, 0) if index < 8 else (0, 1))
        for index in range(10)
    ]
    dataset = _Dataset(samples)

    first = _stratified_subset_indices(dataset, 6, seed=33)
    repeated = _stratified_subset_indices(dataset, 6, seed=33)
    different = _stratified_subset_indices(dataset, 6, seed=34)

    assert first == repeated
    assert first != different
    assert len(first) == 6
    labels = {
        int(np.argmax(dataset[index]["target"].reshape(-1))) for index in first
    }
    assert labels == {0, 1}


def test_setup_applies_subsets_and_class_weights_use_training_subset(monkeypatch):
    _NBData.loaders = {
        "train": _Loader(
            [_sample(index, (1, 0) if index < 8 else (0, 1)) for index in range(10)]
        ),
        "val": _Loader(
            [_sample(index, (1, 0) if index < 4 else (0, 1)) for index in range(6)]
        ),
        "test": _Loader([_sample(index) for index in range(3)]),
    }
    _install_mock_neuralbench(monkeypatch)
    monkeypatch.setattr(
        "foundry.data.neuralbench.datamodule._require_neuralbench", lambda: None
    )

    dm = NeuralBenchDataModule(
        task="p3",
        dataset="mock",
        train_subset_size=6,
        val_subset_size=4,
        subset_seed=33,
    )
    dm.setup("fit")
    task = TaskConfig.from_yaml("configs/tasks/neuralbench/p300.yaml")
    dm.label_map = {0: 1, 1: 2}
    dm._task_configs = {task.name: task}

    assert len(dm._train_adapter) == 6
    assert len(dm._val_adapter) == 4
    assert len(dm._test_adapter) == 3
    weights = dm.compute_class_weights()[task.name]
    assert len(weights) == 2
    assert weights[0] < weights[1]
