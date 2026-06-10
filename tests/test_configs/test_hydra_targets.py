"""Instantiate every Hydra ``_target_`` node under ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from foundry.data.datamodules.ajile import AjileDataModule
from tests.test_configs.conftest import (
    CONFIGS_ROOT,
    load_resolved_config,
)

# Slurm / local GPU launchers need cluster/runtime wiring, not unit tests.
_SKIP_PREFIXES = (
    "hydra/",
    "experiment/",
)

_SKIP_TARGET_PATHS = {
    ("model/poyo_eeg.yaml", ""),
    ("trainer/default.yaml", ""),
}


def _relative_config_path(path: Path) -> str:
    return str(path.relative_to(CONFIGS_ROOT))


def _config_yaml_files() -> list[Path]:
    paths = sorted(CONFIGS_ROOT.rglob("*.yaml"))
    return [
        p
        for p in paths
        if p.name != "config.yaml"
        and not any(
            _relative_config_path(p).startswith(prefix)
            for prefix in _SKIP_PREFIXES
        )
    ]


def _strip_non_constructor_keys(cfg: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(container, dict):
        container.pop("defaults", None)
    return OmegaConf.create(container)


def _collect_target_cases() -> list[tuple[str, str]]:
    """One root ``_target_`` per YAML file (nested nodes covered by root)."""
    cases: list[tuple[str, str]] = []
    for yaml_path in _config_yaml_files():
        rel = _relative_config_path(yaml_path)
        cfg = OmegaConf.load(yaml_path)
        if "_target_" not in cfg:
            continue
        if (rel, "") in _SKIP_TARGET_PATHS:
            continue
        cases.append((rel, ""))
    return cases


def _node_at_path(cfg: DictConfig, path: str) -> DictConfig:
    if not path:
        return cfg
    import re

    node: Any = cfg
    for part in re.split(r"\.(?![^\[]*\])", path):
        if not part:
            continue
        if "[" in part:
            name, index = part[:-1].split("[", 1)
            node = node[name][int(index)]
        else:
            node = node[part]
    return node


def _instantiate_node(yaml_path: Path, target_path: str) -> Any:
    cfg = load_resolved_config(yaml_path)
    node = _strip_non_constructor_keys(_node_at_path(cfg, target_path))

    kwargs: dict[str, Any] = {}
    target = node.get("_target_", "")

    task_configs = AjileDataModule.get_tasks_for_experiment("behavior")

    if "FoundryModule" in target:
        from foundry.models.readout import ReadoutRouter

        task_cfg = next(iter(task_configs.values()))
        heads = {task_cfg.name: instantiate({**task_cfg.head, "embed_dim": 4})}

        class _StubFoundryModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.task_configs = task_configs
                self.router = ReadoutRouter(heads)

            def forward(self, **kwargs):
                return self.router(
                    kwargs.get("output_embs"), kwargs.get("task_index")
                )

        kwargs["model"] = _StubFoundryModel()
    elif any(
        name in target
        for name in (
            "Linear",
            "MLP",
            "GRU",
            "ShallowConvNet",
            "EEGNetEncoder",
            "TemporalConvAvgPool",
            "POYOEEGModel",
        )
    ):
        if "POYOEEGModel" in target:
            pytest.skip(
                "POYO root requires composed tokenizer; see dedicated test"
            )
        from foundry.tasks.config import TaskConfig

        kwargs["task_configs"] = {
            "ajile_inactive_active": TaskConfig.from_yaml(
                CONFIGS_ROOT / "tasks" / "ajile_inactive_active.yaml"
            )
        }
    elif (
        "AjileDataModule" in target
        or "NeurosoftMinipigs2026DataModule" in target
        or "NeurosoftMonkeys2026DataModule" in target
    ):
        kwargs["tokenizer"] = None

    recursive = "task_configs" not in kwargs
    return instantiate(node, **kwargs, _recursive_=recursive)


_TARGET_CASES = _collect_target_cases()


@pytest.mark.parametrize(
    ("config_relpath", "target_path"),
    _TARGET_CASES,
    ids=[f"{rel}::{path or 'root'}" for rel, path in _TARGET_CASES],
)
def test_config_target_instantiates(config_relpath: str, target_path: str):
    yaml_path = CONFIGS_ROOT / config_relpath
    obj = _instantiate_node(yaml_path, target_path)
    assert obj is not None


@pytest.mark.parametrize(
    "config_name",
    sorted(
        p.stem for p in (CONFIGS_ROOT / "model" / "tokenizer").glob("*.yaml")
    ),
)
def test_tokenizer_config_root_instantiates(config_name: str):
    yaml_path = CONFIGS_ROOT / "model" / "tokenizer" / f"{config_name}.yaml"
    tokenizer = instantiate(load_resolved_config(yaml_path))
    assert tokenizer is not None
    assert hasattr(tokenizer, "temporal_embedding")


@pytest.mark.parametrize(
    "config_name",
    sorted(p.stem for p in (CONFIGS_ROOT / "model").glob("*.yaml")),
)
def test_standalone_model_config_instantiates(config_name: str):
    if config_name == "poyo_eeg":
        pytest.skip("covered by test_poyo_eeg_with_tokenizer_configs")
    yaml_path = CONFIGS_ROOT / "model" / f"{config_name}.yaml"
    from foundry.tasks.config import TaskConfig

    task_configs = {
        "ajile_inactive_active": TaskConfig.from_yaml(
            CONFIGS_ROOT / "tasks" / "ajile_inactive_active.yaml"
        )
    }
    model = instantiate(
        _strip_non_constructor_keys(load_resolved_config(yaml_path)),
        task_configs=task_configs,
        _recursive_=False,
    )
    assert model is not None


@pytest.mark.parametrize(
    "callback_key",
    [
        "vocab_initializer",
        "rich_progress_bar",
        "early_stopping",
        "model_checkpoint",
        "parameter_watcher",
    ],
)
def test_trainer_callback_configs(callback_key: str):
    trainer_cfg = OmegaConf.load(CONFIGS_ROOT / "trainer" / "default.yaml")
    callback = instantiate(trainer_cfg.callbacks[callback_key])
    assert callback is not None


@pytest.mark.parametrize(
    "tokenizer_name",
    sorted(
        p.stem for p in (CONFIGS_ROOT / "model" / "tokenizer").glob("*.yaml")
    ),
)
def test_poyo_eeg_with_tokenizer_configs(tokenizer_name: str):
    tok_path = CONFIGS_ROOT / "model" / "tokenizer" / f"{tokenizer_name}.yaml"
    tokenizer = instantiate(load_resolved_config(tok_path))

    poyo_cfg = _strip_non_constructor_keys(
        load_resolved_config(CONFIGS_ROOT / "model" / "poyo_eeg.yaml")
    )
    from foundry.models.poyo_eeg import POYOEEGModel
    from foundry.tasks.config import TaskConfig

    task_configs = {
        "ajile_active_behavior": TaskConfig.from_yaml(
            CONFIGS_ROOT / "tasks" / "ajile_active_behavior.yaml"
        )
    }
    poyo_kwargs = OmegaConf.to_container(poyo_cfg, resolve=True)
    poyo_kwargs.pop("_target_", None)
    model = POYOEEGModel(
        tokenizer=tokenizer,
        task_configs=task_configs,
        **poyo_kwargs,
    )
    assert model is not None
