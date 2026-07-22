"""Shared fixtures for Hydra config instantiation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from foundry.config_resolvers import register_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


def _register_eval_resolver() -> None:
    """Register Hydra-style ``eval`` for concat tokenizer configs in tests."""
    if not OmegaConf.has_resolver("eval"):

        def _eval_resolver(expression: str):
            return eval(expression, {"__builtins__": {}}, {})  # noqa: S307

        OmegaConf.register_new_resolver("eval", _eval_resolver)


@pytest.fixture(scope="session", autouse=True)
def _setup_config_resolvers():
    register_resolvers()
    _register_eval_resolver()


def load_resolved_config(yaml_path: Path) -> DictConfig:
    """Load a YAML file and resolve interpolations without leaking context keys."""
    file_cfg = OmegaConf.load(yaml_path)
    data_defaults = OmegaConf.create(
        {
            "dataset_class": (
                "foundry.data.datasets.PetersonBruntonPoseTrajectory2022"
            ),
            "split_type": "intersession",
            "task_type": "behavior",
            "dataset_kwargs": {
                "fold": 0,
            },
        }
    )
    merged = OmegaConf.merge(instantiation_context(), data_defaults, file_cfg)
    OmegaConf.resolve(merged)
    file_keys = set(file_cfg.keys())
    return OmegaConf.create({key: merged[key] for key in file_keys})


def instantiation_context() -> DictConfig:
    """OmegaConf context for resolving ``${model.*}`` / ``${hyperparameters.*}``."""
    return OmegaConf.create(
        {
            "model": {
                "embed_dim": 256,
                "sequence_length": 2.0,
                "latent_step": 0.1,
                "num_latents_per_step": 16,
                "depth": 2,
                "dim_head": 64,
                "cross_heads": 4,
                "self_heads": 4,
                "ffn_dropout": 0.1,
                "lin_dropout": 0.1,
                "atn_dropout": 0.1,
                "emb_init_scale": 0.02,
                "t_min": 1e-2,
                "t_max": 0.5,
                "zero_output_timestamps": False,
                "tokenizer": {
                    "channel_strategy": {"num_sources": 64},
                    "channel_emb_dim": 64,
                    "embed_dim": 256,
                },
            },
            "hyperparameters": {
                "batch_size": 4,
                "sequence_length": 2.0,
                "patch_duration": 0.1,
                "sampling_rate": 250.0,
                "fold_number": 0,
                "num_workers": 0,
                "learning_rate": 1e-3,
                "scheduler": "warmup_hold_cosine",
                "warmup_steps": 0,
                "hold_steps": 0,
                "cosine_steps": 0,
                "decay_steps": 0,
                "min_lr_factor": 0.1,
                "weight_decay": 0.01,
                "cwt_lr_multiplier": 1.0,
                "num_channels": 32,
                "session_configs": {"sessionA": 16, "sessionB": 20},
            },
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": "/tmp/foundry_config_tests",
                "name": "test",
            },
            "data": {
                "task_type": "behavior",
                "split_type": "intersession",
            },
            "run": {
                "name": "config_test",
                "project": "config_test",
                "group": "config_test",
                "tags": ["test"],
                "seed": 42,
            },
        }
    )
