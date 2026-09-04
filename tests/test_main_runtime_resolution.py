"""Regression tests for metadata discovery and hardware precision resolution."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

import main


def _discovery_cfg(role: str | None):
    return OmegaConf.create(
        {
            "data": {
                "_target_": "foundry.data.datamodules.NeuralDataModule",
                "role": role,
            },
            "hyperparameters": {
                "session_configs": None,
                "num_channels": None,
            },
            "task_configs": [],
        }
    )


def test_source_metadata_discovery_explicitly_skips_normalization():
    cfg = _discovery_cfg("source_pretraining")
    datamodule = Mock()
    datamodule.get_session_configs.return_value = {"source/session": 32}
    datamodule.get_num_channels.return_value = 32

    with (
        patch.object(main, "instantiate", return_value=datamodule),
        patch.object(main, "normalize_data_config"),
        patch.object(main, "_load_task_configs", return_value={}),
    ):
        main._populate_data_driven_hyperparams(cfg)

    datamodule.setup.assert_called_once_with("fit", fit_normalization=False)
    assert cfg.hyperparameters.num_channels == 32


def test_non_source_metadata_discovery_retains_normal_setup_behavior():
    cfg = _discovery_cfg(None)
    datamodule = Mock()
    datamodule.get_session_configs.return_value = {"session": 8}
    datamodule.get_num_channels.return_value = 8

    with (
        patch.object(main, "instantiate", return_value=datamodule),
        patch.object(main, "normalize_data_config"),
        patch.object(main, "_load_task_configs", return_value={}),
    ):
        main._populate_data_driven_hyperparams(cfg)

    datamodule.setup.assert_called_once_with("fit")


def _precision_cfg(precision="bf16-mixed", fallback=None):
    cfg = OmegaConf.create({"trainer": {"precision": precision}, "run": {}})
    if fallback is not None:
        cfg.run.unsupported_bf16_fallback = fallback
    return cfg


def test_rtx_8000_uses_explicit_fp16_fallback():
    cfg = _precision_cfg(fallback="16-mixed")
    with (
        patch.object(main.torch.cuda, "is_available", return_value=True),
        patch.object(
            main.torch.cuda,
            "get_device_name",
            return_value="Quadro RTX 8000",
        ),
        patch.object(
            main.torch.cuda, "get_device_capability", return_value=(7, 5)
        ),
        patch.object(main.torch.cuda, "is_bf16_supported", return_value=False),
    ):
        main._resolve_precision_for_hardware(cfg)

    assert cfg.trainer.precision == "16-mixed"
    assert cfg.run.requested_precision == "bf16-mixed"
    assert cfg.run.effective_precision == "16-mixed"
    assert cfg.run.gpu_compute_capability == "7.5"


def test_a100_keeps_bf16():
    cfg = _precision_cfg(fallback="16-mixed")
    with (
        patch.object(main.torch.cuda, "is_available", return_value=True),
        patch.object(
            main.torch.cuda, "get_device_name", return_value="NVIDIA A100"
        ),
        patch.object(
            main.torch.cuda, "get_device_capability", return_value=(8, 0)
        ),
        patch.object(main.torch.cuda, "is_bf16_supported", return_value=True),
    ):
        main._resolve_precision_for_hardware(cfg)

    assert cfg.trainer.precision == "bf16-mixed"
    assert cfg.run.effective_precision == "bf16-mixed"


def test_unsupported_bf16_without_opt_in_fails_actionably():
    cfg = _precision_cfg()
    with (
        patch.object(main.torch.cuda, "is_available", return_value=True),
        patch.object(
            main.torch.cuda,
            "get_device_name",
            return_value="Quadro RTX 8000",
        ),
        patch.object(
            main.torch.cuda, "get_device_capability", return_value=(7, 5)
        ),
        patch.object(main.torch.cuda, "is_bf16_supported", return_value=False),
        pytest.raises(RuntimeError, match="unsupported_bf16_fallback"),
    ):
        main._resolve_precision_for_hardware(cfg)


def test_cpu_records_requested_precision_without_gpu_rewrite():
    cfg = _precision_cfg(precision="32-true")
    with patch.object(main.torch.cuda, "is_available", return_value=False):
        main._resolve_precision_for_hardware(cfg)

    assert cfg.trainer.precision == "32-true"
    assert cfg.run.effective_precision == "32-true"
    assert cfg.run.gpu_name == "cpu"
    assert cfg.run.gpu_compute_capability is None
