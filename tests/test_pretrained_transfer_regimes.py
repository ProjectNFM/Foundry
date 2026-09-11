"""Tests for model-declared pretrained transfer regimes in the CLI path."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from main import (
    _resolve_pretrained_components,
    _validate_pretrained_transfer_configuration,
)


class _RegimeModel:
    def transferable_components_for_mode(self, mode: str) -> tuple[str, ...]:
        if mode == "frozen_representation":
            return ("temporal_frontend", "gru")
        if mode == "full_finetuning_reset_router":
            return ("temporal_frontend", "gru")
        if mode == "frozen_random_control":
            return ("temporal_frontend", "gru")
        raise ValueError(mode)


def test_named_transfer_regime_selects_model_declared_components():
    cfg = OmegaConf.create(
        {"run": {"pretrained_transfer_regime": "frozen_representation"}}
    )

    assert _resolve_pretrained_components(_RegimeModel(), cfg) == (
        "temporal_frontend",
        "gru",
    )


def test_named_transfer_regime_requires_model_support():
    cfg = OmegaConf.create(
        {"run": {"pretrained_transfer_regime": "frozen_representation"}}
    )

    with pytest.raises(ValueError, match="does not support named"):
        _resolve_pretrained_components(object(), cfg)


@pytest.mark.parametrize(
    "regime", ["full_finetuning_reset_router", "frozen_representation"]
)
def test_new_manifest_backed_regimes_validate(regime):
    cfg = OmegaConf.create(
        {
            "run": {
                "pretrained_transfer_regime": regime,
                "pretrained_checkpoint_manifest": "/manifest.json",
                "pretrained_checkpoint": None,
            }
        }
    )
    _validate_pretrained_transfer_configuration(cfg)


def test_reset_router_requires_manifest():
    cfg = OmegaConf.create(
        {"run": {"pretrained_transfer_regime": "full_finetuning_reset_router"}}
    )
    with pytest.raises(
        ValueError, match="requires a pretrained checkpoint manifest"
    ):
        _validate_pretrained_transfer_configuration(cfg)


def test_random_control_rejects_manifest_and_source_seed():
    cfg = OmegaConf.create(
        {
            "run": {
                "pretrained_transfer_regime": "frozen_random_control",
                "pretrained_checkpoint_manifest": "/manifest.json",
                "source_model_seed": 42,
            }
        }
    )
    with pytest.raises(ValueError, match="random frozen-backbone control"):
        _validate_pretrained_transfer_configuration(cfg)
