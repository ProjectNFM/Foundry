"""Tests for model-declared pretrained transfer regimes in the CLI path."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from main import _resolve_pretrained_components


class _RegimeModel:
    def transferable_components_for_mode(self, mode: str) -> tuple[str, ...]:
        if mode == "frozen_representation":
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
