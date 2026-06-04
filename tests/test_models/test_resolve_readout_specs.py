"""Tests for resolve_readout_specs (OmegaConf DictConfig handling)."""

from omegaconf import OmegaConf

import foundry.data.datasets.modalities  # noqa: F401 — register modalities
from foundry.data.readout_specs import clone_readout_spec
from foundry.data.datasets.modalities import MappedCrossEntropyLoss
from foundry.models.utils import resolve_readout_specs
from torch_brain.registry import MODALITY_REGISTRY


def test_resolve_readout_specs_preserves_effective_dictconfig():
    """DictConfig from Hydra merge must not be iterated as modality name strings."""
    base = MODALITY_REGISTRY["neurosoft_acoustic_stim"]
    mapping = {4: 0, 6: 0, 7: 1, 11: 1, 14: 2, 16: 2}
    effective = clone_readout_spec(
        base,
        dim=3,
        loss_fn=MappedCrossEntropyLoss(mapping),
    )
    effective_specs = {"neurosoft_acoustic_stim": effective}
    # Same path as hydra.utils.instantiate: merge model cfg with readout_specs kwarg
    model_cfg = OmegaConf.create(
        {"_target_": "foundry.models.EEGNetEncoder", "num_channels": 64},
        flags={"allow_objects": True},
    )
    merged = OmegaConf.merge(model_cfg, {"readout_specs": effective_specs})
    as_dictconfig = merged.readout_specs

    resolved = resolve_readout_specs(as_dictconfig)

    assert resolved["neurosoft_acoustic_stim"].dim == 3
    assert isinstance(resolved["neurosoft_acoustic_stim"].loss_fn, MappedCrossEntropyLoss)
    assert MODALITY_REGISTRY["neurosoft_acoustic_stim"].dim == 26
