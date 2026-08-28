"""Focused contract tests for the NeuroSoft convolutional BiGRU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from torch_brain.batching import collate
from torch_brain.data import Data, Interval, RegularTimeSeries

from foundry.models import NeurosoftConvBiGRU
from foundry.tasks.config import TaskConfig
from foundry.training import PretrainedTransferError, load_pretrained_weights


class _Channels:
    def __init__(self, count: int):
        self.id = np.array([f"ch-{i}" for i in range(count)])
        self.type = np.array(["ECOG"] * count)


class _Session:
    def __init__(self, session_id: str):
        self.id = session_id


def _task_configs():
    return {
        "neurosoft": TaskConfig.from_dict(
            {
                "name": "neurosoft",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 8,
                },
                "target_extractor": {
                    "_target_": "foundry.tasks.targets.TargetExtractor",
                    "timestamp_key": "neurosoft.timestamps",
                    "value_key": "neurosoft.values",
                },
                "loss": {
                    "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"
                },
                "metrics": {
                    "_target_": "foundry.tasks.metrics.classification_metrics",
                    "num_classes": 8,
                },
                "class_names": [str(i) for i in range(8)],
            }
        )
    }


def _data(session_id: str, channels: int, samples: int = 1000) -> Data:
    data = Data(
        ecog=RegularTimeSeries(
            signal=np.random.randn(samples, channels).astype(np.float32),
            sampling_rate=2000.0,
            domain_start=0.0,
        ),
        domain=Interval(0.0, samples / 2000.0),
    )
    data.channels = _Channels(channels)
    data.session = _Session(session_id)
    data._absolute_start = 0.0

    class _Task:
        timestamps = np.array([0.1])
        values = np.array([2])

    data.neurosoft = _Task()
    return data


def _model(session_configs=None, **kwargs):
    session_configs = session_configs or {"minipig": 3, "monkey": 5}
    defaults = {
        "adapter_dim": 8,
        "temporal_channels": 12,
        "gru_hidden_size": 8,
        "dropout_rate": 0.0,
    }
    defaults.update(kwargs)
    return NeurosoftConvBiGRU(
        task_configs=_task_configs(),
        session_configs=session_configs,
        **defaults,
    )


def _save_checkpoint(model, path):
    torch.save(
        {
            "state_dict": {
                f"model.{k}": v.detach().clone()
                for k, v in model.state_dict().items()
            }
        },
        path,
    )


def test_tokenize_collate_forward_and_backward_across_channel_counts():
    model = _model()
    batch = collate(
        [
            model.tokenize(_data("minipig", 3)),
            model.tokenize(_data("monkey", 5)),
        ]
    )
    assert batch["input_values"].shape == (2, 5, 1000)
    assert batch["input_seq_len"].tolist() == [1000, 1000]
    outputs = model(
        input_values=batch["input_values"],
        task_index=batch["task_index"],
        input_session_ids=batch["input_session_ids"],
        input_channel_counts=batch["input_channel_counts"],
        input_seq_len=batch["input_seq_len"],
    )
    logits = outputs["neurosoft"]
    assert logits.shape == (2, 8)
    task_loss = instantiate(model.task_configs["neurosoft"].loss)
    task_loss(logits, torch.tensor([2, 2])).backward()
    assert model.session_adapter.layers["minipig"].weight.grad is not None
    assert model.session_adapter.layers["monkey"].weight.grad is not None


def test_fixed_recipe_shape_and_raw_sampling_rate_reach_frontend():
    model = _model()
    data = _data("minipig", 3)
    tokens = model.tokenize(data)
    assert data.ecog.sampling_rate == 2000.0
    assert tokens["input_values"].obj.shape == (3, 1000)
    assert (
        model.temporal_frontend[0].output_length(torch.tensor([1000])).item()
        == 250
    )
    embedding = model.encode(
        input_values=torch.randn(1, 3, 1000),
        input_session_ids=["minipig"],
        input_channel_counts=[3],
        input_seq_len=[1000],
    )
    assert embedding.shape == (1, 16)


def test_adapter_never_consumes_padded_channels_and_unknown_sessions_fail():
    model = _model()
    # The trailing values would cause a shape error if the three-channel layer
    # received all five padded channels; the valid slice is the only input.
    x = torch.randn(1, 5, 1000)
    x[:, 3:] = 1e6
    out = model.session_adapter(
        x,
        input_session_ids=["minipig"],
        input_channel_counts=[3],
        input_seq_len=[1000],
    )
    expected = model.session_adapter.layers["minipig"](
        x[0, :3].transpose(0, 1)
    ).transpose(0, 1)
    assert torch.equal(out[0], expected)
    with pytest.raises(KeyError, match="Unknown NeuroSoft session ID"):
        model.encode(
            input_values=x,
            input_session_ids=["new-session"],
            input_channel_counts=[3],
            input_seq_len=[1000],
        )


def test_time_padding_does_not_change_valid_embedding_or_prediction():
    model = _model().eval()
    # A trained affine LayerNorm has a non-zero bias. It must not turn padded
    # frames into signal that reaches valid right-edge convolution windows.
    model.temporal_frontend[0].input_norm.bias.data.fill_(1.0)
    valid = torch.randn(1, 3, 1000)
    padded = torch.nn.functional.pad(valid, (0, 100))
    with torch.no_grad():
        base = model.encode(
            input_values=valid,
            input_session_ids=["minipig"],
            input_channel_counts=[3],
            input_seq_len=[1000],
        )
        padded_embedding = model.encode(
            input_values=padded,
            input_session_ids=["minipig"],
            input_channel_counts=[3],
            input_seq_len=[1000],
        )
        base_logits = model(
            input_values=valid,
            task_index=torch.ones(1, 1, dtype=torch.long),
            input_session_ids=["minipig"],
            input_channel_counts=[3],
            input_seq_len=[1000],
        )["neurosoft"]
        padded_logits = model(
            input_values=padded,
            task_index=torch.ones(1, 1, dtype=torch.long),
            input_session_ids=["minipig"],
            input_channel_counts=[3],
            input_seq_len=[1000],
        )["neurosoft"]
    torch.testing.assert_close(base, padded_embedding, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(base_logits, padded_logits, rtol=1e-6, atol=1e-7)


def test_optional_depth_and_gru_layer_construction_paths():
    model = _model(conv_depth=2, gru_num_layers=1)
    assert len(model.temporal_frontend) == 2
    assert model.gru.num_layers == 1


def test_transfer_modes_exclude_source_adapter_and_are_atomic(tmp_path):
    source = _model({"source": 3})
    checkpoint = tmp_path / "source.ckpt"
    _save_checkpoint(source, checkpoint)

    full_target = _model({"target": 5})
    fresh_adapter = {
        k: v.detach().clone()
        for k, v in full_target.session_adapter.state_dict().items()
    }
    report = load_pretrained_weights(full_target, checkpoint)
    assert any(
        key.startswith("session_adapter.layers.source")
        for key in report.skipped_excluded
    )
    assert all(
        torch.equal(source.state_dict()[key], full_target.state_dict()[key])
        for key in full_target.state_dict()
        if key.startswith(("temporal_frontend.", "gru.", "router."))
    )
    assert all(
        torch.equal(value, full_target.session_adapter.state_dict()[key])
        for key, value in fresh_adapter.items()
    )
    assert all(
        param.requires_grad
        for param in full_target.session_adapter.parameters()
    )

    frozen_target = _model({"target": 5})
    fresh_router = {
        k: v.detach().clone()
        for k, v in frozen_target.router.state_dict().items()
    }
    report = load_pretrained_weights(
        frozen_target,
        checkpoint,
        freeze=True,
        components=frozen_target.transferable_components_for_mode(
            "frozen_representation"
        ),
    )
    assert all(
        key.startswith(("temporal_frontend.", "gru.")) for key in report.loaded
    )
    assert all(
        not param.requires_grad
        for name, param in frozen_target.named_parameters()
        if name.startswith(("temporal_frontend.", "gru."))
    )
    assert all(
        param.requires_grad
        for name, param in frozen_target.named_parameters()
        if name.startswith(("session_adapter.", "router."))
    )
    assert all(
        torch.equal(value, frozen_target.router.state_dict()[key])
        for key, value in fresh_router.items()
    )

    mismatched = _model({"target": 5}, temporal_channels=13)
    before = {k: v.detach().clone() for k, v in mismatched.state_dict().items()}
    with pytest.raises(PretrainedTransferError, match="Shape mismatches"):
        load_pretrained_weights(mismatched, checkpoint)
    assert all(
        torch.equal(value, mismatched.state_dict()[key])
        for key, value in before.items()
    )
