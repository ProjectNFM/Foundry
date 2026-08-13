from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from foundry.models import EEGTokenizer, FixedChannelStrategy, POYOEEGModel
from foundry.models.embeddings.temporal import PatchLinearEmbedding
from foundry.tasks.config import TaskConfig


def _make_model(**kwargs) -> POYOEEGModel:
    embed_dim = 16
    tokenizer = EEGTokenizer(
        channel_strategy=FixedChannelStrategy(num_channels=4),
        temporal_embedding=PatchLinearEmbedding(
            embed_dim=embed_dim,
            num_input_channels=4,
            patch_samples=50,
        ),
        embed_dim=embed_dim,
        patch_duration=0.5,
        stride=0.5,
        channel_fusion="add",
    )
    task = TaskConfig.from_yaml(
        "configs/tasks/neurosoft_acoustic_stim_8band.yaml"
    )
    return POYOEEGModel(
        tokenizer=tokenizer,
        task_configs={task.name: task},
        embed_dim=embed_dim,
        sequence_length=2.0,
        latent_step=0.5,
        num_latents_per_step=1,
        **kwargs,
    )


def test_decoder_source_ids_must_be_unique():
    with pytest.raises(ValueError, match="duplicate"):
        _make_model(decoder_source_ids=["minipigs", "minipigs"])


def test_source_ids_are_expanded_to_each_output_query():
    model = _make_model(decoder_source_ids=["minipigs", "monkeys"])
    task_index = torch.tensor([[1, 1, 0], [1, 0, 0]])

    source_index = model.source_ids_to_output_index(
        ["minipigs", "monkeys"], task_index
    )

    assert torch.equal(
        source_index,
        torch.tensor([[0, 0, 0], [1, 1, 1]]),
    )


def test_unknown_decoder_source_fails_clearly():
    model = _make_model(decoder_source_ids=["minipigs", "monkeys"])

    with pytest.raises(ValueError, match="Unknown decoder source 'humans'"):
        model.source_ids_to_output_index(
            ["humans"], torch.ones(1, 2, dtype=torch.long)
        )


def test_decoder_queries_include_source_embedding():
    model = _make_model(decoder_source_ids=["minipigs", "monkeys"])
    model.session_emb.initialize_vocab(["session"])
    with torch.no_grad():
        model.session_emb.weight.zero_()
        model.task_emb.weight.zero_()
        model.decoder_source_emb.weight[0].fill_(1.0)
        model.decoder_source_emb.weight[1].fill_(2.0)

    queries, _ = model._build_downstream_queries(
        output_session_index=torch.ones(2, 1, dtype=torch.long),
        task_index=torch.ones(2, 1, dtype=torch.long),
        output_timestamps=torch.zeros(2, 1),
        output_source_index=torch.tensor([[0], [1]]),
    )

    assert torch.equal(queries[0], torch.ones_like(queries[0]))
    assert torch.equal(queries[1], torch.full_like(queries[1], 2.0))


def test_source_conditioned_queries_require_source_index():
    model = _make_model(decoder_source_ids=["minipigs", "monkeys"])
    model.session_emb.initialize_vocab(["session"])

    with pytest.raises(ValueError, match="output_source_index"):
        model._build_downstream_queries(
            output_session_index=torch.ones(1, 1, dtype=torch.long),
            task_index=torch.ones(1, 1, dtype=torch.long),
            output_timestamps=torch.zeros(1, 1),
        )


def test_encoder_session_embedding_can_be_disabled_independently():
    model = _make_model(use_encoder_session_embedding=False)
    model.session_emb.initialize_vocab(["session"])
    with torch.no_grad():
        model.session_emb.weight.fill_(3.0)
    base_tokens = torch.randn(1, 3, model.embed_dim)

    with patch.object(
        model.tokenizer,
        "forward",
        return_value=(base_tokens.clone(), None),
    ):
        inputs, session_emb, _ = model._tokenize_and_add_session(
            input_values=torch.zeros(1, 4, 100),
            input_channel_index=torch.zeros(1, 4, dtype=torch.long),
            input_session_index=torch.ones(1, dtype=torch.long),
        )

    assert torch.equal(inputs, base_tokens)
    assert not torch.equal(session_emb, torch.zeros_like(session_emb))
