from types import SimpleNamespace

from foundry.models.embeddings import PatchLinearEmbedding, PerChannelStrategy
from foundry.models.poyo_eeg import POYOEEGModel
from foundry.models.tokenizer import EEGTokenizer


def _make_small_poyo_eeg_model():
    tokenizer = EEGTokenizer(
        channel_strategy=PerChannelStrategy(max_channels=4),
        temporal_embedding=PatchLinearEmbedding(
            embed_dim=32,
            num_input_channels=1,
            patch_samples=10,
        ),
        embed_dim=32,
        patch_duration=0.1,
    )
    model = POYOEEGModel(
        readout_specs=["cursor_velocity_2d"],
        tokenizer=tokenizer,
        embed_dim=32,
        sequence_length=0.5,
        latent_step=0.1,
        num_latents_per_step=2,
        depth=1,
        dim_head=8,
        cross_heads=1,
        self_heads=1,
    )
    return model


def test_initialize_vocabs_accepts_duplicate_ids():
    model = _make_small_poyo_eeg_model()

    model.initialize_vocabs(
        {
            "session_ids": ["session_a", "session_b", "session_a"],
            "channel_ids": ["ch1", "ch2", "ch1", "ch2"],
        }
    )

    assert not model.has_lazy_vocabs()


def test_readout_config_filter_preserves_normalization_options():
    model = _make_small_poyo_eeg_model()
    data = SimpleNamespace(
        config={
            "multitask_readout": [
                {
                    "readout_id": "cursor_velocity_2d",
                    "normalize_mean": [1.0, 2.0],
                    "normalize_std": [3.0, 4.0],
                },
                {"readout_id": "unsupported_readout"},
            ]
        }
    )

    model._ensure_supported_readout_config(data)

    assert data.config["multitask_readout"] == [
        {
            "readout_id": "cursor_velocity_2d",
            "normalize_mean": [1.0, 2.0],
            "normalize_std": [3.0, 4.0],
        }
    ]
