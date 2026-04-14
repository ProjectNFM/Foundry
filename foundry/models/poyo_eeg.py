from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from temporaldata import Data
from torch_brain.data import chain, pad8
from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    RotaryTimeEmbedding,
    prepare_for_multitask_readout,
)
from torch_brain.registry import ModalitySpec
from torch_brain.utils import create_linspace_latent_tokens

from foundry.models.backbones import PerceiverIOBackbone
from foundry.models.tokenizer import EEGTokenizer
from foundry.models.utils import resolve_readout_specs


class POYOEEGModel(nn.Module):
    """POYO-style EEG model with built-in Perceiver architecture.

    This model uses a PerceiverIO backbone internally and accepts an
    :class:`EEGTokenizer` that composes a channel strategy with a
    temporal embedding to convert raw EEG signal into token sequences.

    Args:
        tokenizer: Composable tokenizer handling channel strategy,
            optional GPU patching, and temporal embedding.
        readout_specs: List/dict of task specifications for multitask readout.
            Can be ModalitySpec objects or string names that resolve from
            the registry.
        embed_dim: Embedding dimension (must match components).
        sequence_length: Length of sequences in seconds.
        latent_step: Time step between latent tokens in seconds.
        num_latents_per_step: Number of latent tokens per time step.
        depth: Number of processor layers.
        dim_head: Dimension per attention head.
        cross_heads: Number of attention heads for cross-attention.
        self_heads: Number of attention heads for self-attention.
        ffn_dropout: Dropout rate for feedforward networks.
        lin_dropout: Dropout rate for linear layers in processor.
        atn_dropout: Dropout rate for attention layers.
        emb_init_scale: Initialization scale for embeddings.
        t_min: Minimum time value for rotary encoding.
        t_max: Maximum time value for rotary encoding.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}

    def __init__(
        self,
        tokenizer: EEGTokenizer,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        embed_dim: int,
        sequence_length: float,
        latent_step: float = 0.1,
        num_latents_per_step: int = 1,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self._latent_index, self._latent_timestamps = (
            create_linspace_latent_tokens(
                0,
                self.sequence_length,
                step=self.latent_step,
                num_latents_per_step=self.num_latents_per_step,
            )
        )

        self._readout_specs = resolve_readout_specs(readout_specs)
        self.global_to_local_task_id = {
            spec.id: idx for idx, spec in enumerate(self.readout_specs.values())
        }

        self.backbone = PerceiverIOBackbone(
            embed_dim=embed_dim,
            depth=depth,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
        )

        self.channel_emb = InfiniteVocabEmbedding(
            self.embed_dim, init_scale=emb_init_scale
        )
        self.session_emb = InfiniteVocabEmbedding(
            self.embed_dim, init_scale=emb_init_scale
        )
        self.task_emb = Embedding(
            len(self.readout_specs), self.embed_dim, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(
            num_latents_per_step, self.embed_dim, init_scale=emb_init_scale
        )
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.readout = MultitaskReadout(
            dim=self.embed_dim,
            readout_specs=self.readout_specs,
        )

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        input_sampling_rate: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor,
        output_decoder_index: torch.Tensor,
        unpack_output: bool = False,
    ) -> Dict:
        """Forward pass through the model.

        Args:
            input_values: (B, C, T) raw signal (padded to max T in batch).
            input_timestamps: (B, num_tokens) timestamps per token.
            input_channel_index: (B, C) channel identity tokens.
            input_session_index: (B,) session index for input.
            input_mask: (B, C) channel validity mask.
            input_sampling_rate: (B,) per-item sampling rate.
            input_seq_len: (B,) per-item true sample count.
            latent_index: (B, n_latent) indices for latent tokens.
            latent_timestamps: (B, n_latent) timestamps for latent tokens.
            output_session_index: (B, n_out) session indices for outputs.
            output_timestamps: (B, n_out) timestamps for output predictions.
            output_decoder_index: (B, n_out) task/decoder indices.
            unpack_output: Whether to unpack outputs by batch sample.

        Returns:
            Dictionary of task-specific outputs.
        """
        self._validate_vocab_initialization()

        inputs = self.tokenizer(
            input_values,
            input_channel_index=input_channel_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            channel_emb_fn=self.channel_emb,
        )

        session_emb = self.session_emb(input_session_index).unsqueeze(1)
        inputs = inputs + session_emb

        input_timestamp_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        local_task_index = torch.zeros_like(output_decoder_index)
        for global_id, local_id in self.global_to_local_task_id.items():
            local_task_index[output_decoder_index == global_id] = local_id

        output_queries = self.session_emb(output_session_index) + self.task_emb(
            local_task_index
        )
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        output_latents = self.backbone(
            inputs=inputs,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=output_queries,
            output_timestamp_emb=output_timestamp_emb,
        )

        output = self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output

    @property
    def readout_specs(self) -> dict[str, ModalitySpec]:
        return self._readout_specs

    def _resolve_signal_source(self, data: Data):
        """Find the signal source and default modality type from the data.

        Returns:
            Tuple of (signal_source, default_type) where signal_source is the
            time series object and default_type is the uppercase modality name
            used when channels.type is absent.
        """
        for modality in ["eeg", "ecog", "seeg"]:
            signal = getattr(data, modality, None)
            if signal is not None:
                return signal, modality.upper()

        raise ValueError("Data must have an 'eeg', 'ecog', or 'seeg' field")

    def tokenize(self, data: Data) -> dict:
        """Tokenize the input data.

        Delegates signal preparation to the :class:`EEGTokenizer` which
        handles channel strategy, optional GPU patching, and temporal
        embedding concerns.

        Args:
            data: TemporalData object containing raw EEG/ECoG/sEEG signal.
                  If ``data.config["multitask_readout"]`` is set by the
                  dataset, it will be intersected with the model's
                  ``readout_specs`` to use only supported modalities.

        Returns:
            dict with model_inputs, target_values, target_weights, and
            metadata.
        """
        if not hasattr(data, "config") or data.config is None:
            data.config = {}

        if "multitask_readout" not in data.config:
            data.config["multitask_readout"] = [
                {"readout_id": spec_id} for spec_id in self.readout_specs.keys()
            ]
        else:
            available = [
                cfg["readout_id"] for cfg in data.config["multitask_readout"]
            ]
            data.config["multitask_readout"] = [
                {"readout_id": name}
                for name in available
                if name in self.readout_specs
            ]

        signal_source, default_type = self._resolve_signal_source(data)

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )

        channel_ids = data.channels.id[modality_mask].astype(str)
        channel_tokens = np.asarray(self.channel_emb.tokenizer(channel_ids))

        sample_deltas = np.diff(signal_source.timestamps)
        sampling_rate = 1.0 / float(sample_deltas[0])

        pretokenized = self.tokenizer.pretokenize(
            signal=signal_source.signal[:, modality_mask],
            channel_tokens=channel_tokens,
            sampling_rate=sampling_rate,
            sequence_length=self.sequence_length,
        )
        input_timestamps = pretokenized.pop("input_timestamps")

        latent_index = self._latent_index
        latent_timestamps = self._latent_timestamps

        input_session_index = self.session_emb.tokenizer(data.session.id)

        (
            output_timestamps,
            output_values,
            output_task_index,
            output_weights,
            output_eval_mask,
        ) = prepare_for_multitask_readout(
            data,
            self.readout_specs,
        )
        output_timestamps = torch.zeros_like(output_timestamps)

        output_session_index = np.full(
            len(output_timestamps), input_session_index
        )

        return {
            **pretokenized,
            "input_timestamps": input_timestamps,
            "input_session_index": input_session_index,
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "output_session_index": pad8(output_session_index),
            "output_timestamps": pad8(output_timestamps),
            "output_decoder_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }

    def initialize_vocabs(self, vocab_info: dict):
        """Initialize vocabularies from dataset information.

        Args:
            vocab_info: Dictionary with ``session_ids`` and ``channel_ids``
                keys.
        """
        if "session_ids" in vocab_info and self.session_emb.is_lazy():
            self.session_emb.initialize_vocab(vocab_info["session_ids"])

        if "channel_ids" in vocab_info and self.channel_emb.is_lazy():
            self.channel_emb.initialize_vocab(vocab_info["channel_ids"])

    def has_lazy_vocabs(self) -> bool:
        """Check if vocabularies are still lazy (uninitialized)."""
        return self.channel_emb.is_lazy() or self.session_emb.is_lazy()

    def _validate_vocab_initialization(self):
        """Validate that vocabularies have been properly initialized."""
        if self.channel_emb.is_lazy():
            raise ValueError(
                "Channel vocabulary has not been initialized, please use "
                "`model.channel_emb.initialize_vocab(channel_ids)`"
            )
        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )
