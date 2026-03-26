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

from foundry.data.transforms import Patching
from foundry.models.backbones import PerceiverIOBackbone
from foundry.models.utils import resolve_readout_specs


class POYOEEGModel(nn.Module):
    """
    POYO-style EEG model with built-in Perceiver architecture.

    This model uses a PerceiverIO backbone internally and only accepts
    input_embedding as a configurable module. All other components
    (encoder, processor, decoder) are built-in.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}

    def __init__(
        self,
        input_embedding: nn.Module,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        embed_dim: int,
        sequence_length: float,
        patch_duration: float,
        stride: Optional[float] = None,
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
        """
        Args:
            input_embedding: Module that converts raw inputs to embeddings.
                Receives (batch, num_patches, num_channels, patch_samples).
            readout_specs: List/dict of task specifications for multitask readout.
                Can be ModalitySpec objects or string names that resolve from registry.
            embed_dim: Embedding dimension (must match components)
            sequence_length: Length of sequences in seconds
            patch_duration: Duration of each patch in seconds
            stride: Step size between patches in seconds. Defaults to patch_duration (non-overlapping).
            latent_step: Time step between latent tokens in seconds
            num_latents_per_step: Number of latent tokens per time step
            depth: Number of processor layers
            dim_head: Dimension per attention head
            cross_heads: Number of attention heads for cross-attention
            self_heads: Number of attention heads for self-attention
            ffn_dropout: Dropout rate for feedforward networks
            lin_dropout: Dropout rate for linear layers in processor
            atn_dropout: Dropout rate for attention layers
            emb_init_scale: Initialization scale for embeddings
            t_min: Minimum time value for rotary encoding
            t_max: Maximum time value for rotary encoding
        """
        super().__init__()

        self.input_embedding = input_embedding
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

        self.patching = Patching(patch_duration=patch_duration, stride=stride)

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
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor,
        output_decoder_index: torch.Tensor,
        unpack_output: bool = False,
    ) -> Dict:
        """
        Forward pass through the model.

        Args:
            input_values: Patched signal (batch_size, num_patches, num_channels, patch_samples)
            input_timestamps: Timestamps per patch (batch_size, num_patches)
            input_channel_index: Channel tokens (batch_size, num_channels)
            input_session_index: Session index for input (batch_size,)
            input_mask: Optional channel validity mask (batch_size, num_channels)
            latent_index: Indices for latent tokens (batch_size, n_latent)
            latent_timestamps: Timestamps for latent tokens (batch_size, n_latent)
            output_session_index: Session indices for output queries (batch_size, n_out)
            output_timestamps: Timestamps for output predictions (batch_size, n_out)
            output_decoder_index: Task/decoder indices for each output (batch_size, n_out)
            unpack_output: Whether to unpack outputs by batch sample

        Returns:
            Dictionary of task-specific outputs
        """
        self._validate_vocab_initialization()

        inputs = self.input_embedding(
            input_values,
            input_channel_index=input_channel_index,
            input_mask=input_mask,
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
        # Returns task specs
        return self._readout_specs

    def _resolve_signal_source(self, data: Data):
        """Find the signal source and default modality type from the data.

        Searches for the first available modality field (eeg, ecog, seeg) on
        the data object.

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
        """
        Tokenize the input data. Performs patching internally and converts to model tokens.

        The signal is kept in its natural (num_patches, num_channels, patch_samples) shape.
        Only the channel dimension is padded to match the embedding layer's num_channels.

        Args:
            data: TemporalData object containing raw EEG/ECoG/sEEG signal.
                  If data.config["multitask_readout"] is set by the dataset, it will be
                  intersected with the model's readout_specs to use only supported modalities.

        Returns:
            dict with model_inputs, target_values, target_weights, and metadata
        """
        data = self.patching(data)

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

        patches_array = signal_source.signal[:, modality_mask, :]
        patch_center_times = signal_source.timestamps

        channel_ids = data.channels.id[modality_mask].astype(str)
        channel_tokens = np.asarray(self.channel_emb.tokenizer(channel_ids))

        num_patches, num_channels_actual, patch_samples = patches_array.shape
        num_channels = self.input_embedding.num_channels

        if num_channels_actual > num_channels:
            # raise ValueError(
            #     f"Data has {num_channels_actual} channels but model expects "
            #     f"at most {num_channels}"
            # )
            # When there are too many channels, we just use the first num_channels channels
            patches_array = patches_array[:, :num_channels, :]
            channel_ids = channel_ids[:num_channels]
            channel_tokens = channel_tokens[:num_channels]
            num_channels_actual = num_channels

        padded_signal = np.zeros(
            (num_patches, num_channels, patch_samples),
            dtype=patches_array.dtype,
        )
        padded_signal[:, :num_channels_actual, :] = patches_array

        padded_channel_tokens = np.zeros(
            num_channels, dtype=channel_tokens.dtype
        )
        padded_channel_tokens[:num_channels_actual] = channel_tokens

        channel_mask = np.zeros(num_channels, dtype=bool)
        channel_mask[:num_channels_actual] = True

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
            "input_values": torch.from_numpy(padded_signal).float(),
            "input_timestamps": torch.from_numpy(
                np.asarray(patch_center_times)
            ).float(),
            "input_channel_index": torch.from_numpy(
                padded_channel_tokens
            ).long(),
            "input_session_index": input_session_index,
            "input_mask": torch.from_numpy(channel_mask),
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
            vocab_info: Dictionary with 'session_ids' and 'channel_ids' keys
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
