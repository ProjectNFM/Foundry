from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from temporaldata import Data
from torch_brain.data import chain, pad8, track_mask8
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


class EEGModel(nn.Module):
    """
    POYO-style EEG model with built-in Perceiver architecture.

    This model uses a PerceiverIO backbone internally and only accepts
    input_embedding as a configurable module. All other components
    (encoder, processor, decoder) are built-in.
    """

    def __init__(
        self,
        input_embedding: nn.Module,
        readout_specs: list[ModalitySpec] | dict[str, ModalitySpec],
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
        """
        Args:
            input_embedding: Module that converts raw inputs to embeddings
            readout_specs: List of task specifications for multitask readout
            embed_dim: Embedding dimension (must match components)
            sequence_length: Length of sequences in seconds
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

        if isinstance(readout_specs, list):
            self.readout_specs = {spec.id: spec for spec in readout_specs}
        else:
            self.readout_specs = readout_specs
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

        self.unit_emb = InfiniteVocabEmbedding(
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
            input_values: Raw input tensor (shape depends on input_embedding)
            input_timestamps: Timestamps for input sequence
            input_channel_index: Channel/unit indices (batch_size, n_channels)
            input_session_index: Session index for input (batch_size,)
            input_mask: Optional mask for input sequence
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

        inputs = self.input_embedding(input_values)
        inputs = self._add_context_embeddings(
            inputs, input_channel_index, input_session_index
        )
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
            input_mask=input_mask,
        )

        output = self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output

    def _validate_vocab_initialization(self):
        """Validate that vocabularies have been properly initialized."""
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )
        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

    def _add_context_embeddings(
        self,
        inputs: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add channel and session embeddings to input patch embeddings.

        Args:
            inputs: Patch embeddings (batch_size, n_patches, embed_dim)
            input_channel_index: Channel indices (batch_size, n_channels)
            input_session_index: Session index (batch_size,)

        Returns:
            Inputs with added channel and session context (batch_size, n_patches, embed_dim)
        """
        channel_emb = self.unit_emb(input_channel_index)
        channel_emb = channel_emb.mean(dim=1, keepdim=True)

        session_emb = self.session_emb(input_session_index).unsqueeze(1)

        return inputs + channel_emb + session_emb

    def tokenize(self, data: Data) -> dict:
        """
        Tokenize the input data. Assumes data has already been patched.

        Args:
            data: TemporalData object containing patched EEG signal and timestamps.
                  Must have data.config["multitask_readout"] configured.
                  The data.eeg field should already be patched (use Patching transform).

        Returns:
            dict with model_inputs, target_values, target_weights, and metadata
        """
        start, end = 0, self.sequence_length

        if not hasattr(data, "eeg") or data.eeg is None:
            raise ValueError("Data must have an 'eeg' field")

        if data.eeg.signal.ndim != 3:
            raise ValueError(
                f"Data must be patched before tokenization. Expected 3D signal "
                f"(num_patches, channels, patch_samples), got {data.eeg.signal.ndim}D "
                f"with shape {data.eeg.signal.shape}. Use Patching transform first."
            )

        modality_field = (
            data.channels.types.astype(str)
            if hasattr(data.channels, "types")
            else np.array(["EEG"] * len(data.channels)).astype(str)
        )
        modality_mask = np.char.lower(modality_field) == "eeg"

        patches_array = data.eeg.signal[:, modality_mask, :]
        patch_center_times = data.eeg.timestamps

        channel_ids = data.channels.id[modality_mask]
        input_channel_index = self.unit_emb.tokenizer(channel_ids)

        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

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

        output_session_index = np.repeat(
            input_session_index, len(output_timestamps)
        )

        tokenized_data = {
            "input_values": pad8(torch.from_numpy(patches_array).float()),
            "input_timestamps": pad8(
                torch.from_numpy(patch_center_times).float()
            ),
            "input_channel_index": input_channel_index,
            "input_session_index": input_session_index,
            "input_mask": track_mask8(patch_center_times),
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

        return tokenized_data
