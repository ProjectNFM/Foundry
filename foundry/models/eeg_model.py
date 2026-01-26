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
from torch_brain.utils import create_linspace_latent_tokens


class EEGModel(nn.Module):
    """
    Flexible EEG model that composes independent building blocks.

    This model is a reference implementation showing how to wire together
    embedding, backbone, and readout components. You can use this as-is or
    build your own composition using the individual building blocks however
    you want.

    The model doesn't enforce any interfaces - components are wired together
    through explicit method calls in the forward pass.
    """

    def __init__(
        self,
        input_embedding: nn.Module,
        backbone: nn.Module,
        readout_specs: list[str],
        embed_dim: int,
        sequence_length: float,
        patch_size_seconds: float = 0.5,
        patch_overlap_percentage: float = 0.5,
        latent_step: float = 0.1,
        num_latents_per_step: int = 1,
        emb_init_scale: float = 0.02,
        dim_head: int = 64,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
    ):
        """
        Args:
            input_embedding: Module that converts raw inputs to embeddings
            backbone: Module that processes embeddings (can be any architecture)
            readout_specs: List of task specifications for multitask readout
            embed_dim: Embedding dimension (must match components)
            sequence_length: Length of sequences in seconds
            patch_size_seconds: Size of each patch in seconds
            patch_overlap_percentage: Overlap percentage between consecutive patches
            latent_step: Time step between latent tokens in seconds
            num_latents_per_step: Number of latent tokens per time step
            emb_init_scale: Initialization scale for embeddings
            dim_head: Dimension per attention head for rotary embeddings
            t_min: Minimum time value for rotary encoding
            t_max: Maximum time value for rotary encoding
        """
        super().__init__()

        self.input_embedding = input_embedding
        self.backbone = backbone
        self.embed_dim = embed_dim

        self.sequence_length = sequence_length
        self.patch_size_seconds = patch_size_seconds
        self.patch_overlap_percentage = patch_overlap_percentage
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.readout_specs = {spec.id: spec for spec in readout_specs}
        self.global_to_local_task_id = {
            spec.id: idx for idx, spec in enumerate(self.readout_specs.values())
        }

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

        This implementation shows one way to wire components together.
        You can build your own forward pass using the components however you want.

        Args:
            input_values: Raw input tensor (shape depends on input_embedding)
            input_timestamps: Timestamps for input sequence
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
        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        inputs = self.input_embedding(input_values)
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

    def tokenize(self, data: Data) -> dict:
        """
        Tokenize the input data by creating overlapping patches.

        Args:
            data: TemporalData object containing EEG signal and timestamps.
                  Must have data.config["multitask_readout"] configured.

        Returns:
            dict with model_inputs, target_values, target_weights, and metadata
        """
        start, end = 0, self.sequence_length

        patch_length_samples = int(
            self.patch_size_seconds * data.eeg.sampling_rate
        )

        duration = data.domain.end - data.domain.start
        stride_seconds = self.patch_size_seconds * (
            1 - self.patch_overlap_percentage
        )
        stride_samples = int(stride_seconds * data.eeg.sampling_rate)

        num_patches = (
            int((duration.item() - self.patch_size_seconds) / stride_seconds)
            + 1
        )

        patch_indices = np.arange(num_patches)
        start_indices = patch_indices * stride_samples
        end_indices = start_indices + patch_length_samples

        valid_mask = end_indices <= len(data.eeg.signal)

        if not np.any(valid_mask):
            raise ValueError("No valid patches created from the data")

        start_indices = start_indices[valid_mask]
        end_indices = end_indices[valid_mask]

        mid_indices = (start_indices + end_indices) // 2
        patch_center_times = data.eeg.timestamps[mid_indices]

        indices = (
            start_indices[:, None] + np.arange(patch_length_samples)[None, :]
        )

        modality_field = (
            data.units.modality.astype(str)
            if hasattr(data.units, "modality")
            else data.units.standard_types
        )
        modality_mask = modality_field == "EEG"
        patches_array = data.eeg.signal[indices][:, :, modality_mask]

        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        session_index = self.session_emb.tokenizer(data.session.id)

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

        session_index = np.repeat(session_index, len(output_timestamps))

        tokenized_data = {
            "input_values": pad8(torch.from_numpy(patches_array).float()),
            "input_timestamps": pad8(
                torch.from_numpy(patch_center_times).float()
            ),
            "input_mask": track_mask8(patch_center_times),
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "output_session_index": pad8(session_index),
            "output_timestamps": pad8(output_timestamps),
            "output_decoder_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }

        return tokenized_data
