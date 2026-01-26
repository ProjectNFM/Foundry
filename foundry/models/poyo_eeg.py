from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from temporaldata import Data
from torch_brain.data import pad8, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain.utils import create_linspace_latent_tokens


class PoyoEEG(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        sequence_length: float,
        output_head: nn.Module,
        patch_size_seconds: float = 0.5,
        patch_overlap_percentage: float = 0.5,
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
        Encoder with POYOPlus-style architecture for EEG patches.

        Projects variable-sized patches to embeddings, then uses cross-attention
        to compress into latent tokens, processes with self-attention, and
        decodes through a user-provided output head.

        Args:
            output_head: User-provided nn.Module that takes latent embeddings
                of shape (batch_size, n_latents, embed_dim) and returns predictions.
        """
        super().__init__()

        self.projections = nn.ModuleDict()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.patch_size_seconds = patch_size_seconds
        self.patch_overlap_percentage = patch_overlap_percentage
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.latent_emb = Embedding(
            num_latents_per_step, embed_dim, init_scale=emb_init_scale
        )
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        self.enc_atn = RotaryCrossAttention(
            dim=embed_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim), FeedForward(dim=embed_dim, dropout=ffn_dropout)
        )

        self.proc_layers = nn.ModuleList([])
        for _ in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=embed_dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(embed_dim),
                            FeedForward(dim=embed_dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        self.output_head = output_head

    def get_projection(self, time_steps: int, channels: int) -> nn.Module:
        """
        Get or create projection layer for given dimensions.
        """
        key = f"{time_steps}_{channels}"
        if key not in self.projections:
            projection = nn.Linear(time_steps * channels, self.embed_dim)
            nn.init.xavier_uniform_(projection.weight, gain=1.0)
            nn.init.zeros_(projection.bias)
            self.projections[key] = projection
        return self.projections[key]

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with encoder-processor architecture.

        Args:
            input_values: Tensor of shape (batch_size, num_patches, time_steps, channels)
            input_timestamps: Tensor of shape (batch_size, num_patches)
            input_mask: Optional mask for input sequence
            latent_index: Indices for latent tokens (batch_size, n_latent)
            latent_timestamps: Timestamps for latent tokens (batch_size, n_latent)

        Returns:
            Output from the user-provided output head
        """
        batch_size, num_patches, time_steps, channels = input_values.shape
        projection = self.get_projection(time_steps, channels)
        flattened = input_values.view(batch_size * num_patches, -1)
        inputs = projection(flattened).view(batch_size, num_patches, self.embed_dim)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        return self.output_head(latents)

    def tokenize(self, data: Data) -> dict:
        """
        Tokenize the input data by creating overlapping patches.

        Args:
            data: TemporalData object containing EEG signal and timestamps.

        Returns:
            dict with model inputs and metadata
        """
        start, end = 0, self.sequence_length

        patch_length_samples = int(self.patch_size_seconds * data.eeg.sampling_rate)

        duration = data.domain.end - data.domain.start
        stride_seconds = self.patch_size_seconds * (1 - self.patch_overlap_percentage)
        stride_samples = int(stride_seconds * data.eeg.sampling_rate)

        num_patches = (
            int((duration.item() - self.patch_size_seconds) / stride_seconds) + 1
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

        indices = start_indices[:, None] + np.arange(patch_length_samples)[None, :]

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

        tokenized_data = {
            "input_values": pad8(torch.from_numpy(patches_array).float()),
            "input_timestamps": pad8(torch.from_numpy(patch_center_times).float()),
            "input_mask": track_mask8(patch_center_times),
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
        }

        return tokenized_data
