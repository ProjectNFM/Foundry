"""LaBraM for Masked EEG Modeling (Stage 2 pre-training).

This module implements the transformer-based model for Stage 2 of LaBraM pre-training,
which learns to predict VQ-NSP codebook token IDs at masked positions in EEG patches.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_brain.data import Data
from torch_brain.batching import pad2d
from braindecode.models.labram import Labram

from foundry.models.patch_utils import (
    extract_labram_patches,
    labram_index_tensor_to_names,
    labram_names_to_index_tensor,
)
from foundry.tasks.masking import MaskingStrategy, RandomTokenMasking


class MaskedLaBram(nn.Module):
    """MaskedLaBram: Masked EEG modeling with VQ-NSP Stage-2 (discrete codebook CE).

    Predicts VQ-NSP codebook token IDs at masked patch positions using
    a transformer backbone with channel and temporal position embeddings.

    Args:
        num_channels: Number of EEG channels.
        num_samples: Number of samples at 200 Hz (= sequence_length * 200).
        embed_dim: Embedding dimension (default: 200 for Base).
        num_layers: Number of transformer layers (default: 12 for Base).
        num_heads: Number of attention heads (default: 10 for Base).
        vocab_size: Codebook vocabulary size (default: 8192).
        drop_path_prob: Stochastic depth probability (default: 0.0).
    """

    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        embed_dim: int = 200,
        num_layers: int = 12,
        num_heads: int = 10,
        vocab_size: int = 8192,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.backbone = Labram(
            n_times=num_samples,
            n_chans=num_channels,
            n_outputs=0,
            patch_size=200,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_out_channels=8,
            drop_path_prob=drop_path_prob,
            use_mean_pooling=False,
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.lm_head.weight, std=0.02)

        self._ch_names: Optional[list[str]] = None

    def tokenize(self, data: Data) -> dict:
        """Tokenize a torch_brain Data sample into patches.

        Args:
            data: torch_brain Data object.

        Returns:
            Dictionary with:
            - input_patches: Padded patches [C, N_patches, 200]
            - channel_index: Long tensor of shape [C] into LABRAM_CHANNEL_ORDER
            - session_id: Session identifier
            - absolute_start: Segment start timestamp
        """
        input_patches, ch_names = extract_labram_patches(
            data, self.num_channels, self.num_samples
        )

        return {
            "input_patches": input_patches,
            "channel_index": labram_names_to_index_tensor(ch_names),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

    def forward(
        self,
        input_patches: torch.Tensor,
        bool_mask: Optional[torch.Tensor] = None,
        channel_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through LaBraM backbone.

        Input patches should already be masked (via apply_masking). This method
        just encodes and projects to vocabulary logits.

        Args:
            input_patches: Patches of shape [B, C, N_patches, 200] (already masked).
            bool_mask: Unused; kept for compatibility with apply_masking pipeline.
            channel_index: Optional collated ``[B, C]`` indices from tokenize().
            **kwargs: Ignored.

        Returns:
            Token logits of shape [B, N_seq, vocab_size].
        """
        if channel_index is not None:
            self._ch_names = labram_index_tensor_to_names(channel_index)
        if self._ch_names is None:
            raise RuntimeError(
                "Channel names not initialized. Pass channel_index from "
                "tokenize(), or call tokenize() in the training process."
            )

        B, C, N, patch_size = input_patches.shape
        # Reshape to continuous signal for Braindecode backbone
        input_signal = input_patches.reshape(B, C, N * patch_size)

        features = self.backbone(
            input_signal,
            ch_names=self._ch_names,
            return_all_tokens=True,
        )

        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        token_logits = self.lm_head(features)

        return token_logits

    def save_backbone(self, path: str):
        """Save backbone weights for fine-tuning.

        Saves only the backbone (not mask_token or lm_head) so weights are
        compatible with LaBraMEEGModel for downstream task fine-tuning.

        Args:
            path: Path to save .pth file.
        """
        torch.save(self.backbone.state_dict(), path)


def apply_masking(
    patches: torch.Tensor,
    masking: Optional[MaskingStrategy] = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply masking to patches with optional symmetric BEiT-v2 augmentation.

    Uses a MaskingStrategy to generate mask indices, then zero-fills the masked
    positions in the patch tensor. This is LaBraM-specific (zero-fill + all-token CE).

    Args:
        patches: Patches of shape [B, C, N_patches, patch_size].
        masking: MaskingStrategy to apply (default: RandomTokenMasking(0.5)).
        symmetric: If True, return both mask and complement (BEiT-v2 style).
            Augments batch by 2x with complementary masks.

    Returns:
        If symmetric=False:
            Tuple of (masked_patches, bool_mask) where bool_mask has shape [B, C*N]
            and True indicates a masked position.
        If symmetric=True:
            Tuple of (all_masked_patches, all_bool_masks) where batch is doubled:
            - First B samples: original mask applied
            - Second B samples: complement mask applied
    """
    if masking is None:
        masking = RandomTokenMasking(mask_ratio=0.5)

    B, C, N, patch_size = patches.shape

    # Create a dummy channel mask (all real, no padding)
    channel_mask = torch.ones(B, C, dtype=torch.bool, device=patches.device)

    # Call MaskingStrategy to get mask indices
    # Returns mask_indices: [B, num_masked] and validity_mask: [B, num_masked]
    mask_indices, _ = masking(C, N, channel_mask, device=patches.device)

    # Convert mask indices to bool mask: True where masked
    bool_mask = torch.zeros(B, C * N, dtype=torch.bool, device=patches.device)
    for b in range(B):
        bool_mask[b, mask_indices[b]] = True

    if symmetric:
        complement_mask = ~bool_mask

        # Apply original mask: zero out where mask is True
        masked_patches_a = patches.clone()
        masked_patches_b = patches.clone()

        # Vectorized zero-fill: reshape to [B, C*N, patch_size]
        patches_flat = patches.reshape(B, C * N, patch_size)
        patches_a_flat = masked_patches_a.reshape(B, C * N, patch_size)
        patches_b_flat = masked_patches_b.reshape(B, C * N, patch_size)

        patches_a_flat[bool_mask] = 0.0
        patches_b_flat[complement_mask] = 0.0

        masked_patches_a = patches_a_flat.reshape(B, C, N, patch_size)
        masked_patches_b = patches_b_flat.reshape(B, C, N, patch_size)

        all_patches = torch.cat([masked_patches_a, masked_patches_b], dim=0)
        all_masks = torch.cat([bool_mask, complement_mask], dim=0)

        return all_patches, all_masks
    else:
        # Apply mask: zero out where mask is True
        masked_patches = patches.clone()
        patches_flat = masked_patches.reshape(B, C * N, patch_size)
        patches_flat[bool_mask] = 0.0
        masked_patches = patches_flat.reshape(B, C, N, patch_size)

        return masked_patches, bool_mask
