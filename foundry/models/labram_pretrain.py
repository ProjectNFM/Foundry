"""LaBraM for Masked EEG Modeling (Stage 2 pre-training).

This module implements the transformer-based model for Stage 2 of LaBraM pre-training,
which learns to predict VQ-NSP codebook token IDs at masked positions in EEG patches.
"""

from typing import Optional
import warnings

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


class LaBraMForMaskedEEGModeling(nn.Module):
    """LaBraM Neural Transformer for Masked EEG Modeling (Stage 2).

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

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

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
        """Forward pass with optional masking.

        Args:
            input_patches: Patches of shape [B, C, N_patches, 200] after collate.
            bool_mask: Optional mask of shape [B, N_seq] where True = masked.
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

        if len(input_patches.shape) == 4:
            input_patches = input_patches.reshape(B, C, N * patch_size)

        if bool_mask is not None:
            flat_mask = bool_mask.reshape(-1)
            masked_patches = input_patches.clone()

            for i, is_masked in enumerate(flat_mask):
                if is_masked:
                    masked_patches.reshape(B * C, -1)[i] = 0.0

            input_patches = masked_patches

        features = self.backbone(
            input_patches,
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
    mask_ratio: float = 0.5,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking to patches (symmetric masking for BEiT-v2 style).

    Args:
        patches: Patches of shape [B, C, N_patches, 200].
        mask_ratio: Fraction of patches to mask (default: 0.5).
        symmetric: If True, return both mask and complement (default: True).

    Returns:
        If symmetric=False:
            Tuple of (masked_patches, bool_mask)
        If symmetric=True:
            Tuple of (all_masked_patches, all_bool_masks) where shapes are doubled
            along batch dimension (first half: original mask, second half: complement)
    """
    B, C, N, patch_size = patches.shape
    N_total = C * N

    mask_size = int(N_total * mask_ratio)

    bool_mask = torch.ones(B, N_total, dtype=torch.bool, device=patches.device)
    for b in range(B):
        idx = torch.randperm(N_total, device=patches.device)[:mask_size]
        bool_mask[b, idx] = False

    if symmetric:
        complement_mask = ~bool_mask

        masked_patches_a = patches.clone()
        masked_patches_b = patches.clone()

        for b in range(B):
            for c in range(C):
                for n in range(N):
                    idx = c * N + n
                    if not bool_mask[b, idx]:
                        masked_patches_a[b, c, n] = 0.0
                    if not complement_mask[b, idx]:
                        masked_patches_b[b, c, n] = 0.0

        all_patches = torch.cat([masked_patches_a, masked_patches_b], dim=0)
        all_masks = torch.cat([bool_mask, complement_mask], dim=0)

        return all_patches, all_masks
    else:
        masked_patches = patches.clone()
        for b in range(B):
            for c in range(C):
                for n in range(N):
                    idx = c * N + n
                    if not bool_mask[b, idx]:
                        masked_patches[b, c, n] = 0.0

        return masked_patches, bool_mask
