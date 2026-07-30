"""VQ-NSP (Vector-Quantized Neural Spectrum Prediction) model for LaBraM pre-training Stage 1.

This module implements the neural tokenizer component of LaBraM that converts EEG patches
into discrete spectral codes via a learned codebook. Based on the LaBraM paper:
"Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"
(Jiang et al., ICLR 2024).
"""

from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_brain.data import Data
from torch_brain.batching import pad2d

from foundry.models.patch_utils import extract_labram_patches
from foundry.tasks.config import TaskConfig


class NormEMAVectorQuantizer(nn.Module):
    """Exponential Moving Average Vector Quantizer with L2 normalization.

    Implements a codebook where:
    - Embeddings are L2-normalized (unit norm)
    - Distance metric is squared Euclidean (equivalent to cosine after normalization)
    - Codebook is updated via EMA (no gradients through codes)
    - Straight-through estimator for gradients to embeddings

    Args:
        codebook_size: Number of codebook vectors (default: 8192).
        codebook_dim: Dimension of each codebook vector (default: 32).
        decay: EMA decay factor (default: 0.99).
        epsilon: Small value for numerical stability (default: 1e-5).
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(codebook_size, codebook_dim)
        embed = embed / (torch.norm(embed, dim=1, keepdim=True) + epsilon)
        self.register_buffer("embed", embed)

        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("w", torch.zeros(codebook_size, codebook_dim))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and return codes, quantized vectors, and loss.

        Args:
            z: Input tensor of shape [*, D] where D = codebook_dim.

        Returns:
            Tuple of:
            - codes: Indices into codebook, same shape as z except last dim is 1
            - quantized: Quantized vectors with shape matching z
            - commitment_loss: Scalar commitment loss
        """
        shape = z.shape
        z_flat = z.reshape(-1, self.codebook_dim)

        z_normalized = z_flat / (torch.norm(z_flat, dim=1, keepdim=True) + self.epsilon)

        distances = (
            torch.sum(z_normalized**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_normalized, self.embed.t())
            + torch.sum(self.embed**2, dim=1, keepdim=True).t()
        )

        codes = torch.argmin(distances, dim=1)

        quantized = self.embed[codes].reshape(shape)

        if self.training:
            updated_cluster_size = (
                self.decay * self.cluster_size
                + (1 - self.decay)
                * torch.sum(
                    F.one_hot(codes, self.codebook_size).float(), dim=0
                )
            )

            n = torch.sum(updated_cluster_size)
            updated_cluster_size = (
                (updated_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )

            dw = torch.matmul(
                F.one_hot(codes, self.codebook_size).t().float(),
                z_flat,
            )
            updated_w = self.decay * self.w + (1 - self.decay) * dw

            self.cluster_size.data = updated_cluster_size
            self.w.data = updated_w

            embed_normalized = updated_w / (
                updated_cluster_size.unsqueeze(1) + self.epsilon
            )
            embed_normalized = embed_normalized / (
                torch.norm(embed_normalized, dim=1, keepdim=True) + self.epsilon
            )
            self.embed.data = embed_normalized

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = q_latent_loss + 0.25 * e_latent_loss

        quantized = z + (quantized - z).detach()

        return codes.reshape(*shape[:-1]), quantized, commitment_loss


class VQNSPModel(nn.Module):
    """VQ-NSP: Vector-Quantized Neural Spectrum Prediction.

    Stage 1 of LaBraM pre-training. Learns to encode raw EEG patches into discrete
    spectral codes by predicting FFT amplitude and phase through a learned codebook.

    Args:
        num_channels: Number of EEG channels.
        num_samples: Number of samples at 200 Hz (= sequence_length * 200).
        codebook_size: Codebook vocabulary size (default: 8192).
        codebook_dim: Codebook embedding dimension (default: 32).
        out_chans: Output channels of TemporalConv encoder (default: 8).
        pretrained_path: Optional path to .pth checkpoint to load (skips training).
    """

    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        out_chans: int = 8,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_samples = num_samples
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.out_chans = out_chans

        self.encoder = nn.Sequential(
            nn.Conv1d(1, out_chans, kernel_size=15, stride=8, padding=7),
            nn.GELU(),
            nn.GroupNorm(4, out_chans),
            nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, out_chans),
            nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1),
        )

        self.codebook = NormEMAVectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        self.pre_vq_conv = nn.Conv1d(out_chans, codebook_dim, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, out_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, out_chans),
            nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, out_chans),
            nn.ConvTranspose1d(
                out_chans, 1, kernel_size=15, stride=8, padding=7, output_padding=0
            ),
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to .pth file.
        """
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            warnings.warn(
                f"Loaded pretrained VQ-NSP weights from {checkpoint_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained checkpoint from {checkpoint_path}: {e}"
            ) from e

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches to discrete codes (used by Stage 2 as targets).

        Args:
            patches: Tensor of shape [B, C, N_patches, 200] (after collate).

        Returns:
            Codes of shape [B, C, N_patches].
        """
        B, C, N, patch_size = patches.shape
        patches_flat = patches.reshape(B * C * N, 1, patch_size)

        z = self.encoder(patches_flat)
        z = self.pre_vq_conv(z)
        z = z.transpose(1, 2)

        codes, _, _ = self.codebook(z)

        codes = codes.reshape(B, C, N)
        return codes

    def tokenize(self, data: Data) -> dict:
        """Tokenize a torch_brain Data sample into patches.

        Args:
            data: torch_brain Data object.

        Returns:
            Dictionary with:
            - input_patches: Padded patches [C, N_patches, 200]
            - session_id: Session identifier
            - absolute_start: Segment start timestamp
        """
        input_patches, _ = extract_labram_patches(
            data, self.num_channels, self.num_samples
        )

        return {
            "input_patches": input_patches,
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

    def forward(
        self,
        input_patches: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode patches, quantize, decode.

        Args:
            input_patches: Patches of shape [B, C, N_patches, 200] after collate.
            **kwargs: Ignored (for compatibility).

        Returns:
            Dictionary with:
            - amplitude_pred: Predicted FFT amplitude
            - phase_pred: Predicted FFT phase
            - codes: Discrete codes from codebook
            - vq_loss: Commitment loss from quantizer
        """
        B, C, N, patch_size = input_patches.shape

        patches_flat = input_patches.reshape(B * C * N, 1, patch_size)

        z = self.encoder(patches_flat)
        z = self.pre_vq_conv(z)
        z_t = z.transpose(1, 2)

        codes, quantized, vq_loss = self.codebook(z_t)
        quantized = quantized.transpose(1, 2)

        decoded = self.decoder(quantized)

        fft_real = torch.fft.rfft(patches_flat, dim=2)
        target_amplitude = torch.abs(fft_real)
        target_phase = torch.angle(fft_real)

        decoded_real = torch.fft.rfft(decoded, dim=2)
        decoded_amplitude = torch.abs(decoded_real)
        decoded_phase = torch.angle(decoded_real)

        amplitude_loss = F.mse_loss(decoded_amplitude, target_amplitude)
        phase_loss = F.mse_loss(decoded_phase, target_phase)

        total_loss = amplitude_loss + phase_loss + vq_loss

        return {
            "amplitude_pred": decoded_amplitude,
            "phase_pred": decoded_phase,
            "codes": codes.reshape(B, C, N),
            "vq_loss": vq_loss,
            "amplitude_loss": amplitude_loss,
            "phase_loss": phase_loss,
            "total_loss": total_loss,
        }
