"""LaBraM (Large Brain Model) for EEG/ECoG/sEEG foundation model integration.

LaBraM is a patch-based EEG transformer pretrained on ~2500 hours of diverse EEG data
(Jiang et al., ICLR 2024). This implementation wraps the Braindecode Labram model
to integrate seamlessly with Foundry's torch_brain datamodule and multitask training pipeline.

Key features:
- Patch-based architecture with learned temporal and spatial embeddings
- Pretrained weights available via HuggingFace Hub (braindecode/labram-pretrained)
- Automatic channel reordering to match 128 standard 10-20 EEG channel positions
- Configurable model sizes (Base, Large, Huge)
- Support for fine-tuning or training from scratch
"""

from typing import Dict, Optional, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch_brain.data import Data
from torch_brain.batching import chain, pad8, pad2d

from braindecode.models.labram import Labram
from foundry.models.patch_utils import (
    labram_index_tensor_to_names,
    labram_names_to_index_tensor,
    prepare_labram_continuous_signal,
    resolve_labram_channels,
)
from foundry.models.signal_preparation import (
    infer_sampling_rate_from_timestamps,
    resolve_signal_source,
)
from foundry.models.readout import build_readout_router
from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import extract_multitask_targets


class LaBraMEEGModel(nn.Module):
    """LaBraM foundation model adapted for Foundry's multitask EEG learning.

    This model wraps Braindecode's Labram implementation to support:
    - Tokenization of torch_brain Data objects with arbitrary sampling rates
    - Automatic resampling to 200 Hz (LaBraM's pretraining rate)
    - Filtering to canonical LaBraM channel order (128 standard 10-20 channels)
    - Multitask readout via Foundry's ReadoutRouter

    Args:
        task_configs: Mapping from task name to TaskConfig.
        num_channels: Number of active EEG channels (after filtering to LABRAM_CHANNEL_ORDER).
        num_samples: Number of time samples at 200 Hz (= int(sequence_length * 200)).
        patch_size: Temporal patch size in samples. Default: 200 (1 s at 200 Hz).
        embed_dim: Embedding dimension. Default: 200 (Base). Set to 400 (Large) or 800 (Huge).
        num_layers: Number of transformer layers. Default: 12 (Base). Set to 24 (Large) or 48 (Huge).
        num_heads: Number of attention heads. Default: 10 (Base). Set to 16 (Large/Huge).
        conv_out_channels: Output channels for temporal convolution. Default: 8 (Base). Set to 16 (Large) or 32 (Huge).
        drop_path_prob: Stochastic depth (DropPath) probability. Default: 0.0.
        pretrained: If True, load pretrained weights from HuggingFace Hub (braindecode/labram-pretrained).
        pretrained_path: Path to a local .pth checkpoint to load. Overrides pretrained=True if set.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}
    TARGET_SAMPLING_RATE = 200  # Hz; LaBraM pretraining rate

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int,
        num_samples: int,
        patch_size: int = 200,
        embed_dim: int = 200,
        num_layers: int = 12,
        num_heads: int = 10,
        conv_out_channels: int = 8,
        drop_path_prob: float = 0.0,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self._task_configs = TaskConfig.normalize_task_configs(task_configs)
        self._ch_names: Optional[list[str]] = None

        self.backbone = Labram(
            n_times=num_samples,
            n_chans=num_channels,
            n_outputs=0,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_out_channels=conv_out_channels,
            drop_path_prob=drop_path_prob,
            use_mean_pooling=True,
        )

        self.router = build_readout_router(self._task_configs, embed_dim)

        if pretrained_path is not None:
            self._load_pretrained_checkpoint(pretrained_path)
        elif pretrained:
            self._load_pretrained_hub(num_samples)

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    def _load_pretrained_hub(self, num_samples: int):
        """Load pretrained weights from HuggingFace Hub.

        Args:
            num_samples: Number of time samples (used for compatible shape check).
        """
        try:
            pretrained = Labram.from_pretrained(
                "braindecode/labram-pretrained",
                n_outputs=0,
                n_chans=self.num_channels,
                n_times=num_samples,
            )
            self.backbone.load_state_dict(
                pretrained.state_dict(), strict=False
            )
            warnings.warn(
                "Loaded pretrained LaBraM weights from HuggingFace Hub. "
                "Temporal embedding may have been reinitialized for this sequence length."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained LaBraM from Hub: {e}"
            ) from e

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from a local checkpoint.

        Args:
            checkpoint_path: Path to .pth or .pt checkpoint file.
        """
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)
            warnings.warn(
                f"Loaded pretrained LaBraM weights from {checkpoint_path}. "
                "Some keys may have been skipped (strict=False)."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained checkpoint from {checkpoint_path}: {e}"
            ) from e

    def _resolve_signal_source(self, data: Data) -> tuple[Any, str, float]:
        """Find the signal source, default modality type, and sampling rate (shared impl)."""
        return resolve_signal_source(data)

    def _infer_sampling_rate_from_timestamps(
        self, timestamps: np.ndarray
    ) -> float:
        return infer_sampling_rate_from_timestamps(timestamps)

    def _extract_targets(self, data: Data):
        """Extract multitask targets from data.

        Returns:
            Tuple of (output_timestamps, output_values, output_task_index, output_weights).
        """
        return extract_multitask_targets(self._task_configs, data)

    def tokenize(self, data: Data) -> dict[str, Any]:
        """Tokenize a torch_brain Data sample for the dataloader.

        Converts raw EEG signal to LaBraM-compatible format:
        1. Extracts signal from data.eeg/ecog/seeg
        2. Resamples to 200 Hz (if needed)
        3. Filters/reorders channels to LABRAM_CHANNEL_ORDER
        4. Extracts multitask targets

        Channel names are returned as a ``channel_index`` tensor so they survive
        DataLoader multiprocessing (mutating ``self._ch_names`` in workers does
        not update the training-process model).

        Args:
            data: torch_brain Data object with EEG/ECoG/sEEG signal.

        Returns:
            Dictionary with:
            - input_values: Tensor of shape [T', C] (resampled to 200 Hz)
            - channel_index: Long tensor of shape [C] into LABRAM_CHANNEL_ORDER
            - task_index: Task indices for multitask routing
            - target_values: Multitask target values
            - target_weights: Multitask target weights
            - session_id: Session identifier
            - absolute_start: Segment start timestamp
        """
        signal, ch_names = prepare_labram_continuous_signal(
            data, self.num_channels, self.num_samples, self.TARGET_SAMPLING_RATE
        )
        channel_index = labram_names_to_index_tensor(ch_names)

        x = torch.from_numpy(signal)

        (
            output_timestamps,
            output_values,
            output_task_index,
            output_weights,
        ) = self._extract_targets(data)

        return {
            "input_values": pad2d(x),
            "channel_index": channel_index,
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        channel_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through LaBraM backbone and multitask readout.

        Args:
            input_values: EEG tensor of shape [B, C, T] or [B, T, C].
            task_index: Task indices of shape [B, n_out].
            channel_index: Optional collated ``[B, C]`` (or ``[C]``) indices
                into LABRAM_CHANNEL_ORDER from ``tokenize()``.
            **kwargs: Ignored (for compatibility with other models).

        Returns:
            Dictionary of task-specific outputs from ReadoutRouter.
        """
        if channel_index is not None:
            self._ch_names = labram_index_tensor_to_names(channel_index)
        if self._ch_names is None:
            raise RuntimeError(
                "Channel names not initialized. Pass channel_index from "
                "tokenize(), or call tokenize() in the training process."
            )

        if len(input_values.shape) == 3:
            if input_values.shape[-1] == self.num_channels:
                input_values = input_values.transpose(1, 2)

        features = self.backbone(
            input_values,
            ch_names=self._ch_names,
        )

        batch_size = features.shape[0]
        n_out = task_index.shape[1]

        x = features.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)

    def _route_readout(
        self, x: torch.Tensor, task_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Route embeddings through task-specific readout heads.

        Args:
            x: Features of shape [B, n_out, embed_dim].
            task_index: Task indices of shape [B, n_out].

        Returns:
            Dictionary of task outputs.
        """
        batch_size, n_out, dim = x.shape
        flat_embs = x.reshape(batch_size * n_out, dim)
        flat_task_index = task_index.reshape(batch_size * n_out)
        valid = flat_task_index > 0
        return self.router(
            flat_embs[valid], (flat_task_index[valid] - 1).long()
        )
