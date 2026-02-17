from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from temporaldata import Data
from torch_brain.data import chain, pad8, pad2d
from torch_brain.nn import MultitaskReadout, prepare_for_multitask_readout
from torch_brain.registry import ModalitySpec


def z_score_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-score normalize tensor along the time dimension (last dim).

    Args:
        x: Tensor of shape (..., T) where T is the time dimension

    Returns:
        Normalized tensor with zero mean and unit variance per channel
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with MaxNorm constraint on the weights, similar to Keras' MaxNorm.

    Intended use: spatial depthwise conv in EEGNet.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w = self.weight
            norms = w.norm(p=2, dim=(1, 2, 3), keepdim=True)
            desired = torch.clamp(norms, max=self.max_norm)
            self.weight.copy_(w * (desired / (1e-8 + norms)))
        return super().forward(x)


class EEGNetModel(nn.Module):
    """
    EEGNet v2 model (Lawhern et al., 2018) adapted for Foundry.

    Uses MultitaskReadout for multi-task learning compatibility.
    """

    def __init__(
        self,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        in_chans: int,
        in_times: int,
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        kernel_time: int = 64,
        kernel_time_separable: int = 16,
        pool_time_size_1: int = 4,
        pool_time_size_2: int = 8,
        dropout: float = 0.25,
        spatial_max_norm: float = 1.0,
        normalize_mode: str = "zscore",
    ):
        """
        Args:
            readout_specs: List/dict of task specifications for multitask readout.
                Can be ModalitySpec objects or string names that resolve from registry.
            in_chans: Number of EEG channels
            in_times: Number of time samples per window
            F1: Number of temporal filters in the first block
            D: Depth multiplier for spatial depthwise conv (F2 = F1 * D if F2 is None)
            F2: Number of feature maps after depthwise-separable conv. If None, F2 = F1 * D
            kernel_time: Temporal kernel size for the first convolution
            kernel_time_separable: Temporal kernel size for the depthwise separable conv
            pool_time_size_1: Temporal pooling size for the first pooling layer
            pool_time_size_2: Temporal pooling size for the second pooling layer
            dropout: Dropout probability
            spatial_max_norm: MaxNorm constraint for the spatial depthwise conv
            normalize_mode: How to normalize EEG channels over time in tokenize().
                "zscore": Z-score normalization
                "none": No normalization
        """
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        if normalize_mode not in {"zscore", "none"}:
            raise ValueError(
                f"normalize_mode must be 'zscore' or 'none', got '{normalize_mode}'"
            )

        self.readout_specs = self._resolve_readout_specs(readout_specs)
        self.F2 = F2
        self.in_chans = in_chans
        self.in_times = in_times
        self.normalize_mode = normalize_mode

        self.kernel_time = int(kernel_time)
        self.kernel_time_separable = int(kernel_time_separable)
        self.pool_time_size_1 = int(pool_time_size_1)
        self.pool_time_size_2 = int(pool_time_size_2)

        self.min_T = self.pool_time_size_1 * self.pool_time_size_2
        self.in_times = max(in_times, self.min_T)

        # Block 1: Temporal Convolution
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, self.kernel_time),
            padding=(0, self.kernel_time // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)

        # Block 1: Depthwise Spatial Convolution
        self.conv_spatial = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(in_chans, 1),
            groups=F1,
            bias=False,
            max_norm=spatial_max_norm,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu = nn.ELU()
        self.pool1_layer = nn.AvgPool2d(kernel_size=(1, self.pool_time_size_1))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Depthwise-Separable Convolution
        self.conv_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, self.kernel_time_separable),
            groups=F1 * D,
            padding=(0, self.kernel_time_separable // 2),
            bias=False,
        )
        self.conv_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool2_layer = nn.AvgPool2d(kernel_size=(1, self.pool_time_size_2))
        self.drop2 = nn.Dropout(dropout)

        # MultitaskReadout replaces the final classifier
        self.readout = MultitaskReadout(
            dim=F2,
            readout_specs=self.readout_specs,
        )

        self._init_keras_style()

    def initialize_vocabs(self, vocab_info: dict):
        """EEGNet doesn't use vocabularies, so this is a no-op.

        Args:
            vocab_info: Dictionary with vocabulary information (unused)
        """
        pass

    def _init_keras_style(self):
        """Initialize weights using Keras-style initialization (glorot_uniform + zero biases)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EEGNet conv blocks.

        Args:
            x: Input tensor of shape (B, 1, C, T)

        Returns:
            Feature tensor of shape (B, F2, 1, 1)
        """
        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1_layer(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2_layer(x)
        x = self.drop2(x)

        return x

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        output_decoder_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_values: EEG input tensor of shape (B, C, T)
            output_decoder_index: Task/decoder indices for each output (B, n_out)
            **kwargs: Ignored additional arguments for compatibility

        Returns:
            Dictionary of task-specific outputs
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"EEGNet forward expects input_values of shape [B, C, T], got {tuple(input_values.shape)}"
            )
        if input_values.shape[1] != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} channels, got {input_values.shape[1]}"
            )

        x = input_values.unsqueeze(1)  # (B, 1, C, T)
        x = self._forward_features(x)  # (B, F2, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, F2)
        x = x.unsqueeze(1)  # (B, 1, F2) - add sequence dim for MultitaskReadout

        outputs = self.readout(
            output_embs=x,
            output_readout_index=output_decoder_index,
            unpack_output=False,
        )

        return outputs

    def tokenize(self, data: Data) -> dict:
        """
        Tokenize raw EEG data for EEGNet.

        Args:
            data: TemporalData object containing raw EEG signal (T, C).
                  Should NOT be patched (no Patching transform).

        Returns:
            dict with model_inputs, target_values, target_weights, and metadata
        """
        # TODO: this should probably be separated out so that models can reuse it.
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

        if not hasattr(data, "eeg") or data.eeg is None:
            raise ValueError("Data must have an 'eeg' field")

        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("Sample missing EEG at data.eeg.signal")

        x = np.asarray(sig, dtype=np.float32)  # (T, C)
        if x.ndim != 2:
            raise ValueError(
                f"Expected EEG shape [T, C], got {x.shape}. "
                "EEGNet expects raw (unpatched) data."
            )

        # Convert to (C, T)
        x = x.T
        x_t = torch.from_numpy(x)  # (C, T)

        # Normalization
        if self.normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self.normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self.normalize_mode}")

        # Check min time requirement + pad if needed
        C, T = x_t.shape
        if T < self.min_T:
            pad_t = self.min_T - T
            x_t = F.pad(x_t, (0, pad_t))  # pad time axis
            T_out = self.min_T
        else:
            T_out = T

        # Prepare multitask readout targets
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

        tokenized_data = {
            "input_values": pad2d(x_t),  # (C, T) -> (B, C, T) after collate
            "output_decoder_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id if hasattr(data, "session") else None,
            "absolute_start": getattr(data, "absolute_start", None),
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }

        return tokenized_data

    def unpack_batch(self, batch: Dict[str, Any]) -> tuple:
        """Extract model inputs and targets from batch.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Tuple of (model_inputs, target_values, target_weights, output_decoder_index)
        """
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)
        output_decoder_index = batch["output_decoder_index"]

        model_inputs = {
            "input_values": batch["input_values"],
            "output_decoder_index": output_decoder_index,
        }
        return model_inputs, target_values, target_weights, output_decoder_index

    def _resolve_readout_specs(
        self, readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec]
    ) -> dict[str, ModalitySpec]:
        """Resolve string modality names to ModalitySpec objects.

        Args:
            readout_specs: List or dict of ModalitySpec objects or string modality names

        Returns:
            Dictionary mapping modality names to ModalitySpec objects
        """
        from torch_brain.registry import MODALITY_REGISTRY

        if isinstance(readout_specs, dict):
            return readout_specs

        resolved = {}
        for spec in readout_specs:
            if isinstance(spec, str):
                if spec not in MODALITY_REGISTRY:
                    raise ValueError(
                        f"Unknown modality '{spec}' in registry. "
                        f"Available: {list(MODALITY_REGISTRY.keys())}"
                    )
                resolved[spec] = MODALITY_REGISTRY[spec]
            else:
                for name, registry_spec in MODALITY_REGISTRY.items():
                    if registry_spec.id == spec.id:
                        resolved[name] = spec
                        break
                else:
                    raise ValueError(
                        f"ModalitySpec with id {spec.id} not found in registry"
                    )
        return resolved