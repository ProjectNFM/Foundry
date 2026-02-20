"""Classic baseline models for EEG classification.

These models serve as reference implementations and benchmarks for evaluating the
performance of the foundation model. The provided architectures are intentionally
simple and are widely used on standard EEG classification tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_brain.data import chain, pad8
from torch_brain.nn import MultitaskReadout, prepare_for_multitask_readout
from torch_brain.registry import ModalitySpec
from temporaldata import Data
from typing import Dict, Optional

from foundry.models.utils import resolve_readout_specs


class BaselineModel(nn.Module):
    """
    Base class for all baseline EEG models.
    """
    def __init__(
        self,
        num_channels: int,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
    ):
        """
        Args:
            num_channels (int): Number of EEG channels.
            readout_specs (list[ModalitySpec | str] | dict[str, ModalitySpec]): Readout specification(s) for multitask head.
        """
        super().__init__()
        self.num_channels = num_channels
        self._readout_specs = resolve_readout_specs(readout_specs)

    @property
    def readout_specs(self) -> dict[str, ModalitySpec]:
        return self._readout_specs

    def _check_input_shape_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures input tensor has correct shape for Conv1d layer: (B, C, T).

        Args:
            x (torch.Tensor): Input of shape (B, C, T) or (B, T, C).

        Returns:
            torch.Tensor: Input tensor of shape (B, C, T).
        """
        if len(x.shape) == 3:
            # Convert (B, T, C) to (B, C, T) if needed.
            if x.shape[-1] == self.num_channels:
                x = x.transpose(1, 2)
        return x
    
    def _check_input_shape_conv2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures input tensor has correct shape for Conv2d layer: (B, 1, C, T).

        Args:
            x (torch.Tensor): Input of shape (B, C, T) or (B, T, C).

        Returns:
            torch.Tensor: Input tensor of shape (B, 1, C, T).
        """
        if len(x.shape) == 3:
            # Convert (B, T, C) to (B, C, T) if needed.
            if x.shape[-1] == self.num_channels:
                x = x.transpose(1, 2)
            # Add extra channel dimension
            x = x.unsqueeze(1)
        return x

    def tokenize(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Converts a TemporalData EEG sample to model-ready tensors and multitask readout targets.

        Args:
            data (temporaldata.Data): Input data structure containing fields such as "eeg", "channels", and "config".
                If data.config["multitask_readout"] is present, it is intersected with model-supported modalities.

        Returns:
            dict: {
                "x" (torch.Tensor): Model input of shape (T, C),
                "output_decoder_index" (torch.Tensor): Target output decoder indices,
                "target_values" (dict[str, torch.Tensor] or similar): Multitask target values,
                "target_weights" (dict[str, torch.Tensor] or similar): Multitask target weights,
                "session_id" (Any): Session identifier,
                "absolute_start" (Any): Absolute segment start time,
                "eval_mask" (dict[str, torch.Tensor] or similar): Mask for which outputs should be evaluated,
            }

        Note:
            The tokenized data will retain the same tensor dimensions and layout as present in `data`.
            Input shape normalization and conversion (e.g., unsqueezing or channel placement for Conv1d/Conv2d)
            is handled by the forward model methods, not at tokenization time.

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

        if not hasattr(data, "eeg") or data.eeg is None:
            raise ValueError("Data must have an 'eeg' field")

        signal = data.eeg.signal

        modality_field = (
            data.channels.types.astype(str)
            if hasattr(data.channels, "types")
            else np.array(["EEG"] * len(data.channels)).astype(str)
        )
        modality_mask = np.char.lower(modality_field) == "eeg"

        x = torch.from_numpy(signal[:, modality_mask]).float()

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

        return {
            "x": x,
            "output_decoder_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }


class SimpleEEGClassifier(BaselineModel):
    """
    A simple baseline classifier for EEG data.

    A minimal model consisting of a single temporal convolution, batch normalization, ReLU,
    global average pooling, and a multitask linear readout. Particularly useful
    for quick debugging and as a basic performance reference.
    """

    def __init__(
        self,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        num_channels: int = 64,
        num_filters: int = 32,
        kernel_size: int = 64,
    ):
        """
        Args:
            readout_specs (list[ModalitySpec | str] | dict[str, ModalitySpec]): Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_filters (int, optional): Number of convolutional filters. Default: 32.
            kernel_size (int, optional): Temporal kernel size for Conv1d. Default: 64.
        """
        super().__init__(num_channels, readout_specs)

        self.conv = nn.Conv1d(num_channels, num_filters, kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.readout = MultitaskReadout(
            dim=num_filters,
            readout_specs=self._readout_specs,
        )

    def forward(
        self,
        x: torch.Tensor,
        output_decoder_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the SimpleEEGClassifier.

        Args:
            x (torch.Tensor): EEG input tensor, shape (B, C, T) or (B, T, C).
            output_decoder_index (torch.Tensor): Task index tensor of shape (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._check_input_shape_conv1d(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        batch_size = x.shape[0]
        n_out = output_decoder_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)
        
        return self.readout(
            output_embs=x,
            output_readout_index=output_decoder_index,
            unpack_output=False,
        )


class ShallowConvNet(BaselineModel):
    """
    ShallowConvNet: A Shallow Deep Learning Architecture for EEG-based BCIs.

    This efficient network is a simpler alternative to DeepConvNet and is recommended
    for small datasets or when less capacity is appropriate.
    Reference: https://arxiv.org/abs/1703.05051
    """

    def __init__(
        self,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        num_channels: int = 64,
        num_samples: int = 128,
        dropout_rate: float = 0.5,
        kernel_length: int = 13,
        F1: int = 40,
    ):
        """
        Args:
            readout_specs (list[ModalitySpec | str] | dict[str, ModalitySpec]): Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_samples (int, optional): Number of samples (length of EEG input). Default: 128.
            dropout_rate (float, optional): Dropout rate after pooling. Default: 0.5.
            kernel_length (int, optional): Temporal convolution kernel length. Default: 13.
            F1 (int, optional): Number of spatial/temporal filters. Default: 40.
        """
        super().__init__(num_channels, readout_specs)

        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Spatial convolution
        self.conv2 = nn.Conv2d(F1, F1, (num_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F1)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, 35))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        out_dim = F1 * (num_samples // 35)
        self.readout = MultitaskReadout(
            dim=out_dim,
            readout_specs=self._readout_specs,
        )

    def forward(
        self,
        x: torch.Tensor,
        output_decoder_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ShallowConvNet.

        Args:
            x (torch.Tensor): EEG input tensor, shape (B, T, C).
            output_decoder_index (torch.Tensor): Task index tensor (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._check_input_shape_conv2d(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        
        batch_size = x.shape[0]
        n_out = output_decoder_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)
        
        return self.readout(
            output_embs=x,
            output_readout_index=output_decoder_index,
            unpack_output=False,
        )


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable 2D Convolution layer as used in EEGNet. 
    This is used to decouple the learning of temporal dynamics 
    from the optimal mixing of feature maps, significantly reducing 
    the number of parameters compared to a standard 2D convolution.

    This layer applies a depthwise (per feature map) 2D convolution,
    followed by a pointwise (1x1) convolution to mix feature maps,
    greatly reducing parameter count compared to standard convolutions.

    Args:
        in_channels (int): Number of input feature maps.
        out_channels (int): Number of output feature maps.
        kernel_size (tuple): Depthwise convolution kernel size.
        bias (bool, optional): Whether to add bias to conv layers. Default: False.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
    ):
        super().__init__()
        # Depthwise: Learns temporal summary features individually per spatial filter map
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, # CHECK: kernel_size
            padding="same", # CHECK: padding=(0, kernel_size // 2)
            groups=in_channels, 
            bias=bias,
        )
        # Pointwise: Optimally mixes the summary features across channels
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 1), 
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for depthwise separable 2D convolution.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_C, H, W).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNetEncoder(BaselineModel):
    """
    EEGNet: Compact Convolutional Neural Network for EEG-based BCIs.

    Reference: Lawhern et al., J. Neural Eng. 2018 (https://arxiv.org/abs/1611.08024).
    Designed to generalize across BCI tasks (P300, ERD/ERS, MRCP) while maintaining
    efficiency and strong performance.

    Args:
        readout_specs (list[ModalitySpec | str] | dict[str, ModalitySpec]): Readout specification(s).
        num_channels (int, optional): Number of EEG electrodes/channels. Default: 64.
        num_samples (int, optional): Number of samples in an EEG trial/window. Default: 128.
        F1 (int, optional): Temporal filter count ("bandpass" filters). Default: 8.
        D (int, optional): Depthwise spatial multiplier (# spatial filters per F1). Default: 2.
        F2 (int, optional): Pointwise filter count. Typically F1*D. Default: 16.
        kernel_length (int, optional): Temporal filter kernel length. Default: 64.
        dropout_rate (float, optional): Dropout probability. Default: 0.5.
    """
    def __init__(
        self,
        readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec],
        num_channels: int = 64,
        num_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__(num_channels, readout_specs)
        
        # ----------------------------------------------------------------------
        # Block 1: Bandpass & Spatial Filtering
        # ----------------------------------------------------------------------
        self.block1 = nn.Sequential(
            # 1. Temporal Convolution
            # Intuition: Learns frequency-specific bandpass filters (e.g., Alpha, Beta bands)
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, kernel_length),
                padding="same", # CHECK: padding=(0, kernel_length // 2)
                bias=False,
            ),
            nn.BatchNorm2d(F1), # CHECK: momentum=0.01, eps=1e-3
            
            # 2. Depthwise Spatial Convolution
            # Intuition: Acts as a data-driven spatial filter (similar to CSP).
            # Using groups=F1 ensures each temporal bandpass filter gets 'D' dedicated spatial filters.
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(num_channels, 1),
                groups=F1,
                bias=False,
                # CHECK: max_norm=0.25
            ),
            nn.BatchNorm2d(F1 * D), # CHECK: momentum=0.01, eps=1e-3
            nn.ELU(), # ELU performs better than ReLU for EEG signals
            
            # Downsample to reduce dimensionality and aggregate temporal information
            nn.AvgPool2d(kernel_size=(1, 4)), # CHECK: kernel_size
            nn.Dropout(dropout_rate)
        )
        
        # ----------------------------------------------------------------------
        # Block 2: Feature Mixing & Downsampling
        # ----------------------------------------------------------------------
        self.block2 = nn.Sequential(
            # Separable Convolution
            # Intuition: Efficiently summarizes temporal patterns within each feature map 
            # before mixing them to form final high-level representations.
            SeparableConv2d(
                F1 * D, 
                F2, 
                kernel_size=(1, 16), 
                bias=False,
            ),
            nn.BatchNorm2d(F2), # CHECK: momentum=0.01, eps=1e-3
            nn.ELU(), # CHECK: Is there ELU?
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        # ----------------------------------------------------------------------
        # Classifier Head
        # ----------------------------------------------------------------------
        out_dim = self._calculate_out_dim(num_channels, num_samples) # CHECK: out_dim == F2

        # MultitaskReadout replaces the final classifier
        self.readout = MultitaskReadout(
            dim=out_dim,
            readout_specs=self._readout_specs,
        )

    def _calculate_out_dim(self, channels, samples):
        """Dynamically calculate the flattened output dimension.

        Computes the output dimension after passing through block1 and block2 by
        using a dummy input tensor.
        
        Args:
            channels (int): Number of EEG channels (C).
            samples (int): Number of time samples (T).

        Returns:
            int: Number of flattened features to be fed into the readout.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, samples)
            x = self.block1(dummy_input)
            x = self.block2(x)
            return x.numel() 

    def extract_features(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts deep feature representation (before readout head).

        Useful for transfer/self-supervised learning, feature extraction, or clustering, etc.
        Accepts flexible input shapes and corrects them as needed.

        Args:
            x (torch.Tensor): EEG batch; (B, C, T), (B, T, C), or (B, 1, C, T).

        Returns:
            torch.Tensor: 4D feature tensor (B, F, H, W) prior to flattening.
        """
        x = self._check_input_shape_conv2d(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(
        self,
        x: torch.Tensor, 
        output_decoder_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EEGNetEncoder.

        Args:
            x (torch.Tensor): EEG batch, (B, C, T), (B, T, C), or (B, 1, C, T).
            output_decoder_index (torch.Tensor): Task index tensor (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self.extract_features(x)
        
        batch_size = x.shape[0]
        n_out = output_decoder_index.shape[1]
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)
        
        return self.readout(
            output_embs=x,
            output_readout_index=output_decoder_index,
            unpack_output=False,
        )
