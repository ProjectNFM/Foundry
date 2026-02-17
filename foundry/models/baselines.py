"""Classic baseline models for EEG classification.

These models serve as reference implementations and benchmarks for comparing
against the foundation model. They are simpler architectures that work well
on standard EEG classification tasks.
"""

import torch
import torch.nn as nn
from torch_brain.nn import MultitaskReadout, prepare_for_multitask_readout
from torch_brain.registry import ModalitySpec
from temporaldata import Data
from typing import Dict, Optional, list


class ShallowConvNet(nn.Module):
    """ShallowConvNet: A Shallow Deep Learning Architecture for EEG-based Brain-Computer Interfaces.
    
    A simpler alternative to DeepConvNet that performs well with limited training data.
    Reference: https://arxiv.org/abs/1703.05051
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        kernel_length: int = 13,
        F1: int = 40,
    ):
        """
        Args:
            num_channels: Number of EEG channels
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            kernel_length: Length of temporal convolution kernel
            F1: Number of spatial filters
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Spatial convolution
        self.conv2 = nn.Conv2d(F1, F1, (num_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F1)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, 35))
        self.dropout = nn.Dropout(dropout_rate)

        # Classification
        self.flatten = nn.Flatten()
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Add channel dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, time)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.flatten(x)

        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.num_classes).to(x.device)

        x = self.fc(x)
        return x


class SimpleEEGClassifier(nn.Module):
    """Simple baseline classifier for EEG data.
    
    A minimal model useful for testing and debugging, consisting of
    a single temporal convolution followed by global average pooling
    and a linear classifier.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        num_filters: int = 32,
        kernel_size: int = 64,
    ):
        """
        Args:
            num_channels: Number of EEG channels
            num_classes: Number of output classes
            num_filters: Number of convolutional filters
            kernel_size: Temporal kernel size
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.conv = nn.Conv1d(num_channels, num_filters, kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, time, channels)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Handle both (batch, channels, time) and (batch, time, channels)
        if x.shape[-1] == self.num_channels:
            x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution.
    
    In EEGNet, this is used to decouple the learning of temporal dynamics 
    from the optimal mixing of feature maps, significantly reducing the 
    number of parameters compared to a standard 2D convolution.

    Args:
        in_channels (int): Number of input channels/feature maps.
        out_channels (int): Number of output channels/feature maps.
        kernel_size (tuple): Size of the depthwise convolving kernel.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        # Depthwise: Learns temporal summary features individually per spatial filter map
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, padding="same", bias=bias
        )
        # Pointwise: Optimally mixes the summary features across channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=bias
        )

    def forward(self, x):
        """
        Forward pass for separable convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor after depthwise and pointwise convolutions.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNetEncoder(nn.Module):
    """
    EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces.
    Reference: Lawhern et al., J. Neural Eng. 2018. (https://arxiv.org/abs/1611.08024)

    This architecture is designed to generalize across various BCI paradigms 
    (P300, ERD/ERS, MRCP) while remaining highly parameter-efficient.

    Args:
        num_classes (int): Number of output classes for classification. Defaults to 4.
        num_channels (int): Number of EEG electrodes/channels. Defaults to 64.
        num_samples (int): Number of time samples in a single EEG trial/window. Defaults to 128.
        F1 (int): Number of temporal filters. Acts analogously to bandpass filters. Defaults to 8.
        D (int): Depth multiplier. Number of spatial filters learned per temporal filter. Defaults to 2.
        F2 (int): Number of pointwise filters in Block 2. Usually set to F1 * D. Defaults to 16.
        kernel_length (int): Length of the temporal convolution kernel. 
            Recommendation: Set to half the sampling rate (e.g., 64 for 128Hz). Defaults to 64.
        dropout_rate (float): Dropout probability. 
            Recommendation: 0.25 for within-subject, 0.5 for cross-subject. Defaults to 0.5.
    """
    def __init__(
        self,
        num_classes: int = 4,
        num_channels: int = 64,
        num_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        
        # ----------------------------------------------------------------------
        # Block 1: Bandpass & Spatial Filtering
        # ----------------------------------------------------------------------
        self.block1 = nn.Sequential(
            # 1. Temporal Convolution
            # Intuition: Learns frequency-specific bandpass filters (e.g., Alpha, Beta bands)
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            
            # 2. Depthwise Spatial Convolution
            # Intuition: Acts as a data-driven spatial filter (similar to CSP).
            # Using groups=F1 ensures each temporal bandpass filter gets 'D' dedicated spatial filters.
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(), # ELU performs better than ReLU for EEG signals
            
            # Downsample to reduce dimensionality and aggregate temporal information
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # ----------------------------------------------------------------------
        # Block 2: Feature Mixing & Downsampling
        # ----------------------------------------------------------------------
        self.block2 = nn.Sequential(
            # Separable Convolution
            # Intuition: Efficiently summarizes temporal patterns within each feature map 
            # before mixing them to form final high-level representations.
            SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        # ----------------------------------------------------------------------
        # Classifier Head
        # ----------------------------------------------------------------------
        out_dim = self._calculate_out_dim(num_channels, num_samples)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Note: The original paper applies a MaxNorm constraint of <= 0.25 here
            nn.Linear(out_dim, num_classes)
        )

    def _calculate_out_dim(self, channels, samples):
        """
        Dynamically calculates the number of features entering the classifier.
        
        Args:
            channels (int): Number of EEG channels.
            samples (int): Number of time samples.
            
        Returns:
            int: The flattened dimension size for the Linear layer.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, samples)
            x = self.block1(dummy_input)
            x = self.block2(x)
            return x.numel() 

    def extract_features(self, x):
        """
        Extracts deep embeddings from the EEG input without classifying them.
        Useful for self-supervised learning, transfer learning, or clustering.

        Args:
            x (torch.Tensor): EEG input tensor. Can be shape (B, C, T) or (B, 1, C, T).
            
        Returns:
            torch.Tensor: High-level feature maps before the flattening layer.
        """
        if len(x.shape) == 3:
            # Auto-unsqueeze to add the channel dimension required by Conv2d
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(self, x):
        """
        Standard forward pass for classification.

        Args:
            x (torch.Tensor): EEG input tensor of shape (B, 1, C, T) or (B, C, T).
            
        Returns:
            torch.Tensor: Logits for each class of shape (B, num_classes).
        """
        x = self.extract_features(x)
        x = self.classifier(x)
        return x