"""Classic baseline models for EEG classification.

These models serve as reference implementations and benchmarks for evaluating the
performance of the foundation model. The provided architectures are intentionally
simple and are widely used on standard EEG classification tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_brain.data import Data
from torch_brain.batching import chain, collate, pad8, pad2d, track_batch
from typing import Dict, Any

from foundry.models.readout import ReadoutRouter
from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import TargetExtractor


class BaselineEEGModel(nn.Module):
    """
    Base class for all reference EEG/iEEG models.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}

    def __init__(
        self,
        num_channels: int,
        task_configs: dict[str, TaskConfig],
        num_samples: int | None = None,
    ):
        """
        Args:
            num_channels (int): Number of EEG channels.
            task_configs: Mapping from task name to :class:`~foundry.tasks.config.TaskConfig`.
            num_samples (int | None, optional): Number of time samples per input window. Subclasses that require
                a fixed window length (e.g., for shape checks or flattened readout dimensions) should pass this value.
                Subclasses with adaptive pooling (e.g., TemporalConvAvgPool) may leave it as None. Default: None.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_samples = num_samples
        self._task_configs = task_configs

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    def _build_router(self, embed_dim: int) -> ReadoutRouter:
        heads = {
            name: instantiate({**cfg.head, "embed_dim": embed_dim})
            for name, cfg in self._task_configs.items()
        }
        return ReadoutRouter(heads)

    def _extract_targets(self, data: Data):
        all_timestamps = []
        task_indices = []
        target_values = {}
        target_weights = {}
        name_to_idx = {
            n: i for i, n in enumerate(sorted(self._task_configs.keys()))
        }

        for name in sorted(self._task_configs.keys()):
            cfg = self._task_configs[name]
            ext_kwargs = dict(cfg.target_extractor)
            ext_kwargs.pop("_target_", None)
            extractor = TargetExtractor(**ext_kwargs)

            targets = extractor(data)
            timestamps = targets["timestamps"]
            if timestamps is None or len(timestamps) == 0:
                continue

            idx = name_to_idx[name]
            all_timestamps.append(timestamps)
            task_indices.append(idx)
            target_values[name] = targets["values"]
            target_weights[name] = np.ones_like(timestamps, dtype=np.float32)

        if not all_timestamps:
            raise ValueError(
                "No targets extracted from data for configured tasks"
            )

        if len(all_timestamps) == 1:
            output_task_index = torch.full(
                (len(all_timestamps[0]),),
                task_indices[0] + 1,
                dtype=torch.long,
            )
        else:
            _, batch = collate(
                [
                    (chain(all_timestamps[i]), track_batch(all_timestamps[i]))
                    for i in range(len(all_timestamps))
                ]
            )
            output_task_index = torch.tensor(task_indices)[batch] + 1

        return target_values, output_task_index, target_weights

    def _route_readout(
        self, x: torch.Tensor, task_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_out, dim = x.shape
        flat_embs = x.reshape(batch_size * n_out, dim)
        flat_task_index = task_index.reshape(batch_size * n_out)
        valid = flat_task_index > 0
        return self.router(
            flat_embs[valid], (flat_task_index[valid] - 1).long()
        )

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input tensor to (B, C, T) format.

        Converts (B, T, C) to (B, C, T) if the last dimension matches num_channels.
        This is a generic shape normalization utility for all baseline models.

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

    def _check_input_shape_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures input tensor has correct shape for Conv1d layer: (B, C, T).

        Args:
            x (torch.Tensor): Input of shape (B, C, T) or (B, T, C).

        Returns:
            torch.Tensor: Input tensor of shape (B, C, T).
        """
        return self._normalize_input_shape(x)

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
        Converts a TemporalData EEG/ECoG sample to model-ready tensors and multitask readout targets.

        Args:
            data (torch_brain.data.Data): Input data structure containing an "eeg" or "ecog" field
                along with "channels" and task-specific label fields.

        Returns:
            dict: {
                "input_values" (torch.Tensor): Model input of shape (T, C),
                "task_index" (torch.Tensor): Target output decoder indices,
                "target_values" (dict[str, torch.Tensor] or similar): Multitask target values,
                "target_weights" (dict[str, torch.Tensor] or similar): Multitask target weights,
                "session_id" (Any): Session identifier,
                "absolute_start" (Any): Absolute segment start time,
            }

        Note:
            The tokenized data will retain the same tensor dimensions and layout as present in `data`.
            Input shape normalization and conversion (e.g., unsqueezing or channel placement for Conv1d/Conv2d)
            is handled by the forward model methods, not at tokenization time.

        """
        has_eeg = hasattr(data, "eeg") and data.eeg is not None
        has_ecog = hasattr(data, "ecog") and data.ecog is not None
        has_seeg = hasattr(data, "seeg") and data.seeg is not None

        if not has_eeg and not has_ecog and not has_seeg:
            raise ValueError(
                "Data must have an 'eeg', 'ecog', or 'seeg' channel type"
            )

        if has_eeg:
            signal = data.eeg.signal
            default_type = "EEG"
        elif has_ecog:
            signal = data.ecog.signal
            default_type = "ECOG"
        elif has_seeg:
            signal = data.seeg.signal
            default_type = "SEEG"
        else:
            raise ValueError(f"Data must have an '{default_type}' channel type")

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )

        signal = np.asarray(signal, dtype=np.float32)
        x = torch.from_numpy(signal[:, modality_mask])

        output_values, output_task_index, output_weights = (
            self._extract_targets(data)
        )

        return {
            "input_values": pad2d(x),
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

    def unpack_batch(self, batch: Dict[str, Any]) -> tuple:
        """Extract model inputs and targets from batch.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Tuple of (model_inputs, target_values, target_weights, task_index)
        """
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)
        task_index = batch["task_index"]

        model_inputs = {
            "input_values": batch["input_values"],
            "task_index": task_index,
        }
        return model_inputs, target_values, target_weights, task_index


class Linear(BaselineEEGModel):
    """
    A simple linear baseline for EEG classification.

    This minimal model flattens the full EEG input across channels and time
    and then passes those features directly to a multitask linear readout.
    Useful as a sanity check and lower-bound reference for model performance.
    """

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_samples: int = 128,
    ):
        """
        Args:
            task_configs: Mapping from task name to TaskConfig. Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_samples (int, optional): Number of time samples per input window. Default: 128.
        """
        super().__init__(
            num_channels=num_channels,
            num_samples=num_samples,
            task_configs=task_configs,
        )

        out_dim = num_channels * num_samples
        self.router = self._build_router(out_dim)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the Linear model.

        Args:
            input_values (torch.Tensor): EEG input tensor, shape (B, C, T) or (B, T, C).
            task_index (torch.Tensor): Task index tensor of shape (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._normalize_input_shape(input_values)
        if x.shape[-1] != self.num_samples:
            raise ValueError(
                f"Expected input with {self.num_samples} time samples, got {x.shape[-1]}"
            )
        # Flattens each input in the batch into a single feature vector (combining all channels and time samples)
        x = x.reshape(x.size(0), -1)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]

        # Repeat (broadcast) the feature vector for each output task in the batch,
        # so its shape is (batch_size, n_out, feature_dim) for per-output routing
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)


class MLP(BaselineEEGModel):
    """
    An MLP-based baseline for EEG classification.

    This model flattens the full EEG input across channels and time, then passes
    those features through a configurable MLP (multi-layer perceptron) before the
    final multitask linear readout. Useful as an intermediate-complexity
    reference between Linear and convolutional baselines.
    """

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_samples: int = 128,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            task_configs: Mapping from task name to TaskConfig. Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_samples (int, optional): Number of time samples per input window. Default: 128.
            hidden_dims (list[int], optional): Hidden layer dimensions. Default: [128, 64].
            dropout_rate (float, optional): Dropout rate after each hidden layer. Default: 0.5.
        """
        super().__init__(
            num_channels=num_channels,
            num_samples=num_samples,
            task_configs=task_configs,
        )

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = num_channels * num_samples
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        out_dim = in_dim

        self.router = self._build_router(out_dim)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the MLP model.

        Args:
            input_values (torch.Tensor): EEG input tensor, shape (B, C, T) or (B, T, C).
            task_index (torch.Tensor): Task index tensor of shape (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._normalize_input_shape(input_values)
        if x.shape[-1] != self.num_samples:
            raise ValueError(
                f"Expected input with {self.num_samples} time samples, got {x.shape[-1]}"
            )
        x = x.reshape(x.size(0), -1)

        x = self.mlp(x)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)


class GRU(BaselineEEGModel):
    """
    A GRU-based baseline for EEG classification.

    This model projects per-timestep channel values into a latent feature space,
    processes the sequence with a (bi)directional GRU, and applies global temporal
    averaging before a multitask readout.
    """

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_samples: int = 128,
        input_proj_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            task_configs: Mapping from task name to TaskConfig. Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_samples (int, optional): Number of time samples per input window. Default: 128.
            input_proj_dim (int, optional): Per-timestep channel projection dimension. Default: 128.
            hidden_size (int, optional): GRU hidden size per direction. Default: 128.
            num_layers (int, optional): Number of stacked GRU layers. Default: 2.
            bidirectional (bool, optional): Whether to use bidirectional GRU. Default: True.
            dropout_rate (float, optional): Dropout rate between stacked GRU layers. Default: 0.3.
        """
        super().__init__(
            num_channels=num_channels,
            num_samples=num_samples,
            task_configs=task_configs,
        )

        self.input_norm = nn.LayerNorm(num_channels)
        self.input_proj = nn.Linear(num_channels, input_proj_dim)
        self.gru = nn.GRU(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.router = self._build_router(out_dim)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for GRU.

        Args:
            input_values (torch.Tensor): EEG input tensor, shape (B, C, T) or (B, T, C).
            task_index (torch.Tensor): Task index tensor of shape (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._normalize_input_shape(input_values)
        if x.shape[-1] != self.num_samples:
            raise ValueError(
                f"Expected input with {self.num_samples} time samples, got {x.shape[-1]}"
            )

        # Convert from (B, C, T) to (B, T, C) for sequence modeling over time.
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.input_proj(x)
        x, _ = self.gru(x)

        # Global average pooling across the temporal dimension.
        x = x.mean(dim=1)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)


class TemporalConvAvgPool(BaselineEEGModel):
    """
    A simple baseline classifier for EEG data.

    A minimal model consisting of a single temporal convolution, batch normalization, ReLU,
    global average pooling, and a multitask linear readout. Particularly useful
    for quick debugging and as a basic performance reference.
    """

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_filters: int = 32,
        kernel_size: int = 64,
        **kwargs,
    ):
        """
        Args:
            task_configs: Mapping from task name to TaskConfig. Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_filters (int, optional): Number of convolutional filters. Default: 32.
            kernel_size (int, optional): Temporal kernel size for Conv1d. Default: 64.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_channels=num_channels,
            task_configs=task_configs,
        )

        self.conv = nn.Conv1d(
            num_channels, num_filters, kernel_size, padding="same"
        )
        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.router = self._build_router(num_filters)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the SimpleEEGClassifier.

        Args:
            input_values (torch.Tensor): EEG input tensor, shape (B, C, T) or (B, T, C).
            task_index (torch.Tensor): Task index tensor of shape (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._check_input_shape_conv1d(input_values)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)


class ShallowConvNet(BaselineEEGModel):
    """
    ShallowConvNet: A Shallow Deep Learning Architecture for EEG-based BCIs.

    This efficient network is a simpler alternative to DeepConvNet and is recommended
    for small datasets or when less capacity is appropriate.
    Reference: https://arxiv.org/abs/1703.05051
    """

    def __init__(
        self,
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_samples: int = 128,
        dropout_rate: float = 0.5,
        kernel_length: int = 13,
        F1: int = 40,
    ):
        """
        Args:
            task_configs: Mapping from task name to TaskConfig. Readout specification(s).
            num_channels (int, optional): Number of EEG channels. Default: 64.
            num_samples (int, optional): Number of samples (length of EEG input). Default: 128.
            dropout_rate (float, optional): Dropout rate after pooling. Default: 0.5.
            kernel_length (int, optional): Temporal convolution kernel length. Default: 13.
            F1 (int, optional): Number of spatial/temporal filters. Default: 40.
        """
        super().__init__(
            num_channels=num_channels,
            num_samples=num_samples,
            task_configs=task_configs,
        )

        # Temporal convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Spatial convolution
        self.conv2 = nn.Conv2d(F1, F1, (num_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F1)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, 35))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        out_dim = F1 * (num_samples // 35)
        self.router = self._build_router(out_dim)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ShallowConvNet.

        Args:
            input_values (torch.Tensor): EEG input tensor, shape (B, T, C).
            task_index (torch.Tensor): Task index tensor (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self._check_input_shape_conv2d(input_values)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)


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
            kernel_size=kernel_size,  # CHECK: kernel_size
            padding="same",  # CHECK: padding=(0, kernel_size // 2)
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
        input_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for depthwise separable 2D convolution.

        Args:
            input_values (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_C, H, W).
        """
        x = self.depthwise(input_values)
        x = self.pointwise(x)
        return x


class EEGNetEncoder(BaselineEEGModel):
    """
    EEGNet: Compact Convolutional Neural Network for EEG-based BCIs.

    Reference: Lawhern et al., J. Neural Eng. 2018 (https://arxiv.org/abs/1611.08024).
    Designed to generalize across BCI tasks (ERD/ERS, MRCP) while maintaining
    efficiency and strong performance.

    Args:
        task_configs: Mapping from task name to TaskConfig.
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
        task_configs: dict[str, TaskConfig],
        num_channels: int = 64,
        num_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__(
            num_channels=num_channels,
            num_samples=num_samples,
            task_configs=task_configs,
        )

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
                padding="same",  # CHECK: padding=(0, kernel_length // 2)
                bias=False,
            ),
            nn.BatchNorm2d(F1),  # CHECK: momentum=0.01, eps=1e-3
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
            nn.BatchNorm2d(F1 * D),  # CHECK: momentum=0.01, eps=1e-3
            nn.ELU(),  # ELU performs better than ReLU for EEG signals
            # Downsample to reduce dimensionality and aggregate temporal information
            nn.AvgPool2d(kernel_size=(1, 4)),  # CHECK: kernel_size
            nn.Dropout(dropout_rate),
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
            nn.BatchNorm2d(F2),  # CHECK: momentum=0.01, eps=1e-3
            nn.ELU(),  # CHECK: Is there ELU?
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # ----------------------------------------------------------------------
        # Classifier Head
        # ----------------------------------------------------------------------
        out_dim = self._calculate_out_dim(
            num_channels, num_samples
        )  # CHECK: out_dim == F2

        self.router = self._build_router(out_dim)

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
        self,
        input_values: torch.Tensor,
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
        x = self._check_input_shape_conv2d(input_values)
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EEGNetEncoder.

        Args:
            input_values (torch.Tensor): EEG batch, (B, C, T), (B, T, C), or (B, 1, C, T).
            task_index (torch.Tensor): Task index tensor (B, n_out).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multitask readout outputs.
        """
        x = self.extract_features(input_values)

        batch_size = x.shape[0]
        n_out = task_index.shape[1]
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)
