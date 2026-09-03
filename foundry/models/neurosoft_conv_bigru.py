"""Session-adapted convolutional BiGRU for NeuroSoft acoustic decoding.

This module deliberately does not inherit from the historical baseline models:
the per-session input adapter is the only session-specific component and the
rest of the encoder has an explicit checkpoint-transfer boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_brain.batching import chain, pad2d, pad8
from torch_brain.data import Data

from foundry.models.readout import ReadoutRouter, build_readout_router
from foundry.tasks.config import TaskConfig
from foundry.data.utils import resolve_neural_signal
from foundry.tasks.targets import extract_multitask_targets


class SessionInputAdapter(nn.Module):
    """Map each recording's physical channels into a shared feature space."""

    def __init__(self, session_configs: Mapping[str, int], adapter_dim: int):
        super().__init__()
        if not session_configs:
            raise ValueError(
                "session_configs must contain at least one session"
            )
        self.channel_counts = {
            str(key): int(value) for key, value in session_configs.items()
        }
        if any(count <= 0 for count in self.channel_counts.values()):
            raise ValueError("Every session channel count must be positive")
        self.adapter_dim = adapter_dim
        self.layers = nn.ModuleDict(
            {
                session_id: nn.Linear(channel_count, adapter_dim, bias=True)
                for session_id, channel_count in self.channel_counts.items()
            }
        )

    @staticmethod
    def _as_session_id(session_id: object) -> str:
        if torch.is_tensor(session_id):
            session_id = session_id.item()
        return str(session_id)

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_session_ids: Sequence[object],
        input_channel_counts: torch.Tensor | Sequence[int],
        input_seq_len: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        """Adapt a padded ``(B, C_pad, T_pad)`` batch to ``(B, D, T_pad)``."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected input_values with shape (B, C, T), got {tuple(x.shape)}"
            )
        batch_size, padded_channels, padded_time = x.shape
        if len(input_session_ids) != batch_size:
            raise ValueError(
                "input_session_ids must have one ID per batch item"
            )
        if (
            len(input_channel_counts) != batch_size
            or len(input_seq_len) != batch_size
        ):
            raise ValueError(
                "input_channel_counts and input_seq_len must have one value per batch item"
            )

        out = x.new_zeros(batch_size, self.adapter_dim, padded_time)
        for item in range(batch_size):
            session_id = self._as_session_id(input_session_ids[item])
            if session_id not in self.layers:
                known = ", ".join(self.layers.keys())
                raise KeyError(
                    f"Unknown NeuroSoft session ID {session_id!r}; configured IDs: {known}"
                )
            channel_count = int(input_channel_counts[item])
            seq_len = int(input_seq_len[item])
            expected_channels = self.channel_counts[session_id]
            if channel_count != expected_channels:
                raise ValueError(
                    f"Session {session_id!r} is configured for {expected_channels} channels, "
                    f"but batch item declares {channel_count}"
                )
            if not 0 < channel_count <= padded_channels:
                raise ValueError(
                    f"Invalid channel count {channel_count} for padded width {padded_channels}"
                )
            if not 0 < seq_len <= padded_time:
                raise ValueError(
                    f"Invalid sequence length {seq_len} for padded length {padded_time}"
                )

            # The slice is mandatory: padded channels must never reach a
            # session's linear layer. Re-zero time padding because Linear's
            # bias would otherwise turn it into non-zero signal.
            adapted = self.layers[session_id](
                x[item, :channel_count].transpose(0, 1)
            ).transpose(0, 1)
            adapted[:, seq_len:] = 0
            out[item] = adapted
        return out


class _SeparableTemporalBlock(nn.Module):
    """LayerNorm, depthwise temporal convolution, and pointwise mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.output_norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        return (
            torch.div(
                input_length + 2 * self.padding - self.kernel_size,
                self.stride,
                rounding_mode="floor",
            )
            + 1
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the block while preserving zero-valued time padding.

        ``LayerNorm`` and convolution biases can otherwise turn padded frames
        into signal that leaks into valid right-edge convolution windows.  The
        mask is therefore applied both after input normalization and after the
        complete block, so stacked blocks are safe as well.
        """
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        output_lengths = None
        if input_lengths is not None:
            input_mask = torch.arange(x.shape[-1], device=x.device).unsqueeze(
                0
            ) < input_lengths.unsqueeze(1)
            x = x * input_mask.unsqueeze(1).to(x.dtype)
            output_lengths = self.output_length(input_lengths)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.output_norm(x.transpose(1, 2))
        x = self.dropout(self.activation(x)).transpose(1, 2)
        if output_lengths is not None:
            output_mask = torch.arange(x.shape[-1], device=x.device).unsqueeze(
                0
            ) < output_lengths.unsqueeze(1)
            x = x * output_mask.unsqueeze(1).to(x.dtype)
        return x


class NeurosoftConvBiGRU(nn.Module):
    """NeuroSoft's fixed-recipe session-adapted convolutional BiGRU.

    ``session_adapter`` is intentionally separate from the transferable
    frontend, recurrent encoder, and readout router.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}
    _TRANSFERABLE_COMPONENTS = ("temporal_frontend", "gru", "router")
    _FROZEN_REPRESENTATION_COMPONENTS = ("temporal_frontend", "gru")

    def __init__(
        self,
        *,
        task_configs: dict[str, TaskConfig],
        session_configs: Mapping[str, int],
        num_samples: int = 1000,
        adapter_dim: int = 64,
        temporal_channels: int = 128,
        temporal_kernel_samples: int = 64,
        temporal_stride: int = 4,
        conv_depth: int = 1,
        dropout_rate: float = 0.3,
        adapter_initializer: str = "pytorch_default",
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.0,
        id_aliases: dict[str, dict[str, str]] | None = None,
    ):
        super().__init__()
        if num_samples <= 0 or adapter_dim <= 0 or temporal_channels <= 0:
            raise ValueError(
                "num_samples, adapter_dim, and temporal_channels must be positive"
            )
        if (
            temporal_kernel_samples <= 0
            or temporal_stride <= 0
            or conv_depth <= 0
        ):
            raise ValueError(
                "temporal kernel, stride, and conv_depth must be positive"
            )
        if gru_hidden_size <= 0 or gru_num_layers <= 0:
            raise ValueError("GRU hidden size and layer count must be positive")
        if not gru_bidirectional:
            raise ValueError(
                "NeurosoftConvBiGRU is an offline bidirectional model"
            )
        if adapter_initializer != "pytorch_default":
            raise ValueError(
                "adapter_initializer currently supports only 'pytorch_default'"
            )

        self.num_samples = num_samples
        self.adapter_dim = adapter_dim
        self.temporal_channels = temporal_channels
        self.conv_depth = conv_depth
        # nn.Linear's PyTorch-default initializer is deterministic under the
        # run seed. Keep its name on the module so it is recorded by normal
        # hyperparameter/checkpoint serialization.
        self.adapter_initializer = adapter_initializer
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gru_bidirectional = gru_bidirectional
        self._task_configs = TaskConfig.normalize_task_configs(task_configs)
        if id_aliases is None:
            self._id_aliases = None
            self._raw_to_canonical = None
        else:
            self._id_aliases = id_aliases
            raw_to_canonical: dict[str, str] = {}
            ambiguous: set[str] = set()
            for mapping in id_aliases.values():
                for raw_id, canonical_id in mapping.items():
                    if raw_id in ambiguous:
                        continue
                    if raw_id in raw_to_canonical:
                        del raw_to_canonical[raw_id]
                        ambiguous.add(raw_id)
                    else:
                        raw_to_canonical[raw_id] = canonical_id
            self._raw_to_canonical = raw_to_canonical
        self.session_adapter = SessionInputAdapter(session_configs, adapter_dim)

        # The first block is the declared 64-sample/stride-4 recipe. Extra
        # depth uses explicitly length-preserving 3-sample blocks, making its
        # scale an opt-in architectural intervention rather than a per-session
        # adjustment.
        first_padding = (temporal_kernel_samples - temporal_stride) // 2
        blocks: list[nn.Module] = [
            _SeparableTemporalBlock(
                adapter_dim,
                temporal_channels,
                kernel_size=temporal_kernel_samples,
                stride=temporal_stride,
                padding=first_padding,
                dropout_rate=dropout_rate,
            )
        ]
        blocks.extend(
            _SeparableTemporalBlock(
                temporal_channels,
                temporal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dropout_rate=dropout_rate,
            )
            for _ in range(conv_depth - 1)
        )
        self.temporal_frontend = nn.ModuleList(blocks)
        self.gru = nn.GRU(
            input_size=temporal_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.embedding_dim = gru_hidden_size * 2
        self.readout_dropout = nn.Dropout(dropout_rate)
        self.router: ReadoutRouter = build_readout_router(
            self._task_configs, self.embedding_dim
        )

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    @property
    def configured_session_ids(self) -> set[str]:
        return set(self.session_adapter.layers.keys())

    def resolve_session_id(
        self, raw_id: str, namespace: str | None = None
    ) -> str:
        if self._id_aliases is None:
            return raw_id
        if namespace is not None:
            try:
                return self._id_aliases[namespace][raw_id]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown session ID {raw_id!r} for namespace {namespace!r}"
                ) from exc
        if (
            self._raw_to_canonical is not None
            and raw_id in self._raw_to_canonical
        ):
            return self._raw_to_canonical[raw_id]
        namespaces_with_id = [
            ns
            for ns, mapping in self._id_aliases.items()
            if raw_id in mapping
        ]
        if len(namespaces_with_id) > 1:
            raise KeyError(
                f"Session ID {raw_id!r} is ambiguous across namespaces "
                f"{namespaces_with_id}; provide dataset_namespace"
            )
        raise KeyError(f"Unknown session ID {raw_id!r}")

    def transferable_components(self) -> tuple[str, ...]:
        return self._TRANSFERABLE_COMPONENTS

    def transferable_components_for_mode(self, mode: str) -> tuple[str, ...]:
        """Declare the selected components for a documented transfer regime."""
        if mode == "full_finetuning":
            return self._TRANSFERABLE_COMPONENTS
        if mode == "frozen_representation":
            return self._FROZEN_REPRESENTATION_COMPONENTS
        raise ValueError(
            "mode must be 'full_finetuning' or 'frozen_representation'"
        )

    @staticmethod
    def _normalise_lengths(
        lengths: torch.Tensor | Sequence[int] | None,
        batch_size: int,
        default: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full(
                (batch_size,), default, dtype=torch.long, device=device
            )
        result = torch.as_tensor(lengths, dtype=torch.long, device=device)
        if result.ndim == 0:
            result = result.repeat(batch_size)
        if result.shape != (batch_size,):
            raise ValueError(
                f"{name} must have shape ({batch_size},), got {tuple(result.shape)}"
            )
        return result

    def _frontend_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for block in self.temporal_frontend:
            lengths = block.output_length(lengths)
        return lengths

    def encode(
        self,
        *,
        input_values: torch.Tensor,
        input_session_ids: Sequence[object],
        input_channel_counts: torch.Tensor | Sequence[int] | None = None,
        input_seq_len: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return one masked-mean pooled embedding per window."""
        if input_values.ndim != 3:
            raise ValueError("input_values must be a (B, C_pad, T_pad) tensor")
        batch_size, padded_channels, padded_time = input_values.shape
        if isinstance(input_session_ids, (str, bytes)):
            input_session_ids = [input_session_ids]
        session_ids = [
            self.session_adapter._as_session_id(session_id)
            for session_id in input_session_ids
        ]
        if len(session_ids) != batch_size:
            raise ValueError(
                "input_session_ids must have one ID per batch item"
            )
        unknown_ids = sorted(
            {
                session_id
                for session_id in session_ids
                if session_id not in self.session_adapter.layers
            }
        )
        if unknown_ids:
            known = ", ".join(self.session_adapter.layers.keys())
            raise KeyError(
                f"Unknown NeuroSoft session ID(s) {unknown_ids}; configured IDs: {known}"
            )
        if input_channel_counts is None:
            input_channel_counts = [
                self.session_adapter.channel_counts[session_id]
                for session_id in session_ids
            ]
        seq_lens = self._normalise_lengths(
            input_seq_len,
            batch_size,
            padded_time,
            input_values.device,
            "input_seq_len",
        )
        channel_counts = self._normalise_lengths(
            input_channel_counts,
            batch_size,
            padded_channels,
            input_values.device,
            "input_channel_counts",
        )

        x = self.session_adapter(
            input_values,
            input_session_ids=session_ids,
            input_channel_counts=channel_counts,
            input_seq_len=seq_lens,
        )
        feature_lengths = seq_lens
        for block in self.temporal_frontend:
            next_feature_lengths = block.output_length(feature_lengths)
            if torch.any(next_feature_lengths <= 0):
                raise ValueError(
                    "Input sequence is too short for the temporal convolution"
                )
            x = block(x, input_lengths=feature_lengths)
            feature_lengths = next_feature_lengths
        x = x.transpose(1, 2)
        packed = pack_padded_sequence(
            x, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        x, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.shape[1]
        )
        time_index = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        mask = time_index < feature_lengths.unsqueeze(1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / feature_lengths.unsqueeze(
            1
        ).to(x.dtype)

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        input_session_ids: Sequence[object],
        input_channel_counts: torch.Tensor | Sequence[int] | None = None,
        input_seq_len: torch.Tensor | Sequence[int] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        embedding = self.readout_dropout(
            self.encode(
                input_values=input_values,
                input_session_ids=input_session_ids,
                input_channel_counts=input_channel_counts,
                input_seq_len=input_seq_len,
            )
        )
        batch_size, n_out = task_index.shape
        routed = (
            embedding.unsqueeze(1)
            .expand(batch_size, n_out, -1)
            .reshape(-1, self.embedding_dim)
        )
        flat_index = task_index.reshape(-1)
        valid = flat_index > 0
        return self.router(routed[valid], (flat_index[valid] - 1).long())

    def tokenize(self, data: Data) -> dict[str, Any]:
        """Convert a raw NeuroSoft EEG/ECoG window into collatable tensors."""
        _, signal_source, keep, _ = resolve_neural_signal(
            data, frozenset(self.SUPPORTED_MODALITIES)
        )
        signal = np.asarray(signal_source.signal, dtype=np.float32)[:, keep]
        raw_session_id = str(data.session.id)
        namespace = getattr(data, "dataset_namespace", None)
        session_id = self.resolve_session_id(raw_session_id, namespace)
        if session_id not in self.session_adapter.layers:
            raise KeyError(f"Unknown NeuroSoft session ID {session_id!r}")
        if signal.shape[1] != self.session_adapter.channel_counts[session_id]:
            raise ValueError(
                f"Session {session_id!r} has {signal.shape[1]} supported channels; expected {self.session_adapter.channel_counts[session_id]}"
            )
        _, output_values, output_task_index, output_weights = (
            extract_multitask_targets(self._task_configs, data)
        )
        return {
            # ``pad2d`` collates its trailing dimensions, so channel-first
            # input yields the model contract ``(B, C_pad, T_pad)``.
            "input_values": pad2d(torch.from_numpy(signal.T.copy())),
            "input_session_ids": session_id,
            "input_channel_counts": torch.tensor(
                signal.shape[1], dtype=torch.long
            ),
            "input_seq_len": torch.tensor(signal.shape[0], dtype=torch.long),
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": session_id,
            "absolute_start": float(data.absolute_start),
        }
