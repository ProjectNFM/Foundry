from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_brain.data import Data
from torch_brain.batching import chain, pad8
from torch_brain.nn import InfiniteVocabEmbedding, RotaryTimeEmbedding
from foundry.models.backbones import PerceiverIOBackbone
from foundry.models.readout import build_readout_router
from foundry.models.signal_preparation import (
    PreparedSignal,
    normalize_encoder_inputs,
)
from foundry.models.ssl_meta import ModelOutput
from foundry.models.tokenizer import EEGTokenizer
from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import extract_multitask_targets


class POYOEEGModel(nn.Module):
    """POYO-style EEG model with built-in Perceiver architecture.

    This model uses a PerceiverIO backbone internally and accepts an
    :class:`EEGTokenizer` that composes a channel strategy with a
    temporal embedding to convert raw EEG signal into token sequences.

    Args:
        tokenizer: Composable tokenizer handling channel strategy,
            optional GPU patching, and temporal embedding.
        task_configs: Mapping from task name to :class:`~foundry.tasks.config.TaskConfig`.
        embed_dim: Embedding dimension (must match components).
        sequence_length: Length of sequences in seconds.
        latent_step: Time step between latent tokens in seconds.
        num_latents_per_step: Number of latent tokens per time step.
        depth: Number of processor layers.
        dim_head: Dimension per attention head.
        cross_heads: Number of attention heads for cross-attention.
        self_heads: Number of attention heads for self-attention.
        ffn_dropout: Dropout rate for feedforward networks.
        lin_dropout: Dropout rate for linear layers in processor.
        atn_dropout: Dropout rate for attention layers.
        emb_init_scale: Initialization scale for embeddings.
        t_min: Minimum time value for rotary encoding.
        t_max: Maximum time value for rotary encoding.
        zero_output_timestamps: If True, replaces all output query timestamps
            with zeros before decoder cross-attention. This is useful for
            window-level classification tasks where labels are not tied to a
            precise timepoint. Keep this False for timestamp-aware tasks such
            as trajectory regression.
        normalize_inputs: If True, per-channel z-score the input signal in
            ``_prepare_signal()``. Ensures scale invariance across datasets
            with different amplifier gains and physical units. Recommended
            for pretraining; not needed when downstream data is already
            normalized.
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}
    _TRANSFERABLE_COMPONENTS = (
        "tokenizer",
        "backbone",
        "rotary_emb",
        "latent_emb",
    )

    def transferable_components(self) -> tuple[str, ...]:
        """Return the names of top-level submodules whose weights are shared
        between pretraining and finetuning and should be transferred from a
        pretrained checkpoint.

        Dataset/task-specific components (``channel_emb``, ``session_emb``,
        ``task_emb``, ``router``) are excluded by construction — they are
        simply everything *not* listed here.
        """
        return self._TRANSFERABLE_COMPONENTS

    def __init__(
        self,
        tokenizer: EEGTokenizer,
        task_configs: dict[str, TaskConfig],
        embed_dim: int,
        sequence_length: float,
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
        zero_output_timestamps: bool = False,
        normalize_inputs: bool = False,
        rotate_value: bool = True,
        disable_session_emb: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.zero_output_timestamps = zero_output_timestamps
        self.normalize_inputs = normalize_inputs
        self.disable_session_emb = disable_session_emb
        self._task_configs = TaskConfig.normalize_task_configs(task_configs)
        self._latent_index, self._latent_timestamps = (
            create_linspace_latent_tokens(
                0,
                self.sequence_length,
                step=self.latent_step,
                num_latents_per_step=self.num_latents_per_step,
            )
        )

        self.router = build_readout_router(self._task_configs, embed_dim)

        self.backbone = PerceiverIOBackbone(
            embed_dim=embed_dim,
            depth=depth,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            rotate_value=rotate_value,
        )

        self.channel_emb = InfiniteVocabEmbedding(
            self.tokenizer.channel_emb_dim, init_scale=emb_init_scale
        )
        self.session_emb = InfiniteVocabEmbedding(
            self.embed_dim, init_scale=emb_init_scale
        )
        self.task_emb = nn.Embedding(self.router.num_tasks, self.embed_dim)
        nn.init.normal_(self.task_emb.weight, mean=0, std=emb_init_scale)
        self.latent_emb = nn.Embedding(num_latents_per_step, self.embed_dim)
        nn.init.normal_(self.latent_emb.weight, mean=0, std=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

    def _tokenize_and_add_session(
        self,
        input_values: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        input_sampling_rate: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        input_session_ids=None,
        input_channel_counts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU tokenization + session embedding addition.

        Returns:
            ``(inputs, session_emb)`` where *inputs* has shape
            ``(B, num_tokens, embed_dim)`` and *session_emb* has shape
            ``(B, 1, embed_dim)``.
        """
        inputs = self.tokenizer(
            input_values,
            input_channel_index=input_channel_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            input_session_ids=input_session_ids,
            input_channel_counts=input_channel_counts,
            channel_emb_fn=self.channel_emb,
        )
        if self.disable_session_emb:
            session_emb = torch.zeros(
                inputs.shape[0],
                1,
                self.embed_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            session_emb = self.session_emb(input_session_index).unsqueeze(1)
            inputs = inputs + session_emb
        return inputs, session_emb

    # ------------------------------------------------------------------
    # Orchestration helpers – shared between base and masked forward
    # ------------------------------------------------------------------

    def _build_latents(
        self,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct latent embeddings and their rotary timestamp embeddings.

        Returns:
            ``(latents, latent_ts_emb)`` with shapes
            ``(B, n_latent, embed_dim)`` and the corresponding rotary pairs.
        """
        latents = self.latent_emb(latent_index)
        latent_ts_emb = self.rotary_emb(latent_timestamps)
        return latents, latent_ts_emb

    def _build_downstream_queries(
        self,
        output_session_index: torch.Tensor,
        task_index: torch.Tensor,
        output_timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct downstream task queries and their rotary timestamp embeddings.

        Owns the ``padded task_index -> router index`` conversion:
        batch ``task_index`` uses 0 for padding and ``router_idx + 1`` for
        real tasks; the embedding table is indexed by the 0-based router index.

        Returns:
            ``(queries, ts_emb)`` with shapes ``(B, n_out, embed_dim)`` and
            the corresponding rotary pairs.
        """
        task_ids = (task_index - 1).clamp(min=0)
        if self.disable_session_emb:
            queries = self.task_emb(task_ids)
        else:
            queries = self.session_emb(output_session_index) + self.task_emb(
                task_ids
            )
        ts_emb = self.rotary_emb(output_timestamps)
        return queries, ts_emb

    def _encode_and_process(
        self,
        inputs: torch.Tensor,
        input_ts_emb: torch.Tensor,
        latents: torch.Tensor,
        latent_ts_emb: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run encoder cross-attention followed by processor self-attention.

        Args:
            inputs: (B, N_in, D) embedded input tokens.
            input_ts_emb: Rotary pairs for *inputs*.
            latents: (B, N_lat, D) latent embeddings.
            latent_ts_emb: Rotary pairs for *latents*.
            input_mask: Optional (B, N_in) validity mask forwarded to the
                encoder cross-attention.

        Returns:
            Processed latents, same shape as *latents*.
        """
        latents = self.backbone.encoder(
            latents, inputs, latent_ts_emb, input_ts_emb, input_mask
        )
        latents = self.backbone.processor(latents, latent_ts_emb)
        return latents

    def _decode(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        query_ts_emb: torch.Tensor,
        latent_ts_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder cross-attention from processed latents to output queries.

        Returns:
            Output embeddings with the same leading dimensions as *queries*.
        """
        return self.backbone.decoder(
            queries, latents, query_ts_emb, latent_ts_emb
        )

    def _route(
        self,
        output_latents: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Flatten output latents and dispatch through the readout router.

        Uses the padded task-index convention: 0 means padding (skipped),
        positive values are ``router_idx + 1``.
        """
        B, N, D = output_latents.shape
        flat_embs = output_latents.reshape(B * N, D)
        flat_task_index = task_index.reshape(B * N)
        valid = flat_task_index > 0
        return self.router(
            flat_embs[valid], (flat_task_index[valid] - 1).long()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        input_sampling_rate: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        input_session_ids=None,
        input_channel_counts: Optional[torch.Tensor] = None,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor,
        task_index: torch.Tensor,
        unpack_output: bool = False,
    ) -> ModelOutput:
        """Forward pass through the model.

        Args:
            input_values: (B, C, T) raw signal (padded to max T in batch).
            input_timestamps: (B, num_tokens) timestamps per token.
            input_channel_index: (B, C) channel identity tokens.
            input_session_index: (B,) session index for input.
            input_mask: (B, C) channel validity mask.
            input_sampling_rate: (B,) per-item sampling rate.
            input_seq_len: (B,) per-item true sample count.
            latent_index: (B, n_latent) indices for latent tokens.
            latent_timestamps: (B, n_latent) timestamps for latent tokens.
            output_session_index: (B, n_out) session indices for outputs.
            output_timestamps: (B, n_out) timestamps for output predictions.
            task_index: (B, n_out) task/decoder indices.
            unpack_output: Whether to unpack outputs by batch sample.

        Returns:
            :class:`ModelOutput` with task-specific predictions.
        """
        del unpack_output
        self._validate_vocab_initialization()

        inputs, _session_emb = self._tokenize_and_add_session(
            input_values,
            input_channel_index,
            input_session_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            input_session_ids=input_session_ids,
            input_channel_counts=input_channel_counts,
        )
        input_ts_emb = self.rotary_emb(input_timestamps)

        latents, latent_ts_emb = self._build_latents(
            latent_index, latent_timestamps
        )
        queries, query_ts_emb = self._build_downstream_queries(
            output_session_index, task_index, output_timestamps
        )

        latents = self._encode_and_process(
            inputs, input_ts_emb, latents, latent_ts_emb
        )
        output_latents = self._decode(
            queries, latents, query_ts_emb, latent_ts_emb
        )

        return ModelOutput(task_outputs=self._route(output_latents, task_index))

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    def _resolve_signal_source(self, data: Data):
        """Find the signal source, default modality type, and sampling rate.

        Returns:
            Tuple of (signal_source, default_type, sampling_rate) where
            signal_source is the time series object, default_type is the
            uppercase modality name used when channels.type is absent, and
            sampling_rate is resolved from the signal source or inferred from
            timestamps.
        """
        for modality in ["eeg", "ecog", "seeg"]:
            signal = getattr(data, modality, None)
            if signal is not None:
                if (
                    hasattr(signal, "sampling_rate")
                    and signal.sampling_rate is not None
                ):
                    sampling_rate = float(signal.sampling_rate)
                else:
                    sampling_rate = self._infer_sampling_rate_from_timestamps(
                        signal.timestamps
                    )
                return signal, modality.upper(), sampling_rate

        raise ValueError("Data must have an 'eeg', 'ecog', or 'seeg' field")

    def _prepare_signal(self, data: Data) -> PreparedSignal:
        """Filter by modality, sanitize, normalize length, and optionally z-score.

        Shared logic used by both ``tokenize()`` and subclass target computation.
        Always length-normalizes so that encoder inputs and reconstruction targets
        share the same prepared signal.

        When ``self.normalize_inputs`` is True, per-channel z-scoring ensures
        scale invariance across datasets with different amplifier gains,
        impedances, and physical units.

        Args:
            data: Input data sample.

        Returns:
            :class:`PreparedSignal` containing the length-normalized,
            sanitized signal and token-grid metadata.
        """
        signal_source, default_type, sampling_rate = (
            self._resolve_signal_source(data)
        )

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )
        signal = signal_source.signal[:, modality_mask]

        non_finite = ~np.isfinite(signal)
        if non_finite.any():
            signal = np.where(non_finite, 0.0, signal)

        if self.normalize_inputs:
            signal = normalize_encoder_inputs(signal)

        return self.tokenizer.prepare_signal(
            signal, sampling_rate, self.sequence_length, modality_mask
        )

    def _infer_sampling_rate_from_timestamps(
        self, timestamps: np.ndarray
    ) -> float:
        sample_deltas = np.diff(timestamps).astype(np.float64)

        valid_deltas = sample_deltas[
            np.isfinite(sample_deltas) & (sample_deltas > 0)
        ]
        if valid_deltas.size == 0:
            raise ValueError(
                "Could not infer a valid sampling rate from timestamps."
            )
        return 1.0 / float(np.median(valid_deltas))

    def _extract_targets(self, data: Data):
        return extract_multitask_targets(self._task_configs, data)

    def _tokenize_core(self, data: Data) -> tuple[dict, PreparedSignal]:
        """Shared tokenization logic returning intermediate results.

        Returns:
            ``(result_dict, prepared_signal)`` where *prepared_signal* is the
            :class:`PreparedSignal` contract and *result_dict* is a complete
            tokenized sample ready for collation.
        """
        prepared = self._prepare_signal(data)

        channel_ids = data.channels.id[prepared.modality_mask].astype(str)
        channel_tokens = np.asarray(self.channel_emb.tokenizer(channel_ids))

        pretokenized = self.tokenizer.pretokenize(
            signal=prepared.signal,
            channel_tokens=channel_tokens,
            sampling_rate=prepared.sampling_rate,
            sequence_length=self.sequence_length,
        )
        pretokenized["input_session_ids"] = str(data.session.id)
        input_timestamps = pretokenized.pop("input_timestamps")

        latent_index = self._latent_index
        latent_timestamps = self._latent_timestamps

        input_session_index = self.session_emb.tokenizer(data.session.id)

        (
            output_timestamps,
            output_values,
            output_task_index,
            output_weights,
        ) = self._extract_targets(data)
        if self.zero_output_timestamps:
            output_timestamps = torch.zeros_like(output_timestamps)

        output_session_index = np.full(
            len(output_timestamps), input_session_index
        )

        result = {
            **pretokenized,
            "input_timestamps": input_timestamps,
            "input_session_index": input_session_index,
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "output_session_index": pad8(output_session_index),
            "output_timestamps": pad8(output_timestamps),
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
        }
        return result, prepared

    def tokenize(self, data: Data) -> dict:
        """Tokenize the input data.

        Delegates signal preparation to the :class:`EEGTokenizer` which
        handles channel strategy, optional GPU patching, and temporal
        embedding concerns. Target extraction uses :class:`TargetExtractor`
        instances from the configured task configs.

        Args:
            data: TemporalData object containing raw EEG/ECoG/sEEG signal.

        Returns:
            dict with model_inputs, target_values, target_weights, and
            metadata.
        """
        result, _prepared = self._tokenize_core(data)
        return result

    def initialize_vocabs(self, vocab_info: dict):
        """Initialize vocabularies from dataset information.

        Args:
            vocab_info: Dictionary with ``session_ids`` and ``channel_ids``
                keys.
        """
        if "session_ids" in vocab_info and self.session_emb.is_lazy():
            self.session_emb.initialize_vocab(vocab_info["session_ids"])

        if "channel_ids" in vocab_info and self.channel_emb.is_lazy():
            self.channel_emb.initialize_vocab(vocab_info["channel_ids"])

    def has_lazy_vocabs(self) -> bool:
        """Check if vocabularies are still lazy (uninitialized)."""
        return self.channel_emb.is_lazy() or self.session_emb.is_lazy()

    def _validate_vocab_initialization(self):
        """Validate that vocabularies have been properly initialized."""
        if self.channel_emb.is_lazy():
            raise ValueError(
                "Channel vocabulary has not been initialized, please use "
                "`model.channel_emb.initialize_vocab(channel_ids)`"
            )
        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )


def create_linspace_latent_tokens(
    start: float, end: float, step: float, num_latents_per_step: int
):
    """Create a sequence of evenly-spaced latent tokens.

    Each token is defined by a latent index and a timestamp.  Timestamps are
    placed at the centre of each step-sized bin between ``start`` and ``end``.
    Within every bin the group of ``num_latents_per_step`` latent indices is
    repeated.

    Args:
        start: Start time of the sequence.
        end: End time of the sequence.
        step: Time step between successive groups of latent tokens.
        num_latents_per_step: Number of latent tokens sharing each timestamp.

    Returns:
        Tuple of ``(latent_index, latent_timestamps)`` where both are 1-D
        ``np.ndarray`` of length ``num_steps * num_latents_per_step``.
    """
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    T = len(latent_timestamps)
    U = len(latent_index)

    latent_timestamps = np.repeat(latent_timestamps, U)  # (T,) -> (T*U,)
    latent_index = np.tile(latent_index, T)  # (U,) -> (T*U,)
    return latent_index, latent_timestamps
