from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from temporaldata import Data
from torch_brain.data import chain, pad8
from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    RotaryTimeEmbedding,
    prepare_for_multitask_readout,
)
from torch_brain.registry import ModalitySpec
from torch_brain.utils import create_linspace_latent_tokens

from foundry.models.backbones import PerceiverIOBackbone
from foundry.models.tokenizer import EEGTokenizer
from foundry.models.utils import resolve_readout_specs

RECON_DECODER_ID = 9999


class POYOEEGModel(nn.Module):
    """POYO-style EEG model with built-in Perceiver architecture.

    Supports three modes controlled entirely by which optional heads are
    configured:

    - **Supervised only**: ``readout_specs`` provided, no reconstruction head.
    - **Pretrain only**: ``reconstruction_head`` provided, no ``readout_specs``.
    - **Joint**: both provided -- decoder output is split between heads.

    Args:
        tokenizer: Composable tokenizer handling channel strategy,
            optional GPU patching, and temporal embedding.
        embed_dim: Embedding dimension (must match components).
        sequence_length: Length of sequences in seconds.
        readout_specs: Optional task specifications for multitask readout.
        reconstruction_head: Optional module projecting decoder embeddings
            back to signal space for masked reconstruction pretraining.
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
    """

    SUPPORTED_MODALITIES = {"eeg", "ecog", "seeg", "ieeg"}

    def __init__(
        self,
        tokenizer: EEGTokenizer,
        embed_dim: int,
        sequence_length: float,
        readout_specs: list[ModalitySpec | str]
        | dict[str, ModalitySpec]
        | None = None,
        reconstruction_head: nn.Module | None = None,
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
    ):
        super().__init__()

        if readout_specs is None and reconstruction_head is None:
            raise ValueError(
                "At least one of readout_specs or reconstruction_head "
                "must be provided."
            )

        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self._latent_index, self._latent_timestamps = (
            create_linspace_latent_tokens(
                0,
                self.sequence_length,
                step=self.latent_step,
                num_latents_per_step=self.num_latents_per_step,
            )
        )

        # --- Supervised readout (optional) ---
        if readout_specs is not None:
            self._readout_specs = resolve_readout_specs(readout_specs)
            self.global_to_local_task_id = {
                spec.id: idx
                for idx, spec in enumerate(self._readout_specs.values())
            }
            self.task_emb = Embedding(
                len(self._readout_specs), embed_dim, init_scale=emb_init_scale
            )
            self.readout = MultitaskReadout(
                dim=embed_dim,
                readout_specs=self._readout_specs,
            )
        else:
            self._readout_specs = None
            self.global_to_local_task_id = None
            self.task_emb = None
            self.readout = None

        # --- Reconstruction head (optional) ---
        self.reconstruction_head = reconstruction_head
        if reconstruction_head is not None:
            self.recon_task_emb = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.recon_task_emb, std=emb_init_scale)

        self.backbone = PerceiverIOBackbone(
            embed_dim=embed_dim,
            depth=depth,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
        )

        self.channel_emb = InfiniteVocabEmbedding(
            self.embed_dim, init_scale=emb_init_scale
        )
        self.session_emb = InfiniteVocabEmbedding(
            self.embed_dim, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(
            num_latents_per_step, self.embed_dim, init_scale=emb_init_scale
        )
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

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
        output_decoder_index: torch.Tensor,
        masking_mask: Optional[torch.Tensor] = None,
        unpack_output: bool = False,
    ) -> Dict:
        """Forward pass through the model.

        Tokenizes the raw signal, builds output queries for whichever
        heads are active (task readout and/or reconstruction), runs the
        Perceiver IO backbone, and routes the decoder output to the
        appropriate heads.

        Args:
            input_values: ``(B, C, T)`` raw signal (padded to max T).
            input_timestamps: ``(B, num_tokens)`` timestamps per token.
            input_channel_index: ``(B, C)`` channel identity tokens.
            input_session_index: ``(B,)`` session index for input.
            input_mask: ``(B, C)`` channel validity mask.
            input_sampling_rate: ``(B,)`` per-item sampling rate.
            input_seq_len: ``(B,)`` per-item true sample count.
            input_session_ids: Per-item session ID strings (for
                session-specific spatial projectors).
            input_channel_counts: Per-item true channel counts.
            latent_index: ``(B, n_latent)`` indices for latent tokens.
            latent_timestamps: ``(B, n_latent)`` timestamps for latents.
            output_session_index: ``(B, n_out)`` session indices for
                output queries.
            output_timestamps: ``(B, n_out)`` timestamps for outputs.
            output_decoder_index: ``(B, n_out)`` task/decoder indices.
            masking_mask: ``(B, num_tokens)`` boolean mask indicating
                which input tokens are masked (passed through to the
                tokenizer's ``forward``).
            unpack_output: Whether to unpack outputs by batch sample.

        Returns:
            Dictionary whose keys depend on which heads are active.
            See :meth:`_route_decoder_output` for details.
        """
        self._validate_vocab_initialization()

        inputs = self.tokenizer(
            input_values,
            input_channel_index=input_channel_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            input_session_ids=input_session_ids,
            input_channel_counts=input_channel_counts,
            channel_emb_fn=self.channel_emb,
            masking_mask=masking_mask,
        )

        session_emb = self.session_emb(input_session_index).unsqueeze(1)
        inputs = inputs + session_emb

        input_timestamp_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # --- Build output queries ---
        output_queries, output_timestamp_emb, n_task_queries = (
            self._build_output_queries(
                output_session_index,
                output_timestamps,
                output_decoder_index,
            )
        )

        output_latents = self.backbone(
            inputs=inputs,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=output_queries,
            output_timestamp_emb=output_timestamp_emb,
        )

        return self._route_decoder_output(
            output_latents,
            output_decoder_index,
            n_task_queries,
            unpack_output,
        )

    def _build_output_queries(
        self,
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor,
        output_decoder_index: torch.Tensor,
    ):
        """Build combined output queries for task readout and/or reconstruction.

        In supervised (or joint) mode the queries are
        ``session_emb + task_emb`` at task timestamps.  In pretrain-only
        mode they are ``session_emb + recon_task_emb`` at masked
        timestamps.

        Args:
            output_session_index: ``(B, n_out)`` session indices.
            output_timestamps: ``(B, n_out)`` output timestamps.
            output_decoder_index: ``(B, n_out)`` decoder/task IDs.

        Returns:
            Tuple of ``(output_queries, output_timestamp_emb,
            n_task_queries)`` where *n_task_queries* is the number of
            task readout queries in the second dimension (0 when in
            pretrain-only mode).
        """
        if self.readout is not None:
            local_task_index = torch.zeros_like(output_decoder_index)
            for global_id, local_id in self.global_to_local_task_id.items():
                local_task_index[output_decoder_index == global_id] = local_id

            task_queries = self.session_emb(
                output_session_index
            ) + self.task_emb(local_task_index)
            task_timestamp_emb = self.rotary_emb(output_timestamps)
            n_task_queries = output_session_index.shape[1]

            return task_queries, task_timestamp_emb, n_task_queries
        else:
            # Pretrain-only: reconstruction queries are the only output queries
            # They are already included in output_session_index/timestamps
            recon_queries = (
                self.session_emb(output_session_index) + self.recon_task_emb
            )
            recon_timestamp_emb = self.rotary_emb(output_timestamps)
            return recon_queries, recon_timestamp_emb, 0

    def _route_decoder_output(
        self,
        output_latents: torch.Tensor,
        output_decoder_index: torch.Tensor,
        n_task_queries: int,
        unpack_output: bool,
    ) -> Dict:
        """Split decoder output and route to the active head(s).

        * **Supervised only** -- all embeddings go to ``MultitaskReadout``.
          Returns ``{"task_name": tensor, ...}``.
        * **Pretrain only** -- embeddings at ``RECON_DECODER_ID``
          positions go to ``ReconstructionHead``.
          Returns ``{"reconstruction": tensor}``.
        * **Joint** -- the first ``n_task_queries`` columns go to
          ``MultitaskReadout``; remaining ``RECON_DECODER_ID`` columns
          go to ``ReconstructionHead``.
          Returns ``{"task_name": ..., "reconstruction": tensor}``.

        Args:
            output_latents: ``(B, n_out, embed_dim)`` decoder output.
            output_decoder_index: ``(B, n_out)`` per-query decoder IDs.
            n_task_queries: Number of task readout queries occupying the
                leading columns of the decoder output.
            unpack_output: Passed through to ``MultitaskReadout``.

        Returns:
            Dict of named output tensors (see mode descriptions above).
        """
        output = {}

        if self.readout is not None:
            if self.reconstruction_head is not None:
                # Joint mode: split by decoder index
                recon_mask = output_decoder_index == RECON_DECODER_ID

                task_latents = output_latents[:, :n_task_queries]
                task_decoder_index = output_decoder_index[:, :n_task_queries]

                task_output = self.readout(
                    output_embs=task_latents,
                    output_readout_index=task_decoder_index,
                    unpack_output=unpack_output,
                )
                output.update(task_output)

                recon_latents = output_latents[:, n_task_queries:]
                output["reconstruction"] = self.reconstruction_head(
                    recon_latents[recon_mask[:, n_task_queries:]]
                )
            else:
                # Supervised only
                output = self.readout(
                    output_embs=output_latents,
                    output_readout_index=output_decoder_index,
                    unpack_output=unpack_output,
                )
        elif self.reconstruction_head is not None:
            # Pretrain only: all outputs go to reconstruction head
            recon_mask = output_decoder_index == RECON_DECODER_ID
            output["reconstruction"] = self.reconstruction_head(
                output_latents[recon_mask]
            )

        return output

    @property
    def readout_specs(self) -> dict[str, ModalitySpec] | None:
        """Resolved readout specs, or ``None`` in pretrain-only mode."""
        return self._readout_specs

    def _resolve_signal_source(self, data: Data):
        """Find the signal source and default modality type from the data.

        Checks for ``eeg``, ``ecog``, and ``seeg`` attributes on *data*
        and returns the first one found.

        Args:
            data: TemporalData object to inspect.

        Returns:
            Tuple of ``(signal_source, default_type)`` where
            *signal_source* is the time-series object and
            *default_type* is the uppercase modality name (e.g.
            ``"EEG"``).

        Raises:
            ValueError: If none of the supported modality fields exist.
        """
        for modality in ["eeg", "ecog", "seeg"]:
            signal = getattr(data, modality, None)
            if signal is not None:
                return signal, modality.upper()

        raise ValueError("Data must have an 'eeg', 'ecog', or 'seeg' field")

    def tokenize(self, data: Data) -> dict:
        """Tokenize the input data.

        Delegates signal preparation to the :class:`EEGTokenizer` which
        handles channel strategy, optional GPU patching, and temporal
        embedding concerns.

        Supports three modes:
        - **Supervised**: builds task readout queries from ``readout_specs``.
        - **Pretrain**: builds reconstruction queries from masked positions.
        - **Joint**: concatenates both sets of output queries.

        Args:
            data: TemporalData object containing raw EEG/ECoG/sEEG signal.

        Returns:
            dict with model_inputs, target_values, target_weights, and
            metadata.
        """
        if not hasattr(data, "config") or data.config is None:
            data.config = {}

        signal_source, default_type = self._resolve_signal_source(data)

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )

        channel_ids = data.channels.id[modality_mask].astype(str)
        channel_tokens = np.asarray(self.channel_emb.tokenizer(channel_ids))

        sample_deltas = np.diff(signal_source.timestamps)
        sampling_rate = 1.0 / float(sample_deltas[0])

        pretokenized = self.tokenizer.pretokenize(
            signal=signal_source.signal[:, modality_mask],
            channel_tokens=channel_tokens,
            sampling_rate=sampling_rate,
            sequence_length=self.sequence_length,
        )
        pretokenized["input_session_ids"] = str(data.session.id)
        input_timestamps = pretokenized.pop("input_timestamps")

        masking_mask = pretokenized.pop("masking_mask", None)
        reconstruction_targets = pretokenized.pop(
            "reconstruction_targets", None
        )
        masked_timestamps = pretokenized.pop("masked_timestamps", None)

        latent_index = self._latent_index
        latent_timestamps = self._latent_timestamps

        input_session_index = self.session_emb.tokenizer(data.session.id)

        result = {
            **pretokenized,
            "input_timestamps": input_timestamps,
            "input_session_index": input_session_index,
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
        }

        # --- Supervised readout queries ---
        if self._readout_specs is not None:
            if "multitask_readout" not in data.config:
                data.config["multitask_readout"] = [
                    {"readout_id": spec_id}
                    for spec_id in self._readout_specs.keys()
                ]
            else:
                available = [
                    cfg["readout_id"]
                    for cfg in data.config["multitask_readout"]
                ]
                data.config["multitask_readout"] = [
                    {"readout_id": name}
                    for name in available
                    if name in self._readout_specs
                ]

            (
                output_timestamps,
                output_values,
                output_task_index,
                output_weights,
                output_eval_mask,
            ) = prepare_for_multitask_readout(
                data,
                self._readout_specs,
            )
            output_timestamps = torch.zeros_like(output_timestamps)
            output_session_index = np.full(
                len(output_timestamps), input_session_index
            )
        else:
            output_timestamps = torch.tensor([], dtype=torch.float32)
            output_values = {}
            output_task_index = torch.tensor([], dtype=torch.long)
            output_weights = {}
            output_eval_mask = {}
            output_session_index = np.array([], dtype=np.int64)

        # --- Reconstruction queries ---
        if (
            self.reconstruction_head is not None
            and masked_timestamps is not None
        ):
            n_masked = masked_timestamps.shape[0]
            recon_timestamps = torch.zeros(n_masked, dtype=torch.float32)
            recon_decoder_index = torch.full(
                (n_masked,), RECON_DECODER_ID, dtype=torch.long
            )
            recon_session_index = np.full(n_masked, input_session_index)

            output_timestamps = torch.cat([output_timestamps, recon_timestamps])
            output_task_index = torch.cat(
                [output_task_index, recon_decoder_index]
            )
            output_session_index = np.concatenate(
                [output_session_index, recon_session_index]
            )

        result["output_session_index"] = pad8(output_session_index)
        result["output_timestamps"] = pad8(output_timestamps)
        result["output_decoder_index"] = pad8(output_task_index)
        result["target_values"] = chain(output_values, allow_missing_keys=True)
        result["target_weights"] = chain(
            output_weights, allow_missing_keys=True
        )
        result["session_id"] = data.session.id
        result["absolute_start"] = data.absolute_start
        result["eval_mask"] = chain(output_eval_mask, allow_missing_keys=True)

        if masking_mask is not None:
            result["masking_mask"] = masking_mask
        if reconstruction_targets is not None:
            result["reconstruction_targets"] = reconstruction_targets

        return result

    def initialize_vocabs(self, vocab_info: dict):
        """Initialize vocabularies from dataset information.

        Args:
            vocab_info: Dictionary with ``session_ids`` and/or
                ``channel_ids`` keys mapping to lists of ID strings.
        """
        if "session_ids" in vocab_info and self.session_emb.is_lazy():
            self.session_emb.initialize_vocab(vocab_info["session_ids"])

        if "channel_ids" in vocab_info and self.channel_emb.is_lazy():
            self.channel_emb.initialize_vocab(vocab_info["channel_ids"])

    def has_lazy_vocabs(self) -> bool:
        """Check if vocabularies are still lazy (uninitialized)."""
        return self.channel_emb.is_lazy() or self.session_emb.is_lazy()

    def _validate_vocab_initialization(self):
        """Raise ``ValueError`` if channel or session vocabs are still lazy."""
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
