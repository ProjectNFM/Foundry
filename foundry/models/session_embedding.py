"""Dynamic session embedding: signal-conditioned session representations.

Provides three components:

1. :class:`DynamicSessionEncoder` — lightweight pooling head that aggregates
   tokenized context windows into a single session embedding vector.
2. :class:`SessionContextCache` — CPU-side cache of pretokenized context
   windows, populated lazily per session.
3. :class:`SessionEmbeddingCache` — GPU-side cache of computed embeddings,
   used at inference to skip redundant tokenizer + pooling passes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch_brain.data import Data

logger = logging.getLogger(__name__)


class DynamicSessionEncoder(nn.Module):
    """Pool tokenized context windows into a session embedding.

    Operates AFTER the shared :class:`EEGTokenizer` has already handled
    channel/rate heterogeneity.  Input is the token-level output of the
    tokenizer for each context window.

    Architecture:
      1. Receive tokenized context: ``(B, W, N_tokens, embed_dim)``
      2. Pool over tokens within each window → ``(B, W, embed_dim)``
      3. Pool over windows → ``(B, embed_dim)``
      4. Linear projection → ``(B, embed_dim)``

    Args:
        embed_dim: Model embedding dimension.
        pool_mode: Pooling strategy — ``"mean"`` for masked mean pooling,
            ``"attention"`` for learned single-head attention pooling.
    """

    def __init__(self, embed_dim: int, pool_mode: str = "mean"):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_mode = pool_mode

        if pool_mode == "attention":
            self.token_attn = nn.Linear(embed_dim, 1)
            self.window_attn = nn.Linear(embed_dim, 1)
        elif pool_mode != "mean":
            raise ValueError(
                f"Unknown pool_mode '{pool_mode}', expected 'mean' or 'attention'"
            )

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool context tokens into a session embedding.

        Args:
            context_tokens: ``(B, W, N, D)`` tokenized context windows.
            context_token_mask: ``(B, W, N)`` boolean validity mask.
                ``True`` = valid token. When ``None`` all tokens are valid.

        Returns:
            ``(B, 1, D)`` session embedding, broadcast-ready for addition
            to input tokens.
        """
        B, W, N, D = context_tokens.shape

        if context_token_mask is None:
            context_token_mask = torch.ones(
                B, W, N, dtype=torch.bool, device=context_tokens.device
            )

        # --- Step 1: pool tokens within each window → (B, W, D) ---
        window_embs = self._pool_tokens(context_tokens, context_token_mask)

        # --- Step 2: pool across windows → (B, D) ---
        window_mask = context_token_mask.any(dim=-1)  # (B, W)
        session_emb = self._pool_windows(window_embs, window_mask)

        # --- Step 3: project ---
        session_emb = self.proj(session_emb)

        return session_emb.unsqueeze(1)  # (B, 1, D)

    def _pool_tokens(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool over the N (token) dimension per window."""
        if self.pool_mode == "mean":
            return _masked_mean(tokens, mask.unsqueeze(-1), dim=2)

        scores = self.token_attn(tokens).squeeze(-1)  # (B, W, N)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, W, N, 1)
        return (tokens * weights).sum(dim=2)  # (B, W, D)

    def _pool_windows(
        self, window_embs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool over the W (window) dimension."""
        if self.pool_mode == "mean":
            return _masked_mean(window_embs, mask.unsqueeze(-1), dim=1)

        scores = self.window_attn(window_embs).squeeze(-1)  # (B, W)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, W, 1)
        return (window_embs * weights).sum(dim=1)  # (B, D)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute mean over *dim*, ignoring positions where *mask* is False."""
    x = x.masked_fill(~mask, 0.0)
    count = mask.sum(dim=dim).clamp(min=1)
    return x.sum(dim=dim) / count


# ---------------------------------------------------------------------------
# CPU-side context window cache
# ---------------------------------------------------------------------------


class SessionContextCache:
    """Cache pretokenized context windows per session.

    At first access for a session the cache:
      1. Samples ``num_windows`` windows from the recording.
      2. Runs the provided ``prepare_fn`` (signal filtering, normalization).
      3. Runs ``pretokenize_fn`` on each window.
      4. Stacks the result tensors.

    Subsequent samples from the same session reuse the cached tensors.

    Args:
        num_windows: Number of context windows per session.
        context_source: ``"random"`` — uniform random start positions, or
            ``"start"`` — first ``num_windows`` contiguous windows.
        context_duration: Duration of each context window in seconds.
    """

    def __init__(
        self,
        num_windows: int = 5,
        context_source: str = "random",
        context_duration: float = 2.0,
    ):
        self.num_windows = num_windows
        self.context_source = context_source
        self.context_duration = context_duration
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._rng = np.random.RandomState(42)

    def get_or_build(
        self,
        session_id: str,
        data: "Data",
        prepare_fn: Callable,
        pretokenize_fn: Callable,
        channel_vocab_fn: Callable,
    ) -> dict[str, torch.Tensor]:
        """Return cached context or build it from *data*.

        Args:
            session_id: Unique session identifier.
            data: :class:`Data` object for any sample from this session.
                Used to extract channel info and recording domain.
            prepare_fn: ``_prepare_signal``-like callable producing a
                :class:`PreparedSignal`.
            pretokenize_fn: ``tokenizer.pretokenize`` callable.
            channel_vocab_fn: Vocabulary mapping callable (e.g.
                ``channel_emb.tokenizer``) that converts channel string
                IDs to integer token indices.

        Returns:
            Dict with keys ``context_values``, ``context_channel_index``,
            ``context_mask``, ``context_sampling_rate`` — all stacked over
            the window dimension.
        """
        if session_id in self._cache:
            return self._cache[session_id]

        context = self._build_context(
            data, prepare_fn, pretokenize_fn, channel_vocab_fn
        )
        self._cache[session_id] = context
        return context

    def _build_context(
        self,
        data: "Data",
        prepare_fn: Callable,
        pretokenize_fn: Callable,
        channel_vocab_fn: Callable,
    ) -> dict[str, torch.Tensor]:
        """Sample context windows and pretokenize them."""
        starts = self._sample_window_starts(data)

        all_values, all_channel_index, all_mask, all_sr = [], [], [], []

        for start in starts:
            end = start + self.context_duration
            window_data = data.slice(start, end)

            prepared = prepare_fn(window_data)

            channel_ids = window_data.channels.id[
                prepared.modality_mask
            ].astype(str)
            channel_tokens = np.asarray(channel_vocab_fn(channel_ids))

            pretok = pretokenize_fn(
                signal=prepared.signal,
                channel_tokens=channel_tokens,
                sampling_rate=prepared.sampling_rate,
                sequence_length=self.context_duration,
            )

            all_values.append(pretok["input_values"])
            all_channel_index.append(pretok["input_channel_index"])
            all_mask.append(pretok["input_mask"])
            all_sr.append(
                torch.tensor(prepared.sampling_rate, dtype=torch.float32)
            )

        return {
            "context_values": torch.stack(all_values),  # (W, C, T)
            "context_channel_index": torch.stack(all_channel_index),  # (W, C)
            "context_mask": torch.stack(all_mask),  # (W, C)
            "context_sampling_rate": torch.stack(all_sr),  # (W,)
        }

    def _sample_window_starts(self, data: "Data") -> list[float]:
        """Compute context window start positions."""
        ds = data.domain.start
        de = data.domain.end
        domain_start = float(ds.item() if hasattr(ds, "item") else ds)
        domain_end = float(de.item() if hasattr(de, "item") else de)
        usable = domain_end - domain_start - self.context_duration

        if usable <= 0:
            return [domain_start] * self.num_windows

        if self.context_source == "start":
            step = min(
                self.context_duration,
                usable / max(self.num_windows, 1),
            )
            return [domain_start + i * step for i in range(self.num_windows)]

        return [
            domain_start + self._rng.uniform(0, usable)
            for _ in range(self.num_windows)
        ]

    def clear(self) -> None:
        """Invalidate all cached context windows."""
        self._cache.clear()

    def reseed(self, seed: int) -> None:
        """Reset the RNG and clear the cache (for epoch-level re-sampling)."""
        self._rng = np.random.RandomState(seed)
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached sessions."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# GPU-side embedding cache (inference only)
# ---------------------------------------------------------------------------


class SessionEmbeddingCache:
    """Cache computed dynamic session embeddings during inference.

    After the first forward pass for a session, the resulting embedding is
    stored and reused for all subsequent samples.  Should be cleared when
    the model switches back to training mode or between validation epochs.
    """

    def __init__(self):
        self._cache: dict[int, torch.Tensor] = {}

    def get(self, session_index: int) -> torch.Tensor | None:
        """Return the cached embedding for *session_index*, or ``None``."""
        return self._cache.get(session_index)

    def put(self, session_index: int, embedding: torch.Tensor) -> None:
        """Store a detached copy of *embedding* for *session_index*."""
        self._cache[session_index] = embedding.detach()

    def clear(self) -> None:
        """Remove all cached embeddings."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached sessions."""
        return len(self._cache)

    def __contains__(self, session_index: int) -> bool:
        """Check whether *session_index* has a cached embedding."""
        return session_index in self._cache
