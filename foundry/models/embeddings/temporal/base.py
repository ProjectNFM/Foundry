"""Abstract base class for temporal embeddings."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module, ABC):
    """Interface shared by all temporal embedding modules.

    Subclasses must implement :meth:`get_num_time_tokens` (how many tokens
    this embedding produces for a given window) and the
    :attr:`has_fixed_token_count` property (whether the count is independent
    of the input sampling rate).
    """

    @abstractmethod
    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        """Return the number of time tokens for a given window.

        Args:
            sequence_length: Duration of the context window in seconds.
            sampling_rate: Input sampling rate in Hz.
        """
        ...

    @property
    @abstractmethod
    def has_fixed_token_count(self) -> bool:
        """True when the token count depends only on *sequence_length*
        (not on *sampling_rate*).  False for per-timepoint embeddings
        whose count equals the number of input samples."""
        ...


class TokenRateTemporalEmbedding(TemporalEmbedding, ABC):
    """Temporal embedding whose token count is ``round(rate × duration)``.

    Stores ``target_token_rate`` and provides concrete implementations of
    :meth:`get_num_time_tokens`, :attr:`has_fixed_token_count`, and the
    GPU-side :meth:`_compute_target_tokens` helper so subclasses only need
    to implement :meth:`forward`.
    """

    target_token_rate: float

    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        return max(1, round(self.target_token_rate * sequence_length))

    @property
    def has_fixed_token_count(self) -> bool:
        return True

    def _compute_target_tokens(
        self,
        input_seq_len: torch.Tensor,
        input_sampling_rate: torch.Tensor,
    ) -> int:
        cached = getattr(self, "_cached_target_tokens", None)
        if cached is not None:
            return cached
        durations = input_seq_len.float() / input_sampling_rate
        max_duration = durations.max().item()
        result = max(1, round(self.target_token_rate * max_duration))
        self._cached_target_tokens = result
        return result

    def train(self, mode: bool = True):
        if mode != self.training:
            self._cached_target_tokens = None
        return super().train(mode)


__all__ = ["TemporalEmbedding", "TokenRateTemporalEmbedding"]
