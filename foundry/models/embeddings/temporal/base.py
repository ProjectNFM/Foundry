"""Abstract base class for temporal embeddings."""

from abc import ABC, abstractmethod

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


__all__ = ["TemporalEmbedding"]
