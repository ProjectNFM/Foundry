"""Core abstractions for neural data processing.

This module provides modality-agnostic base classes and protocols for working
with neural data from any source (EEG, iEEG, fMRI, PET, etc.).
"""

from typing import Protocol, Any, Dict

import torch
from temporaldata import Data

from typing_extensions import runtime_checkable


class NeuralModel(Protocol):
    """Protocol for neural data models.

    Defines the expected interface for models that can be trained on neural data.
    Models don't need to explicitly inherit from this; any model with these methods
    will be compatible.
    """

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            **kwargs: Model-specific inputs (e.g., input_values, input_timestamps, etc.)

        Returns:
            Dictionary of model outputs, typically with keys for each task/head.
        """
        ...

    @property
    def readout_specs(self) -> Dict[str, Any]:
        """Task specifications for multitask learning.

        Returns:
            Dictionary mapping task names to ModalitySpec objects.
        """
        ...


class Tokenizable(Protocol):
    """Protocol for objects that can tokenize temporal data.

    Tokenization converts raw neural data samples into model-compatible tensor
    representations with associated metadata.
    """

    def tokenize(self, data: Data) -> Dict[str, Any]:
        """Tokenize a temporal data sample.

        Args:
            data: A temporaldata.Data object containing the raw signal and metadata.

        Returns:
            Dictionary with tokenized outputs (e.g., input_values, target_labels, weights).
        """
        ...


@runtime_checkable
class VocabManager(Protocol):
    """Protocol for managing vocabularies (e.g., session IDs, channel IDs).

    Some models use infinite vocabulary embeddings that must be initialized with
    the full set of possible identifiers before training.
    """

    def initialize_vocabs(self, vocab_info: dict):
        """Initialize vocabularies from dataset information.

        Args:
            vocab_info: Dictionary with keys for any identifier list ending in '_ids' (e.g., 'session_ids', 'channel_ids', etc.)
        """
        ...

    def has_lazy_vocabs(self) -> bool:
        """Check if vocabularies are still lazy (uninitialized).

        Returns:
            True if any vocabulary needs initialization, False if all are ready.
        """
        ...
