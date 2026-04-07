"""Backward-compatible shim for base.py module.

.. deprecated::
    Import from ``foundry.models.embeddings.legacy`` and
    ``foundry.models.embeddings.activations`` instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.base is deprecated. "
    "Import EmbeddingBase and FixedChannelWindowEmbedding from "
    "foundry.models.embeddings.legacy, and get_activation from "
    "foundry.models.embeddings.activations instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.activations import get_activation  # noqa: E402
from foundry.models.embeddings.legacy import (  # noqa: E402
    EmbeddingBase,
    FixedChannelWindowEmbedding,
)

__all__ = [
    "EmbeddingBase",
    "FixedChannelWindowEmbedding",
    "get_activation",
]
