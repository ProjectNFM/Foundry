"""Backward-compatible shim for cnn.py module.

.. deprecated::
    Import ``PatchCNNEmbedding`` from ``foundry.models.embeddings.temporal``
    or use the alias ``CNNEmbedding`` from ``foundry.models.embeddings``
    instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.cnn is deprecated. "
    "Import PatchCNNEmbedding from foundry.models.embeddings.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.temporal import (  # noqa: E402
    PatchCNNEmbedding as CNNEmbedding,
)

__all__ = ["CNNEmbedding"]
