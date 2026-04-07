"""Backward-compatible shim for linear.py module.

.. deprecated::
    Import ``PatchLinearEmbedding`` from ``foundry.models.embeddings.temporal``
    or use the alias ``LinearEmbedding`` from ``foundry.models.embeddings``
    instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.linear is deprecated. "
    "Import PatchLinearEmbedding from foundry.models.embeddings.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.temporal import (  # noqa: E402
    PatchLinearEmbedding as LinearEmbedding,
)

__all__ = ["LinearEmbedding"]
