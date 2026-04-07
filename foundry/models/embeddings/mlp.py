"""Backward-compatible shim for mlp.py module.

.. deprecated::
    Import ``PatchMLPEmbedding`` from ``foundry.models.embeddings.temporal``
    or use the alias ``MLPEmbedding`` from ``foundry.models.embeddings``
    instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.mlp is deprecated. "
    "Import PatchMLPEmbedding from foundry.models.embeddings.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.temporal import (  # noqa: E402
    PatchMLPEmbedding as MLPEmbedding,
)

__all__ = ["MLPEmbedding"]
