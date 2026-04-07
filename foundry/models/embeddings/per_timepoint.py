"""Backward-compatible shim for per_timepoint.py module.

.. deprecated::
    Import from ``foundry.models.embeddings.temporal`` instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.per_timepoint is deprecated. "
    "Import from foundry.models.embeddings.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.temporal import PerTimepointEmbedding  # noqa: E402

__all__ = ["PerTimepointEmbedding"]
