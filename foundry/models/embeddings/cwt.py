"""Backward-compatible shim for cwt.py module.

.. deprecated::
    Import from ``foundry.models.embeddings.temporal`` instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.cwt is deprecated. "
    "Import from foundry.models.embeddings.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.temporal import (  # noqa: E402
    CWTEmbedding,
    ContinuousCWTLayer,
)

__all__ = ["ContinuousCWTLayer", "CWTEmbedding"]
