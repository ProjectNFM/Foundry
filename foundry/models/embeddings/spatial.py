"""Backward-compatible shim for spatial.py module.

.. deprecated::
    Import from ``foundry.models.embeddings.channel`` instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.spatial is deprecated. "
    "Import from foundry.models.embeddings.channel instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.channel import (  # noqa: E402
    LinearSpatialProjector,
    PerceiverSpatialProjector,
    SessionSpatialProjector,
)

__all__ = [
    "LinearSpatialProjector",
    "SessionSpatialProjector",
    "PerceiverSpatialProjector",
]
