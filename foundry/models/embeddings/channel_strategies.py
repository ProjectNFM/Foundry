"""Backward-compatible shim for channel_strategies.py module.

.. deprecated::
    Import from ``foundry.models.embeddings.channel`` instead.
"""

import warnings

warnings.warn(
    "foundry.models.embeddings.channel_strategies is deprecated. "
    "Import from foundry.models.embeddings.channel instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foundry.models.embeddings.channel import (  # noqa: E402
    ChannelStrategy,
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
)

__all__ = [
    "ChannelStrategy",
    "FixedChannelStrategy",
    "PerChannelStrategy",
    "SpatialProjectionStrategy",
]
