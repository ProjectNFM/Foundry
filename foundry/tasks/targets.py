from dataclasses import dataclass

import numpy as np
from temporaldata import Data


def _get_nested(data: Data, key: str):
    """Navigate a dot-separated key path into a Data object."""
    obj = data
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True)
class TargetExtractor:
    """Extracts targets from a Data sample during tokenization.

    A pure data transform — no nn.Module, no embed_dim, no GPU tensors.
    Testable in isolation with just a Data object.
    """

    timestamp_key: str
    value_key: str
    label_map: dict[int, int] | None = None

    def __call__(self, data: Data) -> dict:
        timestamps = _get_nested(data, self.timestamp_key)
        values = _get_nested(data, self.value_key)

        if self.label_map is not None:
            mapped = np.empty_like(values)
            for src, dst in self.label_map.items():
                mapped[values == src] = dst
            values = mapped

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}
