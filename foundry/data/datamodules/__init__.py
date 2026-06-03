from .base import NeuralDataModule
from .physionet import PhysionetDataModule
from .neurosoft import NeurosoftMinipigs2026DataModule
from .neurosoft import NeurosoftMonkeys2026DataModule

try:
    from .ajile import AjileDataModule
except ImportError:
    AjileDataModule = None  # type: ignore[misc, assignment]

__all__ = [
    "NeuralDataModule",
    "PhysionetDataModule",
    "NeurosoftMinipigs2026DataModule",
    "NeurosoftMonkeys2026DataModule",
]
if AjileDataModule is not None:
    __all__.append("AjileDataModule")
