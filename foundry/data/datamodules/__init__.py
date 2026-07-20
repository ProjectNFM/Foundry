from .base import NeuralDataModule
from .physionet import PhysionetDataModule
from .ajile import AjileDataModule
from .neuroprobe import NeuroprobeDataModule
from .openneuro import OpenNeuroDataModule

__all__ = [
    "NeuralDataModule",
    "PhysionetDataModule",
    "AjileDataModule",
    "NeurosoftMinipigs2026DataModule",
    "NeurosoftMonkeys2026DataModule",
    "NeuroprobeDataModule",
    "OpenNeuroDataModule",
]


def __getattr__(name: str):
    if name == "NeurosoftMinipigs2026DataModule":
        from .neurosoft import NeurosoftMinipigs2026DataModule

        return NeurosoftMinipigs2026DataModule
    if name == "NeurosoftMonkeys2026DataModule":
        from .neurosoft import NeurosoftMonkeys2026DataModule

        return NeurosoftMonkeys2026DataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
