"""NeuralBench integration for Foundry via NeuralSet runtime adapter.

Requires the ``neuralbench`` optional dependency group::

    uv sync --group neuralbench
"""

from foundry.data.neuralbench.adapter import NeuralSetAdapter
from foundry.data.neuralbench.datamodule import NeuralBenchDataModule

__all__ = ["NeuralSetAdapter", "NeuralBenchDataModule"]
