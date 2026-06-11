"""Readout router for dispatching backbone embeddings to task heads."""

import torch
import torch.nn as nn


class ReadoutRouter(nn.Module):
    """Routes output embeddings to task-specific readout heads.

    For single-task runs (the common case), the router can skip index masking entirely.

    Args:
        heads: Mapping from task name to readout head module (typically
            :class:`~foundry.tasks.heads.ReadoutHead`). Task indices are
            assigned in sorted name order.

    Shape:
        - ``output_embs``: ``(N, embed_dim)``
        - ``task_index``: ``(N,)`` integer tensor; required when
            ``num_tasks > 1``. Ignored for single-task routers.
    """

    def __init__(self, heads: dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)
        self._task_names = sorted(heads.keys())
        self._name_to_idx = {n: i for i, n in enumerate(self._task_names)}
        self._single_task = len(heads) == 1

    def forward(
        self,
        output_embs: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Route embeddings to the appropriate readout heads.

        Returns:
            Dict mapping task name to head output. Tasks with no tokens in
            the current batch are omitted from the dict.
        """
        if self._single_task:
            name = self._task_names[0]
            return {name: self.heads[name](output_embs)}

        outputs = {}
        for idx, name in enumerate(self._task_names):
            mask = task_index == idx
            if not mask.any():
                continue
            outputs[name] = self.heads[name](output_embs[mask])
        return outputs

    def get_task_index_by_name(self, name: str) -> int:
        """Return the integer task index for ``name``.

        Indices follow sorted task name order (e.g. ``"alpha"`` before
        ``"zebra"``).
        """
        return self._name_to_idx[name]

    @property
    def num_tasks(self) -> int:
        """Number of registered task heads."""
        return len(self._task_names)
