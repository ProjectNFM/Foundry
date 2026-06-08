import torch
import torch.nn as nn


class ReadoutRouter(nn.Module):
    """Routes output embeddings to task-specific ReadoutHeads.

    Replaces MultitaskReadout. For single-task runs (the common case),
    use the fast path that skips index masking entirely.
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

    def task_index_for(self, name: str) -> int:
        return self._name_to_idx[name]

    @property
    def num_tasks(self) -> int:
        return len(self._task_names)
