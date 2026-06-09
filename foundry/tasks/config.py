"""Hydra-composable task configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class TaskConfig:
    """Hydra-instantiable task configuration.

    Bundles readout head, target extractor, loss, and metrics for a single
    training task. Dict fields use Hydra ``_target_`` so components can be
    built with :func:`hydra.utils.instantiate`.
    """

    name: str
    head: dict[str, Any]
    target_extractor: dict[str, Any]
    loss: dict[str, Any]
    metrics: dict[str, Any] | None = None
    class_names: list[str] | None = None
    metric_summary_modes: dict[str, str] = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        return self.head.get("output_dim", self.head.get("num_classes", 1))

    @property
    def kind(self) -> str:
        if "CrossEntropy" in self.loss.get("_target_", ""):
            return "binary" if self.output_dim == 2 else "multiclass"
        return "continuous"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        metrics = data.get("metrics")
        class_names = data.get("class_names")
        return cls(
            name=data["name"],
            head=dict(data["head"]),
            target_extractor=dict(data["target_extractor"]),
            loss=dict(data["loss"]),
            metrics=dict(metrics) if metrics is not None else None,
            class_names=list(class_names) if class_names is not None else None,
            metric_summary_modes=dict(data.get("metric_summary_modes") or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert isinstance(data, dict)
        return cls.from_dict(data)
