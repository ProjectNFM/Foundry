"""Hydra-composable task configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from foundry.tasks.classification_mapping import ClassificationMapping

if TYPE_CHECKING:
    from foundry.tasks.targets import TargetExtractor


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
    classification_mapping: ClassificationMapping | None = None

    @property
    def output_dim(self) -> int:
        if self.classification_mapping is not None:
            return self.classification_mapping.num_classes
        return self.head.get("output_dim", self.head.get("num_classes", 1))

    @property
    def kind(self) -> str:
        if "CrossEntropy" in self.loss.get("_target_", ""):
            return "binary" if self.output_dim == 2 else "multiclass"
        return "continuous"

    def get_class_names(self) -> list[str] | None:
        """Canonical class names: mapping-derived takes priority over field."""
        if self.classification_mapping is not None:
            return self.classification_mapping.class_names
        return self.class_names

    def build_extractor(self) -> TargetExtractor:
        """Build a fully-wired :class:`TargetExtractor` for this task.

        Strips the Hydra ``_target_`` key from the stored extractor dict and
        injects ``classification_mapping`` when present, so callers never need
        to perform this wiring themselves.
        """
        from foundry.tasks.targets import TargetExtractor

        kwargs = dict(self.target_extractor)
        kwargs.pop("_target_", None)
        if self.classification_mapping is not None:
            kwargs["classification_mapping"] = self.classification_mapping
        return TargetExtractor(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        metrics = data.get("metrics")
        class_names = data.get("class_names")

        mapping_data = data.get("classification_mapping")
        mapping = (
            ClassificationMapping.from_dict(mapping_data)
            if mapping_data is not None
            else None
        )

        target_extractor = dict(data["target_extractor"])

        return cls(
            name=data["name"],
            head=dict(data["head"]),
            target_extractor=target_extractor,
            loss=dict(data["loss"]),
            metrics=dict(metrics) if metrics is not None else None,
            class_names=list(class_names) if class_names is not None else None,
            metric_summary_modes=dict(data.get("metric_summary_modes") or {}),
            classification_mapping=mapping,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert isinstance(data, dict)
        return cls.from_dict(data)

    @staticmethod
    def normalize_task_configs(
        task_configs: dict[str, Any],
    ) -> dict[str, TaskConfig]:
        """Ensure all values are proper ``TaskConfig`` instances.

        Hydra's ``instantiate`` may wrap dataclass kwargs as OmegaConf
        structured configs, which strips ``@property`` attributes like
        ``output_dim``.  This helper converts any such wrappers back to
        real Python objects.
        """
        out: dict[str, TaskConfig] = {}
        for k, v in task_configs.items():
            if isinstance(v, TaskConfig):
                out[k] = v
            elif isinstance(v, DictConfig):
                out[k] = TaskConfig.from_dict(
                    OmegaConf.to_container(v, resolve=True)
                )
            elif isinstance(v, dict):
                out[k] = TaskConfig.from_dict(v)
            else:
                out[k] = v
        return out
