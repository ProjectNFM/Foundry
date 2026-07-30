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
    target_extractor: dict[str, Any] | None
    loss: dict[str, Any]
    metrics: dict[str, Any] | None = None
    class_names: list[str] | None = None
    metric_summary_modes: dict[str, str] = field(default_factory=dict)
    class_mapping: ClassificationMapping | None = None
    _extractor: Any = field(init=False, default=None, repr=False, compare=False)

    @property
    def output_dim(self) -> int:
        if self.class_mapping is not None:
            return self.class_mapping.num_classes
        return self.head.get("output_dim", self.head.get("num_classes", 1))

    @property
    def kind(self) -> str:
        if "CrossEntropy" in self.loss.get("_target_", ""):
            return "binary" if self.output_dim == 2 else "multiclass"
        return "continuous"

    def get_class_names(self) -> list[str] | None:
        """Canonical class names: mapping-derived takes priority over field."""
        if self.class_mapping is not None:
            return self.class_mapping.class_names
        return self.class_names

    @property
    def extractor(self) -> TargetExtractor | None:
        """Fully-wired extractor instance, or ``None`` for SSL tasks.

        Uses ``_target_`` from the YAML spec for polymorphism (so you can swap
        in a different TargetExtractor subclass), but consumers never need to
        worry about manually injecting the class_mapping.
        """
        if self.target_extractor is None:
            return None

        if self._extractor is None:
            from foundry.tasks.targets import TargetExtractor

            ext_cfg = dict(self.target_extractor)
            target_cls = ext_cfg.pop("_target_", None)

            if self.class_mapping is not None:
                ext_cfg["class_mapping"] = self.class_mapping

            if target_cls is not None and target_cls != (
                "foundry.tasks.targets.TargetExtractor"
            ):
                from hydra.utils import get_class as hydra_get_class

                cls = hydra_get_class(target_cls)
                instance = cls(**ext_cfg)
            else:
                instance = TargetExtractor(**ext_cfg)

            object.__setattr__(self, "_extractor", instance)
        return self._extractor

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        metrics = data.get("metrics")
        class_names = data.get("class_names")

        mapping_data = data.get("class_mapping")
        mapping = (
            ClassificationMapping.from_dict(mapping_data)
            if mapping_data is not None
            else None
        )

        if mapping is not None and metrics is not None:
            metrics = dict(metrics)
            declared = metrics.get("num_classes")
            derived = mapping.num_classes
            if declared is not None and declared != derived:
                raise ValueError(
                    f"metrics.num_classes ({declared}) conflicts with "
                    f"class_mapping.num_classes ({derived}). "
                    f"Remove metrics.num_classes and let it be derived "
                    f"automatically, or fix the class_mapping."
                )
            metrics["num_classes"] = derived
        elif metrics is not None:
            metrics = dict(metrics)

        raw_extractor = data.get("target_extractor")
        target_extractor = (
            dict(raw_extractor) if raw_extractor is not None else None
        )

        return cls(
            name=data["name"],
            head=dict(data["head"]),
            target_extractor=target_extractor,
            loss=dict(data["loss"]),
            metrics=metrics,
            class_names=list(class_names) if class_names is not None else None,
            metric_summary_modes=dict(data.get("metric_summary_modes") or {}),
            class_mapping=mapping,
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
