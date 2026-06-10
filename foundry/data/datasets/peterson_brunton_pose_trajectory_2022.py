from pathlib import Path

from brainsets.datasets import (
    PetersonBruntonPoseTrajectory2022 as _BrainsetsPetersonBrunton,
)

from foundry.tasks.config import TaskConfig

from .mixins import TaskMixin

_TASKS_DIR = Path(__file__).resolve().parents[3] / "configs" / "tasks"


def _load_ajile_tasks() -> dict[str, TaskConfig]:
    try:
        from brainsets.ajile_behavior_labels import (  # noqa: F401
            ACTIVE_BEHAVIOR_LABELS,
        )
    except ImportError:
        return {}

    return {
        "ajile_active_behavior": TaskConfig.from_yaml(
            _TASKS_DIR / "ajile_active_behavior.yaml"
        ),
        "ajile_inactive_active": TaskConfig.from_yaml(
            _TASKS_DIR / "ajile_inactive_active.yaml"
        ),
        "ajile_pose_estimation": TaskConfig.from_yaml(
            _TASKS_DIR / "ajile_pose_estimation.yaml"
        ),
    }


class PetersonBruntonPoseTrajectory2022(TaskMixin, _BrainsetsPetersonBrunton):
    """Foundry wrapper for AJILE with task-config registration."""

    AVAILABLE_TASKS = _load_ajile_tasks()
