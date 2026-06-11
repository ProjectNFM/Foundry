from collections.abc import Callable

from foundry.tasks.config import TaskConfig


class TaskMixin:
    """Mixin for datasets that declare which tasks they support."""

    AVAILABLE_TASKS: dict[str, TaskConfig] = {}
    TASK_TO_READOUT: dict[str, list[str]] = {}

    @classmethod
    def get_task(cls, name: str) -> TaskConfig:
        if name not in cls.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{name}'. Available: {list(cls.AVAILABLE_TASKS)}"
            )
        return cls.AVAILABLE_TASKS[name]

    @classmethod
    def get_tasks(cls, names: list[str] | None = None) -> dict[str, TaskConfig]:
        if names is None:
            return dict(cls.AVAILABLE_TASKS)
        return {n: cls.get_task(n) for n in names}

    @classmethod
    def get_tasks_for_experiment(cls, task_type: str) -> dict[str, TaskConfig]:
        task_names = cls.TASK_TO_READOUT[task_type]
        return cls.get_tasks(task_names)

    @classmethod
    def get_required_transforms(cls, task_type: str) -> list[Callable]:
        return []
