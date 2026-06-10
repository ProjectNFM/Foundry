from foundry.tasks.config import TaskConfig


class TaskMixin:
    """Mixin for datasets that declare which tasks they support."""

    AVAILABLE_TASKS: dict[str, TaskConfig] = {}

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
