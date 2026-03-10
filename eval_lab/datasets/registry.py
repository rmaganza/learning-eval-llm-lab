"""Registry for dataset implementations."""

from collections.abc import Callable
from typing import TypeVar

from eval_lab.datasets.base import BaseTask, EvalDataset

T = TypeVar("T", bound=type[EvalDataset])
TTask = TypeVar("TTask", bound=BaseTask)


class DatasetRegistry:
    """Registry mapping dataset names to implementations."""

    registry: dict[str, type[EvalDataset]] = {}
    task_registry: dict[str, BaseTask] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[T | TTask], T | TTask]:
        """Decorator to register a dataset or task. Use name=None for tasks (uses cls.name)."""

        def decorator(target: T | TTask) -> T | TTask:
            key = name if name is not None else getattr(target, "name", target.__name__)
            if hasattr(target, "load") and callable(getattr(target, "load", None)):
                cls.registry[key] = target  # type: ignore
            else:
                cls.task_registry[key] = target()  # type: ignore
                cls.registry[key] = cls._make_task_dataset_class(key)
            return target

        return decorator

    @classmethod
    def _make_task_dataset_class(cls, task_name: str) -> type[EvalDataset]:
        from eval_lab.datasets.task_adapter import TaskDatasetAdapter
        reg = cls

        class _TaskDataset(TaskDatasetAdapter):
            def __init__(self) -> None:
                super().__init__(reg.get_task(task_name))

        _TaskDataset.__name__ = f"TaskDataset_{task_name}"
        return _TaskDataset

    @classmethod
    def get(cls, name: str) -> type[EvalDataset]:
        """Get dataset class by name. Raises KeyError if not found."""
        if name not in cls.registry:
            available = ", ".join(cls.registry.keys()) or "none"
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return cls.registry[name]

    @classmethod
    def get_task(cls, name: str) -> BaseTask:
        """Get task instance by name."""
        if name not in cls.task_registry:
            available = ", ".join(cls.task_registry.keys()) or "none"
            raise KeyError(f"Task '{name}' not found. Available: {available}")
        return cls.task_registry[name]

    @classmethod
    def list_datasets(cls) -> list[str]:
        """Return registered dataset names."""
        return sorted(set(cls.registry.keys()) | set(cls.task_registry.keys()))

    @classmethod
    def list_tasks(cls) -> list[str]:
        """Return registered task names."""
        return sorted(cls.task_registry.keys())


register = DatasetRegistry.register
