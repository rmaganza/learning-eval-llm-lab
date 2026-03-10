"""Dataset loading and registry."""

# Import tasks to trigger registration
from eval_lab.datasets import tasks  # noqa: F401
from eval_lab.datasets.base import BaseTask, DatasetConfig, EvalDataset, EvalExample
from eval_lab.datasets.builtin import ExampleDataset  # noqa: F401
from eval_lab.datasets.registry import DatasetRegistry

__all__ = [
    "BaseTask",
    "DatasetConfig",
    "EvalExample",
    "EvalDataset",
    "DatasetRegistry",
]
