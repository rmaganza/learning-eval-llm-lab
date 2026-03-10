"""Base classes for evaluation datasets."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class BaseTask(Protocol):
    """Protocol for task-based evaluation (get_items, format_prompt, extract_answer)."""

    name: str

    async def get_items(self) -> AsyncIterator[Any]:
        """Yield evaluation items."""
        ...

    def format_prompt(self, item: Any) -> str:
        """Build prompt for the model."""
        ...

    def extract_answer(self, item: Any, response: str) -> Any:
        """Parse model response for metric comparison."""
        ...


class DatasetConfig(BaseModel):
    """Configuration for loading a dataset."""

    name: str
    path: str | None = None
    split: str = "test"
    max_examples: int | None = Field(default=None, description="Cap examples for quick runs")


class EvalExample(BaseModel):
    """Single evaluation example with input and expected output."""

    example_id: str
    input_prompt: str
    expected_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None  # For slice analysis


class EvalDataset(ABC):
    """Abstract base for evaluation datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""
        ...

    @abstractmethod
    async def load(self, config: DatasetConfig) -> list[EvalExample]:
        """Load examples. Raises DatasetLoadError on failure."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of loaded examples."""
        ...


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""

    def __init__(self, message: str, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        super().__init__(f"{dataset_name}: {message}")
