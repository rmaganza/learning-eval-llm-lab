"""Base classes for evaluation metrics."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    """Configuration for metric computation."""

    name: str
    normalize: bool = True


class MetricResult(BaseModel):
    """Result from a single metric computation."""

    metric_name: str
    score: float
    example_id: str
    details: dict | None = Field(default=None, description="Per-example breakdown")


class Metric(ABC):
    """Abstract base for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric identifier."""
        ...

    @abstractmethod
    def compute(
        self,
        predicted: str,
        expected: str | None,
        example_id: str,
        config: MetricConfig | None = None,
        extra_context: dict | None = None,
    ) -> MetricResult:
        """Compute metric for one example. extra_context carries runner data (e.g. latency)."""
        ...

    @abstractmethod
    def aggregate(self, results: list[MetricResult]) -> float:
        """Aggregate per-example results into a single score."""
        ...


class MetricComputeError(Exception):
    """Raised when metric computation fails."""

    def __init__(self, message: str, metric_name: str) -> None:
        self.metric_name = metric_name
        super().__init__(f"{metric_name}: {message}")
