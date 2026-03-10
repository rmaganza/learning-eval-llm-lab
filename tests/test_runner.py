"""Tests for the async evaluation runner."""

import pytest

from eval_lab.datasets.base import DatasetConfig, EvalDataset, EvalExample
from eval_lab.metrics.base import Metric, MetricResult
from eval_lab.models.base import ModelAdapter, ModelConfig, ModelResponse
from eval_lab.runners.async_runner import AsyncEvalRunner, EvalRunConfig


class MockDataset(EvalDataset):
    """Minimal dataset for testing."""

    def __init__(self) -> None:
        self._examples: list[EvalExample] = []

    @property
    def name(self) -> str:
        return "mock"

    async def load(self, config: DatasetConfig) -> list[EvalExample]:
        self._examples = [
            EvalExample(example_id="1", input_prompt="q1", expected_output="a1"),
            EvalExample(example_id="2", input_prompt="q2", expected_output="a2"),
        ]
        if config.max_examples is not None:
            self._examples = self._examples[: config.max_examples]
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)


class MockMetric(Metric):
    """Metric that returns fixed score."""

    @property
    def name(self) -> str:
        return "mock"

    def compute(self, predicted, expected, example_id, config=None, extra_context=None) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=1.0 if predicted == expected else 0.0,
            example_id=example_id,
        )

    def aggregate(self, results: list[MetricResult]) -> float:
        return sum(r.score for r in results) / len(results) if results else 0.0


class MockAdapter(ModelAdapter):
    """Adapter that returns predefined responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {"q1": "a1", "q2": "a2"}

    async def generate(self, prompt: str, config: ModelConfig) -> ModelResponse:
        text = self._responses.get(prompt, "")
        return ModelResponse(generated_text=text, latency_seconds=0.01)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_runner_basic() -> None:
    """Runner produces correct results for successful run."""
    dataset = MockDataset()
    adapter = MockAdapter()
    metrics = [MockMetric()]
    runner = AsyncEvalRunner(dataset, adapter, metrics)
    config = EvalRunConfig(
        dataset_config=DatasetConfig(name="mock", max_examples=2),
        model_cfg=ModelConfig(model_id="mock"),
        metric_names=["mock"],
        concurrency=2,
    )
    result = await runner.run(config, store=None)
    assert result.dataset_name == "mock"
    assert result.model_id == "mock"
    assert result.total_examples == 2
    assert result.failed_examples == 0
    assert result.metric_scores["mock"] == 1.0
    assert len(result.per_example_results) == 2


@pytest.mark.asyncio
async def test_runner_max_examples() -> None:
    """Runner respects max_examples."""
    dataset = MockDataset()
    adapter = MockAdapter()
    runner = AsyncEvalRunner(dataset, adapter, [MockMetric()])
    config = EvalRunConfig(
        dataset_config=DatasetConfig(name="mock", max_examples=1),
        model_cfg=ModelConfig(model_id="mock"),
        metric_names=["mock"],
    )
    result = await runner.run(config, store=None)
    assert result.total_examples == 1
    assert len(result.per_example_results) == 1


@pytest.mark.asyncio
async def test_runner_failure_tracking() -> None:
    """Runner tracks failed inferences with error details."""
    dataset = MockDataset()

    class FailingAdapter(ModelAdapter):
        async def generate(self, prompt: str, config: ModelConfig) -> ModelResponse:
            return ModelResponse(generated_text="", error="ConnectionError: failed")

        async def close(self) -> None:
            pass

    adapter = FailingAdapter()
    runner = AsyncEvalRunner(dataset, adapter, [MockMetric()])
    config = EvalRunConfig(
        dataset_config=DatasetConfig(name="mock", max_examples=2),
        model_cfg=ModelConfig(model_id="mock"),
        metric_names=["mock"],
    )
    result = await runner.run(config, store=None)
    assert result.total_examples == 2
    assert result.failed_examples == 2
    assert len(result.failed_errors) == 2
    assert "ConnectionError" in result.failed_errors[0]["error"]


@pytest.mark.asyncio
async def test_runner_empty_dataset() -> None:
    """Runner handles empty dataset."""
    dataset = MockDataset()
    adapter = MockAdapter()
    runner = AsyncEvalRunner(dataset, adapter, [MockMetric()])
    config = EvalRunConfig(
        dataset_config=DatasetConfig(name="mock", max_examples=0),
        model_cfg=ModelConfig(model_id="mock"),
        metric_names=["mock"],
    )
    result = await runner.run(config, store=None)
    assert result.total_examples == 0
    assert result.failed_examples == 0
    assert len(result.per_example_results) == 0
