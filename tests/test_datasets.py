"""Tests for datasets."""

import pytest

from eval_lab.datasets.base import DatasetConfig, EvalDataset, EvalExample
from eval_lab.datasets.registry import DatasetRegistry

_TEST_REGISTRY_KEY = "dummy_test"


class DummyDataset(EvalDataset):
    def __init__(self) -> None:
        self._examples: list[EvalExample] = []

    @property
    def name(self) -> str:
        return _TEST_REGISTRY_KEY

    async def load(self, config: DatasetConfig) -> list[EvalExample]:
        self._examples = [
            EvalExample(example_id="1", input_prompt="q1", expected_output="a1"),
        ]
        if config.max_examples:
            self._examples = self._examples[: config.max_examples]
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)


@pytest.fixture
def registered_dummy():
    """Register DummyDataset for tests and clean up afterward."""
    DatasetRegistry.register(_TEST_REGISTRY_KEY)(DummyDataset)
    yield
    DatasetRegistry.registry.pop(_TEST_REGISTRY_KEY, None)


class TestDatasetRegistry:
    def test_register_and_get(self, registered_dummy) -> None:
        cls = DatasetRegistry.get(_TEST_REGISTRY_KEY)
        assert cls is DummyDataset

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            DatasetRegistry.get("nonexistent")

    def test_list_datasets(self, registered_dummy) -> None:
        names = DatasetRegistry.list_datasets()
        assert _TEST_REGISTRY_KEY in names


class TestEvalExample:
    def test_example_creation(self) -> None:
        ex = EvalExample(
            example_id="1",
            input_prompt="What is 2+2?",
            expected_output="4",
        )
        assert ex.example_id == "1"
        assert ex.expected_output == "4"
