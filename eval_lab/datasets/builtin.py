"""Built-in datasets for quick testing."""

from eval_lab.datasets.base import DatasetConfig, EvalDataset, EvalExample
from eval_lab.datasets.registry import DatasetRegistry


@DatasetRegistry.register("example")
class ExampleDataset(EvalDataset):
    """Minimal example dataset for testing."""

    def __init__(self) -> None:
        self._examples: list[EvalExample] = []

    @property
    def name(self) -> str:
        return "example"

    async def load(self, config: DatasetConfig) -> list[EvalExample]:
        self._examples = [
            EvalExample(
                example_id="1",
                input_prompt="What is 2+2?",
                expected_output="4",
            ),
            EvalExample(
                example_id="2",
                input_prompt="Capital of France?",
                expected_output="Paris",
            ),
        ]
        if config.max_examples is not None:
            self._examples = self._examples[: config.max_examples]
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)
