"""Adapter to use BaseTask as EvalDataset for the runner."""

from eval_lab.datasets.base import (
    BaseTask,
    DatasetConfig,
    DatasetLoadError,
    EvalDataset,
    EvalExample,
)
from eval_lab.datasets.registry import DatasetRegistry

_TASK_ITEM_MODELS: dict[str, type] = {}


def _item_model_for_task(task_name: str) -> type | None:
    """Resolve item model for task. Checks task.item_model first, then fallback mapping."""
    try:
        task = DatasetRegistry.get_task(task_name)
        item_cls = getattr(task, "item_model", None)
        if item_cls is not None:
            return item_cls
    except KeyError:
        pass
    if task_name in _TASK_ITEM_MODELS:
        return _TASK_ITEM_MODELS[task_name]
    try:
        from eval_lab.datasets.tasks.hallucination import HallucinationItem
        from eval_lab.datasets.tasks.instruction_following import InstructionItem
        from eval_lab.datasets.tasks.reasoning import ReasoningItem
        from eval_lab.datasets.tasks.summarization import SummarizationItem
        mapping = {
            "reasoning": ReasoningItem,
            "summarization": SummarizationItem,
            "hallucination": HallucinationItem,
            "instruction_following": InstructionItem,
        }
        _TASK_ITEM_MODELS.update(mapping)
        return mapping.get(task_name)
    except ImportError:
        return None


def _to_expected_str(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        import json
        return json.dumps(val, sort_keys=True)
    return str(val).strip().lower()


class TaskDatasetAdapter(EvalDataset):
    """Wraps a BaseTask so it can be used as an EvalDataset by the runner."""

    def __init__(self, task: BaseTask) -> None:
        self._task = task
        self._examples: list[EvalExample] = []

    @property
    def name(self) -> str:
        return self._task.name

    async def load(self, config: DatasetConfig) -> list[EvalExample]:
        self._examples = []
        count = 0
        try:
            async for item in self._task.get_items():
                if config.max_examples and count >= config.max_examples:
                    break
                prompt = self._task.format_prompt(item)
                expected = _to_expected_str(
                    getattr(item, "expected_answer", None)
                    or getattr(item, "reference_summary", None)
                    or getattr(item, "expected_label", None)
                    or getattr(item, "expected_schema", None)
                )
                ex_id = getattr(item, "id", str(count))
                meta = getattr(item, "metadata", None) or {}
                category = (meta.get("category") if isinstance(meta, dict) else None) or self._task.name
                item_dump = item.model_dump() if hasattr(item, "model_dump") else {"id": ex_id}
                self._examples.append(
                    EvalExample(
                        example_id=ex_id,
                        input_prompt=prompt,
                        expected_output=expected or None,
                        metadata={"_task": self._task.name, "_item": item_dump},
                        category=category,
                    )
                )
                count += 1
        except Exception as e:
            raise DatasetLoadError(str(e), self._task.name)
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)

    def post_process(self, example: EvalExample, raw_output: str) -> str:
        """Extract answer from raw model output using task's extract_answer."""
        meta = example.metadata or {}
        item_data = meta.get("_item", {})
        task_name = meta.get("_task", self._task.name)
        if not item_data:
            return raw_output.strip()
        item_cls = _item_model_for_task(task_name)
        if item_cls is None:
            return raw_output.strip()
        try:
            item = item_cls(**item_data)
        except Exception:
            return raw_output.strip()
        task = DatasetRegistry.get_task(task_name)
        extracted = task.extract_answer(item, raw_output)
        return _to_expected_str(extracted) if extracted is not None else raw_output.strip()
