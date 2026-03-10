"""Shared evaluation run logic for CLI, API, and scripts."""

import os
from typing import TYPE_CHECKING

from eval_lab.datasets import DatasetRegistry
from eval_lab.datasets.base import DatasetConfig
from eval_lab.metrics import ExactMatchMetric, F1Metric, LatencyMetric, LLMJudgeMetric
from eval_lab.models.base import ModelConfig
from eval_lab.models.openai_adapter import OpenAIAdapter
from eval_lab.runners.async_runner import AsyncEvalRunner, EvalRunConfig, EvalRunResult

if TYPE_CHECKING:
    from eval_lab.storage.types import EvalStore

_METRICS_REGISTRY = {
    "exact_match": ExactMatchMetric,
    "f1": F1Metric,
    "latency": LatencyMetric,
    "llm_judge": LLMJudgeMetric,
}


def get_metrics_by_names(names: list[str] | None) -> list:
    """Resolve metric names to metric instances. Returns default metrics if names is None/empty."""
    default = ["exact_match", "f1", "latency"]
    resolved = names or default
    return [_METRICS_REGISTRY[m]() for m in resolved if m in _METRICS_REGISTRY]


async def run_evaluation(
    dataset_name: str,
    model_id: str,
    *,
    max_examples: int | None = None,
    metric_names: list[str] | None = None,
    judge_model: str | None = None,
    judge_mode: str = "numeric",
    concurrency: int = 4,
    store: "EvalStore | None" = None,
    run_id: str | None = None,
) -> EvalRunResult:
    """
    Run evaluation on a dataset with a model.

    Shared entry point for CLI, API, and scripts.
    """
    dataset_cls = DatasetRegistry.get(dataset_name)
    dataset = dataset_cls()
    dataset_config = DatasetConfig(name=dataset_name, max_examples=max_examples)

    metrics = get_metrics_by_names(metric_names)
    names = [m.name for m in metrics]
    if judge_model and "llm_judge" not in names:
        names.append("llm_judge")
        metrics.append(LLMJudgeMetric())

    base_url = os.environ.get("EVAL_API_BASE_URL")
    api_key = os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    adapter = OpenAIAdapter(model=model_id, base_url=base_url, api_key=api_key)
    judge_adapter = (
        OpenAIAdapter(model=judge_model, base_url=base_url, api_key=api_key)
        if judge_model
        else None
    )

    try:
        runner = AsyncEvalRunner(dataset, adapter, metrics, judge_adapter=judge_adapter)
        config = EvalRunConfig(
            dataset_config=dataset_config,
            model_cfg=ModelConfig(
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
            ),
            metric_names=names,
            concurrency=concurrency,
            judge_mode=judge_mode,
        )
        return await runner.run(config, run_id=run_id, store=store)
    finally:
        await adapter.close()
        if judge_adapter:
            await judge_adapter.close()
