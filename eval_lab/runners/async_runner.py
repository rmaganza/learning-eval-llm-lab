"""Async evaluation runner."""

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

from eval_lab.datasets.base import DatasetConfig, EvalDataset, EvalExample
from eval_lab.metrics.base import Metric, MetricResult
from eval_lab.models.base import BaseModelAdapter, ModelAdapter, ModelConfig

if TYPE_CHECKING:
    from eval_lab.storage.types import EvalStore


class EvalRunConfig(BaseModel):
    """Configuration for an evaluation run."""

    dataset_config: DatasetConfig
    model_cfg: ModelConfig
    metric_names: list[str] = Field(default_factory=lambda: ["exact_match", "f1"])
    concurrency: int = 4
    judge_mode: str = "numeric"  # "numeric" (1-5) or "binary" (pass/fail)


class EvalRunResult(BaseModel):
    """Aggregated result from an evaluation run."""

    run_id: str | None = None
    dataset_name: str
    model_id: str
    metric_scores: dict[str, float] = Field(default_factory=dict)
    per_example_results: list[dict] = Field(default_factory=dict)
    total_examples: int = 0
    failed_examples: int = 0
    failed_errors: list[dict] = Field(
        default_factory=list,
        description="List of {example_id, error} for failed inferences",
    )


class AsyncEvalRunner:
    """Runs evaluation over dataset with async model inference."""

    def __init__(
        self,
        dataset: EvalDataset,
        model_adapter: ModelAdapter,
        metrics: Sequence[Metric],
        judge_adapter: BaseModelAdapter | None = None,
    ) -> None:
        self._dataset = dataset
        self._model = model_adapter
        self._metrics = list(metrics)
        self._judge_adapter = judge_adapter

    async def run(
        self,
        config: EvalRunConfig,
        run_id: str | None = None,
        store: "EvalStore | None" = None,
    ) -> EvalRunResult:
        """Execute evaluation. Loads dataset, runs inference, computes metrics.

        If store is provided, persists the result after completion.
        run_id defaults to a new UUID when not provided.
        """
        if run_id is None:
            run_id = str(uuid4())
        examples = await self._dataset.load(config.dataset_config)
        if not examples:
            return EvalRunResult(
                run_id=run_id,
                dataset_name=config.dataset_config.name,
                model_id=config.model_cfg.model_id,
                total_examples=0,
            )
        semaphore = asyncio.Semaphore(config.concurrency)
        per_example: list[dict] = []
        failed_errors: list[dict] = []

        async def process_example(example: EvalExample) -> dict | None:
            async with semaphore:
                try:
                    response = await self._model.generate(
                        example.input_prompt,
                        config.model_cfg,
                    )
                except Exception as e:
                    return {"example_id": example.example_id, "error": f"{type(e).__name__}: {e}"}
                if response.error:
                    return {"example_id": example.example_id, "error": response.error}
                predicted = response.generated_text
                if hasattr(self._dataset, "post_process"):
                    predicted = self._dataset.post_process(example, predicted)
                extra_context = {
                    "latency_seconds": response.latency_seconds,
                }
                if (
                    "llm_judge" in config.metric_names
                    and self._judge_adapter is not None
                ):
                    from eval_lab.metrics.llm_judge import llm_judge

                    mode = config.judge_mode if config.judge_mode in ("numeric", "binary") else "numeric"
                    judge_score = 0.0
                    async for s, _ in llm_judge(
                        self._judge_adapter,
                        [example.input_prompt],
                        [predicted],
                        [example.expected_output] if example.expected_output else None,
                        mode=mode,
                    ):
                        judge_score = s
                        break
                    extra_context["llm_judge_score"] = judge_score
                metric_results: list[MetricResult] = []
                for metric in self._metrics:
                    if metric.name not in config.metric_names:
                        continue
                    result = metric.compute(
                        predicted=predicted,
                        expected=example.expected_output,
                        example_id=example.example_id,
                        extra_context=extra_context,
                    )
                    metric_results.append(result)
                return {
                    "example_id": example.example_id,
                    "predicted": predicted,
                    "raw_output": response.generated_text,
                    "latency_seconds": response.latency_seconds,
                    "category": example.category,
                    "metric_results": [r.model_dump() for r in metric_results],
                }

        tasks = [process_example(ex) for ex in examples]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed_errors.append({
                    "example_id": examples[i].example_id,
                    "error": f"{type(outcome).__name__}: {outcome}",
                })
                continue
            if outcome is not None:
                if "error" in outcome:
                    failed_errors.append(outcome)
                else:
                    per_example.append(outcome)

        metric_scores: dict[str, float] = {}
        for metric in self._metrics:
            if metric.name not in config.metric_names:
                continue
            results = [
                MetricResult(**mr)
                for o in per_example
                for mr in o.get("metric_results", [])
                if mr.get("metric_name") == metric.name
            ]
            metric_scores[metric.name] = metric.aggregate(results)

        result = EvalRunResult(
            run_id=run_id,
            dataset_name=config.dataset_config.name,
            model_id=config.model_cfg.model_id,
            metric_scores=metric_scores,
            per_example_results=per_example,
            total_examples=len(examples),
            failed_examples=len(failed_errors),
            failed_errors=failed_errors,
        )
        if store is not None:
            await store.init_db()
            await store.save_run(result)
        return result
