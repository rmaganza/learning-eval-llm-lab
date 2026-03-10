"""LLM-as-judge: 1-5 or pass/fail scoring."""

from collections.abc import AsyncIterator
from typing import Literal

from eval_lab.metrics.base import Metric, MetricComputeError, MetricConfig, MetricResult
from eval_lab.models.base import BaseModelAdapter

JudgeMode = Literal["numeric", "binary"]


def _parse_numeric(raw: str) -> float:
    """Extract 1-5 score from judge output. Returns 0.0 on parse failure."""
    if not raw:
        return 0.0
    s = raw.strip().lower()
    for i in range(5, 0, -1):
        if str(i) in s:
            return float(i)
    return 0.0


def _parse_binary(raw: str) -> float:
    """Extract pass/fail as 1.0/0.0. Returns 0.0 on ambiguity."""
    if not raw:
        return 0.0
    s = raw.strip().lower()
    if any(w in s for w in ("pass", "yes", "correct", "true")):
        return 1.0
    if any(w in s for w in ("fail", "no", "incorrect", "false")):
        return 0.0
    return 0.0


def _build_judge_prompt(
    prompt: str, output: str, reference: str | None, mode: JudgeMode
) -> str:
    ref_part = f"\nReference: {reference}" if reference else ""
    instr = (
        "Score the response from 1 (worst) to 5 (best). Reply with only the number."
        if mode == "numeric"
        else "Reply with only PASS or FAIL."
    )
    return f"Prompt: {prompt}\nResponse: {output}{ref_part}\n\n{instr}"


async def llm_judge(
    model: BaseModelAdapter,
    prompts: list[str],
    outputs: list[str],
    references: list[str] | None = None,
    mode: JudgeMode = "numeric",
) -> AsyncIterator[tuple[float, float]]:
    """
    Use an LLM to score each (prompt, output) pair.
    Yields (score, latency_seconds) per sample.
    mode: 'numeric' (1-5) or 'binary' (pass/fail).
    """
    refs = references if references and len(references) == len(prompts) else [None] * len(prompts)
    judge_prompts = [
        _build_judge_prompt(p, o, r, mode)
        for p, o, r in zip(prompts, outputs, refs)
    ]
    parse_fn = _parse_numeric if mode == "numeric" else _parse_binary
    async for raw, lat in model.generate(judge_prompts):
        yield parse_fn(raw), lat


class LLMJudgeMetric(Metric):
    """LLM-as-judge scoring. Requires runner to pass judge_adapter; score is in extra_context['llm_judge_score']."""

    @property
    def name(self) -> str:
        return "llm_judge"

    def compute(
        self,
        predicted: str,
        expected: str | None,
        example_id: str,
        config: MetricConfig | None = None,
        extra_context: dict | None = None,
    ) -> MetricResult:
        score = (
            extra_context.get("llm_judge_score")
            if extra_context
            else None
        )
        if score is None:
            raise MetricComputeError(
                "llm_judge requires judge_adapter to be passed to the runner; "
                "use --judge-model (CLI) or judge_model_id (API)",
                self.name,
            )
        return MetricResult(
            metric_name=self.name,
            score=float(score),
            example_id=example_id,
            details={"llm_judge_score": score},
        )

    def aggregate(self, results: list[MetricResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
