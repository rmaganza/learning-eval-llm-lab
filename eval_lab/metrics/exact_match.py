"""Exact match metric with normalization."""

import re
import unicodedata

from eval_lab.metrics.base import Metric, MetricConfig, MetricResult


def _normalize(s: str) -> str:
    """Normalize for comparison: NFKC, lowercase, strip, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s.strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(prediction: str, reference: str) -> bool:
    """Compare prediction to reference after normalization. Handles empty/unicode."""
    return _normalize(prediction) == _normalize(reference)


def exact_match_score(
    predictions: list[str], references: list[str]
) -> float:
    """Exact match accuracy over lists. Returns 0.0 if empty or length mismatch."""
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if exact_match(p, r))
    return matches / len(predictions)


class ExactMatchMetric(Metric):
    """Exact string match (case-insensitive, unicode-normalized when enabled)."""

    @property
    def name(self) -> str:
        return "exact_match"

    def _normalize(self, text: str) -> str:
        return _normalize(text)

    def compute(
        self,
        predicted: str,
        expected: str | None,
        example_id: str,
        config: MetricConfig | None = None,
        extra_context: dict | None = None,
    ) -> MetricResult:
        if expected is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                example_id=example_id,
                details={"reason": "no_expected_output"},
            )
        normalize = config.normalize if config else True
        pred_norm = self._normalize(predicted) if normalize else predicted.strip()
        exp_norm = self._normalize(expected) if normalize else expected.strip()
        score = 1.0 if pred_norm == exp_norm else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            example_id=example_id,
            details={"predicted": pred_norm, "expected": exp_norm},
        )

    def aggregate(self, results: list[MetricResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
