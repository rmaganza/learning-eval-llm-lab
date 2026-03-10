"""F1 score metric (token overlap)."""

import re

from eval_lab.metrics.base import Metric, MetricConfig, MetricResult


def _tokenize(text: str, normalize: bool = True) -> set[str]:
    if normalize:
        text = text.strip().lower()
    tokens = set(re.findall(r"\b\w+\b", text, re.UNICODE))
    return tokens


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1: overlap of word tokens between prediction and reference.
    Returns 0.0 if either side has no tokens.
    """
    pred_toks = _tokenize(prediction)
    ref_toks = _tokenize(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = pred_toks & ref_toks
    if not overlap:
        return 0.0
    prec = len(overlap) / len(pred_toks)
    rec = len(overlap) / len(ref_toks)
    return 2 * prec * rec / (prec + rec)


def token_f1_scores(
    predictions: list[str], references: list[str]
) -> list[float]:
    """Compute token F1 for each (pred, ref) pair."""
    if len(predictions) != len(references):
        return []
    return [token_f1(p, r) for p, r in zip(predictions, references)]


class F1Metric(Metric):
    """F1 score based on token overlap (precision/recall)."""

    @property
    def name(self) -> str:
        return "f1"

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
        pred_tokens = _tokenize(predicted, normalize)
        exp_tokens = _tokenize(expected, normalize)
        if not exp_tokens:
            return MetricResult(
                metric_name=self.name,
                score=1.0 if not pred_tokens else 0.0,
                example_id=example_id,
                details={"precision": 0.0, "recall": 0.0, "overlap": 0},
            )
        overlap = pred_tokens & exp_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(overlap) / len(exp_tokens)
        score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return MetricResult(
            metric_name=self.name,
            score=score,
            example_id=example_id,
            details={
                "precision": precision,
                "recall": recall,
                "overlap": len(overlap),
            },
        )

    def aggregate(self, results: list[MetricResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
