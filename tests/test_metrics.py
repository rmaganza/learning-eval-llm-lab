"""Tests for metrics."""

import pytest

from eval_lab.metrics.exact_match import ExactMatchMetric
from eval_lab.metrics.f1 import F1Metric
from eval_lab.metrics.latency import LatencyMetric


class TestExactMatchMetric:
    def test_exact_match_correct(self) -> None:
        metric = ExactMatchMetric()
        result = metric.compute("4", "4", "1")
        assert result.score == 1.0

    def test_exact_match_wrong(self) -> None:
        metric = ExactMatchMetric()
        result = metric.compute("5", "4", "1")
        assert result.score == 0.0

    def test_exact_match_normalized(self) -> None:
        metric = ExactMatchMetric()
        result = metric.compute("  PARIS  ", "Paris", "1")
        assert result.score == 1.0

    def test_exact_match_no_expected(self) -> None:
        metric = ExactMatchMetric()
        result = metric.compute("foo", None, "1")
        assert result.score == 0.0

    def test_aggregate(self) -> None:
        metric = ExactMatchMetric()
        results = [
            metric.compute("4", "4", "1"),
            metric.compute("5", "4", "2"),
        ]
        assert metric.aggregate(results) == 0.5


class TestF1Metric:
    def test_f1_perfect(self) -> None:
        metric = F1Metric()
        result = metric.compute("hello world", "hello world", "1")
        assert result.score == 1.0

    def test_f1_partial(self) -> None:
        metric = F1Metric()
        result = metric.compute("hello there", "hello world", "1")
        assert 0 < result.score < 1.0

    def test_f1_no_overlap(self) -> None:
        metric = F1Metric()
        result = metric.compute("foo bar", "baz qux", "1")
        assert result.score == 0.0

    def test_f1_no_expected(self) -> None:
        metric = F1Metric()
        result = metric.compute("foo", None, "1")
        assert result.score == 0.0


class TestLatencyMetric:
    def test_latency_with_value(self) -> None:
        metric = LatencyMetric()
        result = metric.compute(
            "output",
            "expected",
            "1",
            extra_context={"latency_seconds": 0.5},
        )
        assert result.score == 0.5
        assert result.details and result.details["latency_seconds"] == 0.5

    def test_latency_no_context(self) -> None:
        metric = LatencyMetric()
        result = metric.compute("output", "expected", "1")
        assert result.score == 0.0

    def test_aggregate(self) -> None:
        metric = LatencyMetric()
        r1 = metric.compute("a", "b", "1", extra_context={"latency_seconds": 0.2})
        r2 = metric.compute("a", "b", "2", extra_context={"latency_seconds": 0.4})
        assert metric.aggregate([r1, r2]) == pytest.approx(0.3)
