"""Latency metric and tracking utilities."""

from dataclasses import dataclass

from eval_lab.metrics.base import Metric, MetricConfig, MetricResult


@dataclass
class LatencyRecord:
    """Single sample latency in seconds."""

    seconds: float
    sample_index: int


def record_latency(latencies: list[float]) -> list[LatencyRecord]:
    """Wrap raw latency values into records with indices."""
    return [LatencyRecord(sec, i) for i, sec in enumerate(latencies)]


def aggregate_latency(records: list[LatencyRecord]) -> dict[str, float]:
    """Compute mean, p50, p95, p99 from latency records."""
    if not records:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_secs = sorted(r.seconds for r in records)
    n = len(sorted_secs)
    return {
        "mean": sum(sorted_secs) / n,
        "p50": sorted_secs[int(0.50 * n)] if n else 0.0,
        "p95": sorted_secs[int(0.95 * n)] if n else 0.0,
        "p99": sorted_secs[min(int(0.99 * n), n - 1)] if n else 0.0,
    }


class LatencyMetric(Metric):
    """Tracks inference latency. Aggregate returns mean latency in seconds."""

    @property
    def name(self) -> str:
        return "latency"

    def compute(
        self,
        predicted: str,
        expected: str | None,
        example_id: str,
        config: MetricConfig | None = None,
        extra_context: dict | None = None,
    ) -> MetricResult:
        latency_seconds = extra_context.get("latency_seconds") if extra_context else None
        if latency_seconds is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                example_id=example_id,
                details={"reason": "no_latency_provided"},
            )
        return MetricResult(
            metric_name=self.name,
            score=latency_seconds,
            example_id=example_id,
            details={"latency_seconds": latency_seconds},
        )

    def aggregate(self, results: list[MetricResult]) -> float:
        if not results:
            return 0.0
        valid = [r for r in results if r.details and "latency_seconds" in r.details]
        if not valid:
            return 0.0
        return sum(r.details["latency_seconds"] for r in valid) / len(valid)
