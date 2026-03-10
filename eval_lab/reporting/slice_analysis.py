"""Slice analysis: group results by task category."""

from collections import defaultdict

from eval_lab.runners.async_runner import EvalRunResult


def compute_slices(result: EvalRunResult) -> dict[str, dict[str, float]]:
    """
    Group per-example results by category and compute metric scores per slice.

    Returns dict mapping category -> {metric_name: aggregate_score}.
    Examples without a category are grouped under "_default".
    """
    by_category: dict[str, list[dict]] = defaultdict(list)
    for ex in result.per_example_results:
        cat = ex.get("category") or "_default"
        by_category[cat].append(ex)

    slices: dict[str, dict[str, float]] = {}
    for cat, examples in by_category.items():
        metric_sums: dict[str, list[float]] = defaultdict(list)
        for ex in examples:
            for mr in ex.get("metric_results", []):
                name = mr.get("metric_name")
                score = mr.get("score")
                if name is not None and score is not None:
                    metric_sums[name].append(float(score))
        slices[cat] = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in metric_sums.items()
        }
    return dict(slices)


def slice_summary(result: EvalRunResult) -> dict:
    """Return slice analysis plus overall scores for comparison."""
    slices = compute_slices(result)
    return {
        "run_id": result.run_id,
        "dataset": result.dataset_name,
        "model_id": result.model_id,
        "overall": result.metric_scores,
        "slices": slices,
        "slice_counts": {
            cat: len(
                [e for e in result.per_example_results if (e.get("category") or "_default") == cat]
            )
            for cat in slices
        },
    }
