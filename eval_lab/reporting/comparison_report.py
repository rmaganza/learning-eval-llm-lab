"""Model comparison report generator (Markdown and HTML)."""

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from eval_lab.reporting.slice_analysis import slice_summary
from eval_lab.runners.async_runner import EvalRunResult


def comparison_markdown(
    results: list[EvalRunResult],
    dataset_name: str | None = None,
    include_slices: bool = False,
) -> str:
    """Generate Markdown comparison report for multiple runs."""
    if not results:
        return "# Model Comparison\n\nNo runs to compare."
    dataset = dataset_name or results[0].dataset_name
    metric_names = sorted(set(m for r in results for m in r.metric_scores.keys()))
    lines = [
        f"# Model Comparison: {dataset}",
        "",
        "## Summary",
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|" + "-------|" * (len(metric_names) + 1),
    ]
    for r in results:
        scores = [r.metric_scores.get(m, 0.0) for m in metric_names]
        lines.append(f"| {r.model_id} | " + " | ".join(f"{s:.4f}" for s in scores) + " |")
    lines.append("")

    if include_slices:
        lines.append("## Slice Analysis")
        lines.append("")
        for r in results:
            summary = slice_summary(r)
            if summary.get("slices"):
                lines.append(f"### {r.model_id}")
                for cat, scores in summary["slices"].items():
                    lines.append(
                        f"- **{cat}** (n={summary['slice_counts'].get(cat, 0)}): "
                        + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
                    )
                lines.append("")

    return "\n".join(lines)


def comparison_html(
    results: list[EvalRunResult],
    dataset_name: str | None = None,
    include_slices: bool = True,
) -> str:
    """Generate HTML comparison report."""
    if not results:
        return "<html><body><h1>No runs to compare</h1></body></html>"
    dataset = dataset_name or results[0].dataset_name
    metric_names = sorted(set(m for r in results for m in r.metric_scores.keys()))
    rows = [
        {
            "run_id": r.run_id,
            "model_id": r.model_id,
            "metric_scores": r.metric_scores,
            "total_examples": r.total_examples,
            "failed_examples": r.failed_examples,
        }
        for r in results
    ]
    slice_data = [slice_summary(r) for r in results] if include_slices else []
    env = Environment(
        loader=PackageLoader("eval_lab", "reporting/templates"),
        autoescape=select_autoescape(["html"]),
    )
    try:
        template = env.get_template("comparison.html")
    except Exception:
        template = env.from_string(_comparison_fallback_html())
    return template.render(
        dataset_name=dataset,
        metric_names=metric_names,
        rows=rows,
        slice_data=slice_data,
    )


def _comparison_fallback_html() -> str:
    return """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Model Comparison</title></head>
<body>
  <h1>Model Comparison: {{ dataset_name }}</h1>
  <table border="1">
    <tr><th>Model</th>{% for m in metric_names %}<th>{{ m }}</th>{% endfor %}</tr>
    {% for r in rows %}
    <tr><td>{{ r.model_id }}</td>{% for m in metric_names %}<td>{{ "%.4f"|format(r.metric_scores.get(m, 0)) }}</td>{% endfor %}</tr>
    {% endfor %}
  </table>
</body>
</html>
"""


def write_comparison_report(
    results: list[EvalRunResult],
    output_path: Path,
    format: str = "markdown",
    dataset_name: str | None = None,
) -> str:
    """Write comparison report to file. Returns content."""
    if format.lower() in ("html", "htm"):
        content = comparison_html(results, dataset_name=dataset_name)
    else:
        content = comparison_markdown(results, dataset_name=dataset_name)
    output_path.write_text(content, encoding="utf-8")
    return content
