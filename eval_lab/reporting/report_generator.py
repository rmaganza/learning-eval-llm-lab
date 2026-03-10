"""Report generator for evaluation results."""

from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from eval_lab.runners.async_runner import EvalRunResult


class ReportFormat:
    """Supported report formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


class ReportGenerator(ABC):
    """Abstract base for report generators."""

    @abstractmethod
    def generate(self, eval_result: EvalRunResult, output_path: Path | None) -> str:
        """Generate report. Returns content; writes to output_path if provided."""
        ...


class JsonReportGenerator(ReportGenerator):
    """Outputs evaluation results as JSON."""

    def generate(self, eval_result: EvalRunResult, output_path: Path | None) -> str:
        content = eval_result.model_dump_json(indent=2)
        if output_path:
            output_path.write_text(content, encoding="utf-8")
        return content


class MarkdownReportGenerator(ReportGenerator):
    """Outputs evaluation results as Markdown."""

    def generate(self, eval_result: EvalRunResult, output_path: Path | None) -> str:
        lines = [
            f"# Eval Report: {eval_result.dataset_name}",
            f"\n**Model:** {eval_result.model_id}",
            f"**Examples:** {eval_result.total_examples}",
            f"**Failed:** {eval_result.failed_examples}",
            "\n## Metrics\n",
        ]
        for metric_name, score in eval_result.metric_scores.items():
            lines.append(f"- **{metric_name}:** {score:.4f}")
        content = "\n".join(lines)
        if output_path:
            output_path.write_text(content, encoding="utf-8")
        return content


class HtmlReportGenerator(ReportGenerator):
    """Outputs evaluation results as HTML using Jinja2."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("eval_lab", "reporting/templates"),
            autoescape=select_autoescape(["html"]),
        )

    def generate(self, eval_result: EvalRunResult, output_path: Path | None) -> str:
        try:
            template = self._env.get_template("report.html")
        except Exception:
            template = self._env.from_string(self._fallback_template())
        content = template.render(
            dataset_name=eval_result.dataset_name,
            model_id=eval_result.model_id,
            total_examples=eval_result.total_examples,
            failed_examples=eval_result.failed_examples,
            metric_scores=eval_result.metric_scores,
            per_example_results=eval_result.per_example_results,
        )
        if output_path:
            output_path.write_text(content, encoding="utf-8")
        return content

    def _fallback_template(self) -> str:
        return """
<!DOCTYPE html>
<html>
<head><title>Eval Report</title></head>
<body>
  <h1>{{ dataset_name }} - {{ model_id }}</h1>
  <p>Examples: {{ total_examples }}, Failed: {{ failed_examples }}</p>
  <ul>
  {% for name, score in metric_scores.items() %}
    <li><strong>{{ name }}:</strong> {{ "%.4f"|format(score) }}</li>
  {% endfor %}
  </ul>
</body>
</html>
"""
