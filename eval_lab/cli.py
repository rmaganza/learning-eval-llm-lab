"""CLI entrypoint for running evaluations."""

import asyncio
import os

import typer
from rich.console import Console

from eval_lab.datasets import DatasetRegistry
from eval_lab.run import run_evaluation
from eval_lab.storage.store import get_store

app = typer.Typer()
console = Console()


def _should_persist() -> bool:
    return os.environ.get("EVAL_LAB_PERSIST", "true").lower() == "true"


@app.command()
def run(
    dataset: str = typer.Option("example", "--dataset", "-d", help="Dataset or task name"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Model ID (OpenAI-compatible)"),
    max_examples: int | None = typer.Option(None, "--max-examples", help="Cap examples for quick runs"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrent inference requests"),
    metrics: str = typer.Option("exact_match,f1,latency", "--metrics", help="Comma-separated metrics"),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model for LLM-as-judge (enables llm_judge)"),
    judge_mode: str = typer.Option("numeric", "--judge-mode", help="Judge mode: numeric (1-5) or binary (pass/fail)"),
    config_file: str | None = typer.Option(None, "--config", help="Path to YAML config (overrides CLI args)"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip saving results to DB"),
    report: str | None = typer.Option(None, "--report", "-r", help="Output path for report (.md or .html)"),
) -> None:
    """Run evaluation on a dataset."""
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    if judge_model and "llm_judge" not in metric_names:
        metric_names.append("llm_judge")

    if config_file:
        from eval_lab.config import load_config

        cfg = load_config(config_file)
        dataset = cfg.dataset.name
        model = cfg.model.model_id
        max_examples = cfg.dataset.max_examples
        concurrency = cfg.runner.get("concurrency", 4)

    store = get_store() if (not no_persist and _should_persist()) else None
    result = asyncio.run(
        run_evaluation(
            dataset,
            model,
            max_examples=max_examples,
            metric_names=metric_names,
            judge_model=judge_model,
            judge_mode=judge_mode,
            concurrency=concurrency,
            store=store,
        )
    )

    console.print(f"[green]Dataset:[/green] {result.dataset_name}")
    console.print(f"[green]Model:[/green] {result.model_id}")
    if result.run_id:
        console.print(f"[green]Run ID:[/green] {result.run_id}")
    for name, score in result.metric_scores.items():
        console.print(f"  {name}: {score:.4f}")

    if report:
        from pathlib import Path

        from eval_lab.reporting.report_generator import HtmlReportGenerator, MarkdownReportGenerator

        path = Path(report)
        if path.suffix.lower() in (".html", ".htm"):
            HtmlReportGenerator().generate(result, path)
        else:
            MarkdownReportGenerator().generate(result, path)
        console.print(f"[green]Report:[/green] {path}")


@app.command()
def list_datasets() -> None:
    """List available datasets and tasks."""
    names = DatasetRegistry.list_datasets()
    console.print("[green]Available datasets:[/green]")
    for n in names:
        console.print(f"  - {n}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
