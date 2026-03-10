#!/usr/bin/env -S uv run python
"""Compare models and generate a report from stored runs or fresh evaluations."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from eval_lab.reporting.comparison_report import write_comparison_report
from eval_lab.run import run_evaluation
from eval_lab.runners.async_runner import EvalRunResult
from eval_lab.storage.store import get_store

app = typer.Typer()
console = Console()


async def _fetch_runs(run_ids: list[str]) -> list[EvalRunResult]:
    """Fetch runs by ID from storage."""
    store = get_store()
    await store.init_db()
    results: list[EvalRunResult] = []
    for rid in run_ids:
        r = await store.get_run(rid)
        if r:
            results.append(r)
        else:
            console.print(f"[yellow]Run {rid} not found[/yellow]")
    return results


@app.command()
def from_runs(
    run_ids: str = typer.Argument(..., help="Comma-separated run IDs"),
    output: Path = typer.Option(Path("comparison_report.md"), "--output", "-o"),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown or html"),
) -> None:
    """Generate comparison report from stored run IDs."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    results = asyncio.run(_fetch_runs(ids))
    if not results:
        console.print("[red]No runs found[/red]")
        raise typer.Exit(1)
    write_comparison_report(results, output, format=format)
    console.print(f"[green]Report written to {output}[/green]")


@app.command()
def run(
    models: str = typer.Argument(..., help="Comma-separated model IDs (e.g. gpt-5-mini,gpt-4o)"),
    dataset: str = typer.Option("reasoning", "--dataset", "-d"),
    max_examples: int | None = typer.Option(None, "--max-examples"),
    output: Path = typer.Option(Path("comparison_report.md"), "--output", "-o"),
    format: str = typer.Option("markdown", "--format", "-f"),
) -> None:
    """Run evaluations for multiple models and generate comparison report."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        console.print("[red]Provide at least one model ID[/red]")
        raise typer.Exit(1)

    results: list[EvalRunResult] = []
    for model_id in model_list:
        console.print(f"Evaluating [cyan]{model_id}[/cyan] on {dataset}...")
        r = asyncio.run(
            run_evaluation(
                dataset,
                model_id,
                max_examples=max_examples,
                metric_names=["exact_match", "f1", "latency"],
                store=None,
            )
        )
        results.append(r)
        console.print(f"  exact_match={r.metric_scores.get('exact_match', 0):.4f}")

    write_comparison_report(results, output, format=format)
    console.print(f"[green]Report written to {output}[/green]")


if __name__ == "__main__":
    app()
