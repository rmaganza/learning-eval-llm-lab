#!/usr/bin/env -S uv run python
"""Visualization scripts for comparing models and metrics."""

import json
from pathlib import Path

import typer

app = typer.Typer()


def _load_results(path: Path) -> list[dict]:
    """Load evaluation results from JSON file or directory of JSON files."""
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        return [data]
    if path.is_dir():
        out = []
        for f in sorted(path.glob("*.json")):
            out.extend(_load_results(f))
        return out
    return []


@app.command()
def metrics_chart(
    input_path: Path = typer.Argument(..., help="JSON file or dir of eval results"),
    output: Path = typer.Option(Path("metrics_chart.html"), "--output", "-o"),
    metric: str = typer.Option("exact_match", "--metric", "-m"),
) -> None:
    """
    Generate an HTML bar chart comparing models on a single metric.

    Input JSON should have model_id and metric_scores (or be EvalRunResult format).
    """
    results = _load_results(input_path)
    if not results:
        typer.echo("No results found", err=True)
        raise typer.Exit(1)
    models = []
    scores = []
    for r in results:
        mid = r.get("model_id") or r.get("model", "unknown")
        ms = r.get("metric_scores", r.get("metrics", {}))
        val = ms.get(metric, 0.0)
        models.append(mid)
        scores.append(float(val))
    html = _bar_chart_html(models, scores, metric)
    output.write_text(html, encoding="utf-8")
    typer.echo(f"Chart written to {output}")


def _bar_chart_html(models: list[str], scores: list[float], metric: str) -> str:
    """Inline HTML with CSS/JS for a simple bar chart."""
    max_val = max(scores) if scores else 1.0
    bars = "".join(
        f'<div class="bar-row"><span class="label">{m}</span>'
        f'<div class="bar-wrap"><div class="bar" style="width:{100 * s / max_val}%"></div></div>'
        f'<span class="value">{s:.4f}</span></div>'
        for m, s in zip(models, scores)
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Metrics: {metric}</title>
<style>
  body {{ font-family: system-ui; margin: 2rem; }}
  .bar-row {{ display: flex; align-items: center; margin-bottom: 0.5rem; }}
  .label {{ width: 180px; }}
  .bar-wrap {{ flex: 1; height: 20px; background: #eee; margin: 0 1rem; border-radius: 4px; overflow: hidden; }}
  .bar {{ height: 100%; background: #4a9; border-radius: 4px; }}
  .value {{ font-variant-numeric: tabular-nums; width: 80px; }}
</style>
</head>
<body>
  <h1>{metric}</h1>
  {bars}
</body>
</html>"""


@app.command()
def export_runs(
    output_dir: Path = typer.Argument(Path("runs_export"), help="Output directory"),
    dataset: str | None = typer.Option(None, "--dataset"),
    model_id: str | None = typer.Option(None, "--model"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Export runs from DB to JSON files (for use with metrics_chart)."""
    import asyncio

    from eval_lab.storage.store import get_store

    async def _export():
        store = get_store()
        await store.init_db()
        runs = await store.list_runs(dataset=dataset, model_id=model_id, limit=limit)
        output_dir.mkdir(parents=True, exist_ok=True)
        for r in runs:
            path = output_dir / f"{r.run_id}.json"
            path.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        return len(runs)

    n = asyncio.run(_export())
    typer.echo(f"Exported {n} runs to {output_dir}")


if __name__ == "__main__":
    app()
