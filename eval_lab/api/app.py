"""FastAPI app with evaluation endpoints."""

import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from eval_lab import __version__
from eval_lab.datasets import DatasetRegistry
from eval_lab.run import run_evaluation
from eval_lab.storage.store import get_store


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Eval Lab API",
        version=__version__,
        description="LLM evaluation framework—run benchmarks, compare models, track experiments",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        """Health check for load balancers and monitoring."""
        return {"status": "ok", "version": __version__}

    @app.get("/datasets", tags=["Datasets"])
    async def list_datasets() -> dict:
        """List available datasets."""
        datasets = DatasetRegistry.list_datasets()
        return {"datasets": datasets}

    class EvalRunRequest(BaseModel):
        dataset: str = Field(..., description="Dataset name")
        model_id: str = Field(..., description="Model identifier")
        max_examples: int | None = Field(None, description="Cap examples for quick runs")
        metrics: list[str] | None = Field(
            None,
            description="Metric names (default: exact_match, f1, latency)",
        )
        judge_model_id: str | None = Field(
            None,
            description="Model for LLM-as-judge; enables llm_judge when set",
        )
        judge_mode: str = Field("numeric", description="Judge mode: numeric or binary")

    @app.post("/eval/run", tags=["Evaluation"])
    async def run_eval(body: EvalRunRequest) -> dict:
        """Run evaluation on a dataset with a model."""
        try:
            DatasetRegistry.get(body.dataset)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        store = (
            get_store() if os.environ.get("EVAL_LAB_PERSIST", "true").lower() == "true" else None
        )
        try:
            result = await run_evaluation(
                body.dataset,
                body.model_id,
                max_examples=body.max_examples,
                metric_names=body.metrics,
                judge_model=body.judge_model_id,
                judge_mode=body.judge_mode,
                concurrency=4,
                store=store,
                run_id=str(uuid4()),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "run_id": result.run_id,
            "dataset": result.dataset_name,
            "model_id": result.model_id,
            "metric_scores": result.metric_scores,
            "total_examples": result.total_examples,
            "failed_examples": result.failed_examples,
            "failed_errors": getattr(result, "failed_errors", []) or [],
        }

    @app.get("/runs", tags=["Runs"])
    async def list_runs(
        dataset: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> dict:
        """List past evaluation runs from storage."""
        store = get_store()
        await store.init_db()
        runs = await store.list_runs(dataset=dataset, model_id=model_id, limit=limit)
        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "dataset": r.dataset_name,
                    "model_id": r.model_id,
                    "metric_scores": r.metric_scores,
                    "total_examples": r.total_examples,
                    "failed_examples": r.failed_examples,
                }
                for r in runs
            ]
        }

    @app.get("/runs/{run_id}", tags=["Runs"])
    async def get_run(run_id: str) -> dict:
        """Get details for a specific run."""
        store = get_store()
        await store.init_db()
        result = await store.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return {
            "run_id": result.run_id,
            "dataset": result.dataset_name,
            "model_id": result.model_id,
            "metric_scores": result.metric_scores,
            "per_example_results": result.per_example_results,
            "total_examples": result.total_examples,
            "failed_examples": result.failed_examples,
            "failed_errors": getattr(result, "failed_errors", []) or [],
        }

    @app.get("/compare", tags=["Comparison"])
    async def compare_models(
        run_ids: str | None = None,
        model_ids: str | None = None,
        dataset: str | None = None,
        limit: int = 20,
    ) -> dict:
        """
        Compare multiple models.

        Provide either run_ids (comma-separated) or model_ids (comma-separated).
        If model_ids given, fetches latest runs for each model (optionally filtered by dataset).
        """
        from eval_lab.runners.async_runner import EvalRunResult

        store = get_store()
        await store.init_db()

        if run_ids:
            ids = [r.strip() for r in run_ids.split(",") if r.strip()]
            results: list[EvalRunResult] = []
            for rid in ids:
                r = await store.get_run(rid)
                if r:
                    results.append(r)
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="None of the specified run_ids were found",
                )
        elif model_ids:
            ids = [m.strip() for m in model_ids.split(",") if m.strip()]
            all_runs = await store.list_runs(dataset=dataset, limit=limit * 10)
            seen: set[str] = set()
            results = []
            for r in all_runs:
                if r.model_id in ids and r.model_id not in seen:
                    seen.add(r.model_id)
                    results.append(r)
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="No runs found for the specified model_ids",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide run_ids or model_ids as query parameters",
            )

        comparison = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "dataset": r.dataset_name,
                    "model_id": r.model_id,
                    "metric_scores": r.metric_scores,
                    "total_examples": r.total_examples,
                }
                for r in results
            ],
            "metrics": list(results[0].metric_scores.keys()) if results else [],
        }
        return comparison

    return app


app = create_app()
