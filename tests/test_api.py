"""Tests for the FastAPI application."""

import asyncio
import importlib
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from eval_lab.api.app import app
from eval_lab.runners.async_runner import EvalRunResult
from eval_lab.storage.store import get_store

# Module where run_evaluation is used (patch must target this, not eval_lab.run)
_app_module = importlib.import_module("eval_lab.api.app")


@pytest.fixture
def client() -> TestClient:
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def db_path(tmp_path):
    """Use a file-based SQLite DB for tests that need persistence across requests."""
    path = tmp_path / "test_api.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{path}"
    yield path
    os.environ.pop("DATABASE_URL", None)


async def _seed_runs() -> list[str]:
    """Seed storage with test runs. Returns list of run_ids."""
    store = get_store()
    await store.init_db()
    run1 = EvalRunResult(
        run_id="run-111",
        dataset_name="reasoning",
        model_id="gpt-5-mini",
        metric_scores={"exact_match": 0.8, "f1": 0.85},
        total_examples=10,
        failed_examples=0,
    )
    run2 = EvalRunResult(
        run_id="run-222",
        dataset_name="reasoning",
        model_id="claude-sonnet",
        metric_scores={"exact_match": 0.9, "f1": 0.92},
        total_examples=10,
        failed_examples=0,
    )
    await store.save_run(run1)
    await store.save_run(run2)
    return [run1.run_id, run2.run_id]


def test_health(client: TestClient) -> None:
    """Health endpoint returns ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_datasets(client: TestClient) -> None:
    """Datasets endpoint returns list."""
    resp = client.get("/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)
    assert "example" in data["datasets"] or "reasoning" in data["datasets"]


def test_run_evaluation_missing_dataset(client: TestClient) -> None:
    """Run with unknown dataset returns 404."""
    resp = client.post(
        "/eval/run",
        json={"dataset": "nonexistent", "model_id": "gpt-5-mini", "max_examples": 1},
    )
    assert resp.status_code == 404


def test_run_evaluation_success(client: TestClient) -> None:
    """Run evaluation returns result when dataset exists."""
    mock_result = EvalRunResult(
        run_id="run-abc",
        dataset_name="reasoning",
        model_id="gpt-5-mini",
        metric_scores={"exact_match": 0.7, "f1": 0.75, "latency": 0.5},
        total_examples=5,
        failed_examples=0,
    )
    with patch.object(_app_module, "run_evaluation", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        resp = client.post(
            "/eval/run",
            json={"dataset": "reasoning", "model_id": "gpt-5-mini", "max_examples": 3},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run-abc"
    assert data["dataset"] == "reasoning"
    assert data["model_id"] == "gpt-5-mini"
    assert data["metric_scores"] == {"exact_match": 0.7, "f1": 0.75, "latency": 0.5}
    assert data["total_examples"] == 5
    assert data["failed_examples"] == 0


def test_list_runs_empty(client: TestClient) -> None:
    """List runs returns empty list when no runs exist."""
    resp = client.get("/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert data["runs"] == []


def test_list_runs_with_filters(client: TestClient, db_path) -> None:
    """List runs accepts dataset and model_id filters."""
    asyncio.run(_seed_runs())
    resp = client.get("/runs?dataset=reasoning&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 2
    assert all(r["dataset"] == "reasoning" for r in data["runs"])


def test_get_run_not_found(client: TestClient) -> None:
    """Get run with non-existent ID returns 404."""
    resp = client.get("/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_run_success(client: TestClient, db_path) -> None:
    """Get run returns run details when it exists."""
    run_ids = asyncio.run(_seed_runs())
    resp = client.get(f"/runs/{run_ids[0]}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_ids[0]
    assert data["dataset"] == "reasoning"
    assert data["model_id"] == "gpt-5-mini"
    assert "metric_scores" in data
    assert "per_example_results" in data


def test_compare_missing_params(client: TestClient) -> None:
    """Compare without run_ids or model_ids returns 400."""
    resp = client.get("/compare")
    assert resp.status_code == 400


def test_compare_with_run_ids(client: TestClient, db_path) -> None:
    """Compare returns comparison when run_ids provided."""
    run_ids = asyncio.run(_seed_runs())
    resp = client.get(f"/compare?run_ids={run_ids[0]},{run_ids[1]}")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 2
    assert "metrics" in data
    assert data["metrics"] == ["exact_match", "f1"]


def test_compare_with_model_ids(client: TestClient, db_path) -> None:
    """Compare returns latest runs per model when model_ids provided."""
    asyncio.run(_seed_runs())
    resp = client.get("/compare?model_ids=gpt-5-mini,claude-sonnet&dataset=reasoning")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 2
    model_ids = {r["model_id"] for r in data["runs"]}
    assert model_ids == {"gpt-5-mini", "claude-sonnet"}
