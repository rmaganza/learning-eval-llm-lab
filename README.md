# Eval Lab

**LLM evaluation framework for experimentation**—inspired by EleutherAI's lm-evaluation-harness but built for rapid iteration. Supports async inference, pluggable datasets, multiple model backends, and experiment tracking.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Eval Lab provides a modular pipeline for evaluating language models: load tasks from a registry, run inference through pluggable adapters (OpenAI, HuggingFace, vLLM), compute metrics, and persist results for comparison. 

*This is just a learning project for me to get comfortable with LLMOps.*

## Features

| Feature | Description |
|---------|-------------|
| **Dataset registry** | Register tasks via decorator; built-in: reasoning, summarization, hallucination detection, instruction following |
| **Model adapters** | OpenAI-compatible APIs, HuggingFace Transformers, vLLM (optional) |
| **Async runner** | Concurrent inference with configurable concurrency |
| **Metrics** | Exact match, token F1, latency (mean/p50/p95/p99), LLM-as-judge |
| **Slice analysis** | Group results by task category for fine-grained comparison |
| **Model comparison** | Markdown/HTML reports comparing multiple models |
| **Storage** | SQLite (default) or PostgreSQL for experiment tracking |
| **REST API** | FastAPI service for running evaluations and querying results |

## Quick Start

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# Copy .env.example to .env and add your API key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# List available datasets
uv run run-eval list-datasets

# Run evaluation (requires OPENAI_API_KEY in .env or environment)
uv run run-eval run -d reasoning -m gpt-5-mini --max-examples 5

# Generate report
uv run run-eval run -d reasoning -m gpt-5-mini --report report.md

# Compare models (use same provider for both; e.g. two OpenAI models)
uv run python scripts/compare_models.py run gpt-5-mini,gpt-4o -d reasoning --max-examples 3 -o compare.html -f html

# Start API server
uv run eval-api
```

## Configuration

Environment variables are loaded from `.env` (via `python-dotenv`) when the package is imported. Create `.env` from the example:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `EVAL_API_KEY` | — | API key for other providers (Anthropic, local, etc.). **Required** with `EVAL_API_BASE_URL` when not using OpenAI. |
| `EVAL_API_BASE_URL` | — | API base URL for other providers (Anthropic, local). Set both this and `EVAL_API_KEY` together. |
| `DATABASE_URL` | `sqlite+aiosqlite:///eval_runs.sqlite` | Storage backend |
| `EVAL_LAB_PERSIST` | `true` | Persist runs to database |
| `EVAL_CONCURRENCY` | `4` | Max concurrent inference requests |
| `EVAL_MAX_EXAMPLES` | — | Cap examples (overrides config) |

## Project Structure

```
eval_lab/
├── datasets/       # Registry, base classes, task adapter, built-in tasks
├── models/         # Adapters (OpenAI, HuggingFace, vLLM)
├── metrics/        # Exact match, F1, latency, LLM judge
├── runners/        # Async evaluation runner
├── reporting/      # Report generators, slice analysis, comparison
├── storage/        # SQLite/PostgreSQL backends
├── api/            # FastAPI service
└── configs/        # Example YAML configs

scripts/            # compare_models, visualize (metrics_chart, export_runs)
docs/               # Architecture documentation
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/datasets` | List available datasets |
| POST | `/eval/run` | Run evaluation |
| GET | `/runs` | List past runs (filter: dataset, model_id) |
| GET | `/runs/{id}` | Get run details |
| GET | `/compare` | Compare models (run_ids or model_ids) |

### API Setup and Examples

Start the server (default: http://localhost:8000):

```bash
uv run eval-api
```

Interactive docs: http://localhost:8000/docs

**Example calls (curl):**

```bash
# Health check
curl http://localhost:8000/health

# List datasets
curl http://localhost:8000/datasets

# Run evaluation (requires OPENAI_API_KEY in .env)
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"dataset": "reasoning", "model_id": "gpt-5-mini", "max_examples": 3}'

# List runs (optional: ?dataset=reasoning&model_id=gpt-5-mini)
curl http://localhost:8000/runs

# Get run details
curl http://localhost:8000/runs/{run_id}

# Compare models by run IDs
curl "http://localhost:8000/compare?run_ids=id1,id2"

# Compare models by model IDs (latest run per model)
curl "http://localhost:8000/compare?model_ids=gpt-5-mini,claude-sonnet-4&dataset=reasoning"
```

## Extending

- **New dataset**: Implement `EvalDataset` or `BaseTask`, use `@DatasetRegistry.register("name")`.
- **New adapter**: Implement `ModelAdapter` or `BaseModelAdapter`.
- **New metric**: Implement `Metric` (compute + aggregate), add to runner.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check eval_lab

# Install pre-commit hook (ruff format on commit)
uv run pre-commit install
```
