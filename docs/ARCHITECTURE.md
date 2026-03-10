# Eval Lab Architecture

Eval Lab is an LLM evaluation framework designed for experimentation. It provides a modular pipeline for loading tasks, running inference through pluggable model adapters, computing metrics, and persisting results.

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Datasets   │───▶│    Runner    │───▶│   Metrics   │───▶│   Storage    │
│  Registry   │    │  (async)     │    │             │    │  (SQLite/PG) │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       │                    │                   │                   │
       │                    ▼                   │                   ▼
       │            ┌──────────────┐            │            ┌──────────────┐
       │            │   Model      │            │            │  Reporting   │
       │            │   Adapters   │            │            │  & Compare    │
       │            └──────────────┘            │            └──────────────┘
       │                    │                   │
       └────────────────────┴───────────────────┘
                    Slice Analysis
```

## Directory Structure

```
eval_lab/
├── datasets/          # Task and dataset registry
│   ├── base.py        # EvalDataset, EvalExample, BaseTask protocol
│   ├── registry.py    # DatasetRegistry, @register decorator
│   ├── task_adapter.py # TaskDatasetAdapter (BaseTask → EvalDataset)
│   ├── builtin.py     # Example dataset
│   └── tasks/         # Example tasks (reasoning, summarization, etc.)
├── models/            # Model adapters
│   ├── base.py        # ModelAdapter, ModelConfig, ModelResponse
│   ├── openai_adapter.py   # OpenAI-compatible APIs
│   ├── huggingface_adapter.py # HuggingFace Transformers
│   └── vllm_adapter.py      # vLLM (optional)
├── metrics/           # Evaluation metrics
│   ├── base.py        # Metric, MetricResult, MetricConfig
│   ├── exact_match.py
│   ├── f1.py
│   ├── latency.py
│   └── llm_judge.py
├── runners/           # Async evaluation runner
├── reporting/         # Report generation
│   ├── report_generator.py  # Single-run Markdown/HTML/JSON
│   ├── comparison_report.py # Multi-model comparison
│   ├── slice_analysis.py   # Group by category
│   └── templates/
├── storage/           # Result persistence
│   ├── store.py       # Factory (get_store)
│   ├── async_store.py # SQLAlchemy backend (SQLite/PostgreSQL)
│   └── base.py        # ORM models
├── api/               # FastAPI service
├── run.py             # Shared run_evaluation() for CLI, API, scripts
└── configs/           # Example YAML configs
```

## Data Flow

### 1. Dataset Loading

- **EvalDataset**: Abstract base with `load(config) -> list[EvalExample]`. Each example has `example_id`, `input_prompt`, `expected_output`, and optional `category` for slice analysis.
- **BaseTask**: Alternative protocol with `get_items()`, `format_prompt(item)`, `extract_answer(item, response)`. Tasks are wrapped by `TaskDatasetAdapter` into `EvalDataset`.
- **Registry**: `@DatasetRegistry.register(name)` registers datasets and tasks. Tasks are auto-wrapped as datasets.

### 2. Inference

- **ModelAdapter**: `generate(prompt, config) -> ModelResponse` (single) or batch via `BaseModelAdapter`.
- **OpenAIAdapter**: Uses `openai.AsyncOpenAI` with configurable `base_url` for local endpoints.
- **HuggingFaceAdapter**: Loads models via `transformers`, runs inference in thread pool.
- **VLLMAdapter**: Optional; uses vLLM for high-throughput local inference.

### 3. Metrics

- **Metric**: `compute(predicted, expected, example_id, extra_context) -> MetricResult` and `aggregate(results) -> float`.
- **ExactMatchMetric**: Normalized string equality (NFKC, lowercase, whitespace).
- **F1Metric**: Token-level F1 overlap.
- **LatencyMetric**: Reads `extra_context["latency_seconds"]`, aggregates mean/p50/p95/p99.
- **LLMJudgeMetric**: Expects `extra_context["llm_judge_score"]` from a judge model call.

### 4. Runner

- **AsyncEvalRunner**: Loads dataset, runs inference with concurrency limit, computes metrics per example, aggregates.
- **Post-processing**: If the dataset has `post_process(example, raw_output)`, the runner uses it before metric computation (e.g., extracting numbers from reasoning responses).
- **Persistence**: Optional `store` saves `EvalRunResult` after each run.

### 5. Storage

- **EvalStore**: Protocol with `save_run`, `get_run`, `list_runs`, `init_db`.
- **SQLite**: Default via `sqlite+aiosqlite:///eval_runs.sqlite`.
- **Postgres**: Set `DATABASE_URL=postgresql://...` for production.

### 6. Reporting

- **ReportGenerator**: Single-run JSON, Markdown, or HTML.
- **ComparisonReport**: Multi-model comparison with optional slice analysis.
- **SliceAnalysis**: Groups per-example results by `category`, computes metric scores per slice.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///eval_runs.sqlite` | Storage backend URL |
| `EVAL_LAB_PERSIST` | `true` | Whether to persist runs to DB |
| `EVAL_CONCURRENCY` | `4` | Max concurrent inference requests |
| `EVAL_MAX_EXAMPLES` | — | Cap examples (overrides config) |
| `OPENAI_API_KEY` | (env) | API key for OpenAI |
| `EVAL_API_KEY` | (env) | API key for other providers |
| `EVAL_API_BASE_URL` | (env) | API base URL for other providers |

## API Endpoints

- `GET /health` — Health check
- `GET /datasets` — List datasets
- `POST /eval/run` — Run evaluation (body: dataset, model_id, max_examples, metrics)
- `GET /runs` — List runs (query: dataset, model_id, limit)
- `GET /runs/{run_id}` — Get run details
- `GET /compare` — Compare models (query: run_ids or model_ids)

## CLI

```bash
uv run run-eval run -d reasoning -m gpt-5-mini --max-examples 5 --report report.md
uv run run-eval list-datasets
uv run eval-api  # Start FastAPI server
```

## Extending

- **New dataset**: Implement `EvalDataset` or `BaseTask`, use `@DatasetRegistry.register("name")`.
- **New adapter**: Implement `ModelAdapter` or `BaseModelAdapter`.
- **New metric**: Implement `Metric` (compute + aggregate), add to runner's metric list.
