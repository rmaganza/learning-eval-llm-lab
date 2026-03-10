"""Configuration loading from YAML and environment."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetConfigSchema(BaseModel):
    """Dataset section of eval config."""

    name: str
    split: str = "test"
    max_examples: int | None = None


class ModelConfigSchema(BaseModel):
    """Model section of eval config."""

    model_id: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 256
    base_url: str | None = None
    api_key: str | None = None


class EvalConfig(BaseModel):
    """Full evaluation configuration from YAML."""

    dataset: DatasetConfigSchema = Field(default_factory=DatasetConfigSchema)
    model: ModelConfigSchema = Field(default_factory=ModelConfigSchema)
    metrics: list[str] = Field(default_factory=lambda: ["exact_match", "f1", "latency"])
    runner: dict[str, Any] = Field(default_factory=lambda: {"concurrency": 4})


def load_config(path: str | Path) -> EvalConfig:
    """Load evaluation config from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config loading: pip install pyyaml")
    raw = yaml.safe_load(path.read_text()) or {}
    return EvalConfig(
        dataset=DatasetConfigSchema(**(raw.get("dataset") or {})),
        model=ModelConfigSchema(**(raw.get("model") or {})),
        metrics=raw.get("metrics", ["exact_match", "f1", "latency"]),
        runner=raw.get("runner", {"concurrency": 4}),
    )
