"""Base protocol and types for model adapters."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Protocol for model adapters that generate text from prompts."""

    async def generate(
        self, prompts: list[str]
    ) -> AsyncIterator[tuple[str, float]]:
        """
        Generate completions for a batch of prompts.

        Yields (completion, latency_seconds) for each prompt in order.
        Latency is wall-clock time for that sample's inference.
        """
        ...


class ModelConfig(BaseModel):
    """Configuration for model inference."""

    model_id: str = Field(default="")
    max_tokens: int = 256
    base_url: str | None = None
    api_key: str | None = None


class ModelResponse(BaseModel):
    """Single inference response."""

    generated_text: str = ""
    latency_seconds: float = 0.0
    error: str | None = None  # Set when inference fails


class ModelAdapter:
    """
    Adapter interface for the eval runner.
    Wraps BaseModelAdapter to support single-prompt generate(prompt, config).
    """

    async def generate(
        self, prompt: str, config: ModelConfig
    ) -> ModelResponse:
        """Generate for a single prompt. Override or use from_batch_adapter."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release resources. Override if needed."""
        pass

    @classmethod
    def from_batch_adapter(
        cls, adapter: BaseModelAdapter, config: ModelConfig
    ) -> "ModelAdapter":
        """Wrap a BaseModelAdapter for runner compatibility."""
        return _BatchAdapterWrapper(adapter, config)


class _BatchAdapterWrapper(ModelAdapter):
    def __init__(self, adapter: BaseModelAdapter, config: ModelConfig) -> None:
        self._adapter = adapter
        self._config = config

    async def generate(
        self, prompt: str, config: ModelConfig
    ) -> ModelResponse:
        texts = []
        lats = []
        async for text, lat in self._adapter.generate([prompt]):
            texts.append(text)
            lats.append(lat)
        return ModelResponse(
            generated_text=texts[0] if texts else "",
            latency_seconds=lats[0] if lats else 0.0,
        )
