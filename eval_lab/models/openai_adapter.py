"""OpenAI-compatible API adapter with configurable base URL."""

import time
from collections.abc import AsyncIterator

from eval_lab.models.base import (
    BaseModelAdapter,
    ModelAdapter,
    ModelConfig,
    ModelResponse,
)


class OpenAIAdapter(ModelAdapter, BaseModelAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, local servers, etc.)."""

    def __init__(
        self,
        model: str = "",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.model = model or "gpt-5-mini"
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _client_and_params(self, config: ModelConfig | None = None):
        from openai import AsyncOpenAI

        model = (config.model_id if config and config.model_id else None) or self.model
        base = config.base_url if config else self.base_url
        key = config.api_key if config else self.api_key
        max_tok = config.max_tokens if config else self.max_tokens
        kwargs: dict = {}
        if base:
            kwargs["base_url"] = base
        if key:
            kwargs["api_key"] = key
        return AsyncOpenAI(**kwargs), model, max_tok or self.max_tokens

    async def generate(
        self, prompt_or_prompts: str | list[str], config: ModelConfig | None = None
    ) -> ModelResponse | AsyncIterator[tuple[str, float]]:
        if isinstance(prompt_or_prompts, str):
            return await self._generate_one(prompt_or_prompts, config or ModelConfig())
        return self._generate_batch(prompt_or_prompts)

    async def _generate_one(self, prompt: str, config: ModelConfig) -> ModelResponse:
        client, model, max_tok = self._client_and_params(config)
        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt or ""}],
                max_tokens=max_tok,
            )
            text = (resp.choices[0].message.content or "").strip()
            return ModelResponse(
                generated_text=text,
                latency_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            return ModelResponse(
                generated_text="",
                latency_seconds=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )

    async def _generate_batch(
        self, prompts: list[str]
    ) -> AsyncIterator[tuple[str, float]]:
        client, model, max_tok = self._client_and_params(None)
        for p in prompts:
            p = "" if p is None else p
            start = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=max_tok,
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception:
                text = ""
            yield text, time.perf_counter() - start

    async def close(self) -> None:
        pass
