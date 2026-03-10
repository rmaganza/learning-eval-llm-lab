"""vLLM adapter with optional import (vllm may not be installed)."""

import asyncio
import time
from collections.abc import AsyncIterator

from eval_lab.models.base import BaseModelAdapter

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class VLLMAdapter(BaseModelAdapter):
    """Adapter for vLLM high-throughput inference."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 32,
        max_tokens: int = 256,
        tensor_parallel_size: int = 1,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed. Install with: pip install vllm or uv add 'eval-lab[vllm]'"
            )
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self._engine: LLM | None = None

    def _ensure_engine(self) -> "LLM":
        if self._engine is None:
            self._engine = LLM(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
            )
        return self._engine

    def _generate_batch_sync(self, prompts: list[str]) -> list[tuple[str, float]]:
        if not prompts:
            return []

        engine = self._ensure_engine()
        params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        start = time.perf_counter()
        outputs = engine.generate(prompts, params)
        elapsed = time.perf_counter() - start

        per_sample = elapsed / len(prompts) if prompts else 0.0
        results = []
        for out in outputs:
            text = (out.outputs[0].text if out.outputs else "").strip()
            results.append((text, per_sample))
        return results

    async def generate(self, prompts: list[str]) -> AsyncIterator[tuple[str, float]]:
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            results = await asyncio.to_thread(self._generate_batch_sync, batch)
            for text, lat in results:
                yield text, lat
