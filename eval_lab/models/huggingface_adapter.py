"""HuggingFace model adapter with batch inference and token limit handling."""

import asyncio
import time
from collections.abc import AsyncIterator

from eval_lab.models.base import BaseModelAdapter


class HuggingFaceAdapter(BaseModelAdapter):
    """Adapter for HuggingFace transformers models."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        max_input_tokens: int | None = None,
        device_map: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map or "auto",
        )
        self._model.eval()

    def _prepare_batch(self, prompts: list[str]) -> tuple[dict, list[int]]:
        """Tokenize and truncate. Returns inputs and prompt lengths per sample."""
        if not prompts:
            return {}, []

        max_len = self.max_input_tokens or self._tokenizer.model_max_length
        if max_len <= 0:
            max_len = 2048

        inputs = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        prompt_lens = []
        if "attention_mask" in inputs:
            for row in inputs["attention_mask"]:
                prompt_lens.append(int(row.sum().item()))
        else:
            for p in prompts:
                tok = self._tokenizer.encode(p, add_special_tokens=True)
                prompt_lens.append(min(len(tok), max_len))

        return inputs, prompt_lens

    def _generate_batch_sync(self, prompts: list[str]) -> list[tuple[str, float]]:
        if self._model is None or self._tokenizer is None:
            self._load()

        inputs, prompt_lens = self._prepare_batch(prompts)
        if not inputs:
            return []

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        start = time.perf_counter()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        elapsed = time.perf_counter() - start

        results = []
        for out, plen in zip(outputs, prompt_lens):
            gen_ids = out[plen:]
            text = self._tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            per_sample_time = elapsed / len(prompts) if prompts else 0.0
            results.append((text.strip(), per_sample_time))

        return results

    async def generate(self, prompts: list[str]) -> AsyncIterator[tuple[str, float]]:
        if not prompts:
            return

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            results = await asyncio.to_thread(self._generate_batch_sync, batch)
            for text, lat in results:
                yield text, lat
