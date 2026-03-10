"""Instruction following: format instructions (JSON output, specific structure)."""

import json
import re
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from eval_lab.datasets.base import BaseTask
from eval_lab.datasets.registry import register


class InstructionItem(BaseModel):
    """Task requiring specific output format (JSON structure)."""

    id: str
    instruction: str
    input_data: str | dict
    expected_schema: dict  # Expected JSON schema for validation
    metadata: dict = Field(default_factory=dict)


INSTRUCTION_ITEMS: list[InstructionItem] = [
    InstructionItem(
        id="i1",
        instruction="Extract the following information as JSON: name, age, occupation. Use these exact keys.",
        input_data="John Smith is 34 years old and works as a software engineer.",
        expected_schema={"name": "string", "age": "number", "occupation": "string"},
    ),
    InstructionItem(
        id="i2",
        instruction="List the items mentioned as a JSON array of strings. Output only the array, no other text.",
        input_data="The recipe requires flour, sugar, eggs, and butter.",
        expected_schema={"type": "array", "items": {"type": "string"}},
    ),
    InstructionItem(
        id="i3",
        instruction="Output a JSON object with keys 'summary' (one sentence) and 'sentiment' (positive/negative/neutral).",
        input_data="The product received mixed reviews. Customers praised the design but complained about the battery life.",
        expected_schema={"summary": "string", "sentiment": "string"},
    ),
    InstructionItem(
        id="i4",
        instruction="Return JSON with 'entities' (array of extracted names/places) and 'date' (if mentioned, else null).",
        input_data="On March 15, 2024, Alice met Bob at Central Park in New York.",
        expected_schema={"entities": "array", "date": "string|null"},
    ),
    InstructionItem(
        id="i5",
        instruction="Format as JSON: { \"pros\": [...], \"cons\": [...] }. Each list has string items.",
        input_data="Pros: fast, reliable. Cons: expensive, heavy.",
        expected_schema={"pros": "array", "cons": "array"},
    ),
    InstructionItem(
        id="i6",
        instruction="Output valid JSON only. Structure: { \"answer\": <boolean>, \"confidence\": <0-1> }.",
        input_data="Is the following claim supported? 'All mammals lay eggs.'",
        expected_schema={"answer": "boolean", "confidence": "number"},
    ),
    InstructionItem(
        id="i7",
        instruction="Respond with a JSON object containing 'translation' and 'detected_language'.",
        input_data="Hola, ¿cómo estás?",
        expected_schema={"translation": "string", "detected_language": "string"},
    ),
    InstructionItem(
        id="i8",
        instruction="Return JSON: { \"steps\": [ordered list of steps as strings], \"total_time_minutes\": number }.",
        input_data="First boil water. Then add pasta and cook for 10 minutes. Drain and serve. Total time about 15 minutes.",
        expected_schema={"steps": "array", "total_time_minutes": "number"},
    ),
]


@register()
class InstructionFollowingTask(BaseTask):
    """Follow format instructions (JSON output, specific structure)."""

    name = "instruction_following"
    item_model = InstructionItem

    async def get_items(self) -> AsyncIterator[InstructionItem]:
        for item in INSTRUCTION_ITEMS:
            yield item

    def format_prompt(self, item: InstructionItem) -> str:
        input_str = (
            json.dumps(item.input_data) if isinstance(item.input_data, dict) else item.input_data
        )
        return f"""Follow the instruction exactly. Output only valid JSON, no markdown or extra text.

Instruction: {item.instruction}

Input:
{input_str}

JSON output:"""

    def extract_answer(self, item: InstructionItem, response: str) -> dict | list | None:
        """Extract and parse JSON from the response."""
        response = response.strip()
        # Remove markdown code blocks if present
        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object/array in response
            for start in ("{", "["):
                idx = response.find(start)
                if idx >= 0:
                    depth = 0
                    for i, c in enumerate(response[idx:], start=idx):
                        if c in "{[":
                            depth += 1
                        elif c in "}]":
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(response[idx : i + 1])
                                except json.JSONDecodeError:
                                    break
        return None
