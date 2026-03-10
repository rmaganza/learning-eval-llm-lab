"""Math and logic reasoning evaluation tasks."""

import re
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from eval_lab.datasets.base import BaseTask
from eval_lab.datasets.registry import register


class ReasoningItem(BaseModel):
    """Single reasoning problem with question and expected answer."""

    id: str
    question: str
    expected_answer: str | int | float
    metadata: dict = Field(default_factory=dict)


REASONING_ITEMS: list[ReasoningItem] = [
    ReasoningItem(
        id="r1",
        question="A store sells apples for $2 each and oranges for $3 each. If Maria buys 4 apples and 2 oranges, how much does she pay in total?",
        expected_answer=14,
        metadata={"category": "word_problem"},
    ),
    ReasoningItem(
        id="r2",
        question="What is 2 + 2?",
        expected_answer=4,
        metadata={"category": "arithmetic"},
    ),
    ReasoningItem(
        id="r3",
        question="If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?",
        expected_answer=60,
    ),
    ReasoningItem(
        id="r4",
        question="Three friends split a $45 dinner bill equally. How much does each person pay?",
        expected_answer=15,
    ),
    ReasoningItem(
        id="r5",
        question="A rectangle has a length of 8 cm and width of 5 cm. What is its area in square centimeters?",
        expected_answer=40,
    ),
    ReasoningItem(
        id="r6",
        question="If you have 17 candies and give away 9, how many do you have left?",
        expected_answer=8,
    ),
    ReasoningItem(
        id="r7",
        question="A bookshelf has 4 shelves. Each shelf holds 12 books. How many books can the shelf hold in total?",
        expected_answer=48,
    ),
    ReasoningItem(
        id="r8",
        question="Sarah is 3 years older than Tom. Tom is 15. How old is Sarah?",
        expected_answer=18,
    ),
    ReasoningItem(
        id="r9",
        question="A shirt originally costs $40. It's on sale for 25% off. What is the sale price?",
        expected_answer=30,
    ),
]


@register()
class ReasoningTask(BaseTask):
    """Simple math and logic reasoning evaluation."""

    name = "reasoning"
    item_model = ReasoningItem

    async def get_items(self) -> AsyncIterator[ReasoningItem]:
        for item in REASONING_ITEMS:
            yield item

    def format_prompt(self, item: ReasoningItem) -> str:
        return f"""Solve the following problem. Show your reasoning step by step, then provide your final answer as a number.

Problem: {item.question}

Your answer:"""

    def extract_answer(self, item: ReasoningItem, response: str) -> str | int | float | None:
        """Extract numeric answer from response. Handles 'answer is X', '= X', or trailing number."""
        # Normalize expected for comparison
        expected = item.expected_answer
        if isinstance(expected, float) and expected == int(expected):
            expected = int(expected)

        # Try to find explicit "answer is X" or "= X" patterns
        patterns = [
            r"(?:answer|result|total|equals?)\s*[:\s=]+\s*(\d+(?:\.\d+)?)",
            r"(?:=\s*)(\d+(?:\.\d+)?)\s*$",
            r"\b(\d+(?:\.\d+)?)\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                val = m.group(1)
                try:
                    return int(val) if "." not in val else float(val)
                except ValueError:
                    pass

        # Fallback: last number in response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
        if numbers:
            try:
                last = numbers[-1]
                return int(last) if "." not in last else float(last)
            except ValueError:
                pass
        return None
