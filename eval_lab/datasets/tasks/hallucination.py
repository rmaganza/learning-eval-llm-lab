"""Factual statement classification: true, false, or hallucination."""

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from eval_lab.datasets.base import BaseTask
from eval_lab.datasets.registry import register


class HallucinationItem(BaseModel):
    """Statement to classify as true, false, or hallucination (unverifiable/unsupported)."""

    id: str
    statement: str
    expected_label: str  # "true", "false", or "hallucination"
    context: str | None = None  # Optional context for verification
    metadata: dict = Field(default_factory=dict)


HALLUCINATION_ITEMS: list[HallucinationItem] = [
    HallucinationItem(
        id="h1",
        statement="The Earth orbits the Sun.",
        expected_label="true",
    ),
    HallucinationItem(
        id="h2",
        statement="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        expected_label="true",
    ),
    HallucinationItem(
        id="h3",
        statement="Napoleon Bonaparte was the first president of France.",
        expected_label="false",
        context="Napoleon was Emperor; the first president was Louis-Napoleon Bonaparte (1848).",
    ),
    HallucinationItem(
        id="h4",
        statement="The company's Q4 revenue increased by 23% year-over-year.",
        expected_label="hallucination",
        context="No specific company or financial data provided. Cannot verify without source.",
    ),
    HallucinationItem(
        id="h5",
        statement="Humans have 206 bones in the adult body.",
        expected_label="true",
    ),
    HallucinationItem(
        id="h6",
        statement="The Great Wall of China is visible from the Moon with the naked eye.",
        expected_label="false",
    ),
    HallucinationItem(
        id="h7",
        statement="According to a 2023 study, 78% of respondents preferred option A.",
        expected_label="hallucination",
        context="No study or survey cited. Specific percentage is unverifiable.",
    ),
    HallucinationItem(
        id="h8",
        statement="Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        expected_label="true",
    ),
    HallucinationItem(
        id="h9",
        statement="The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        expected_label="true",
    ),
    HallucinationItem(
        id="h10",
        statement="The novel was published in 1847 and sold over 2 million copies in its first year.",
        expected_label="hallucination",
        context="Which novel? No source. Specific sales figures unverifiable.",
    ),
]


@register()
class HallucinationTask(BaseTask):
    """Identify whether statements are true, false, or hallucination (unverifiable)."""

    name = "hallucination"
    item_model = HallucinationItem

    async def get_items(self) -> AsyncIterator[HallucinationItem]:
        for item in HALLUCINATION_ITEMS:
            yield item

    def format_prompt(self, item: HallucinationItem) -> str:
        prompt = """Classify the following statement as exactly one of: true, false, hallucination.

- true: The statement is factually correct and verifiable.
- false: The statement is factually incorrect.
- hallucination: The statement cannot be verified, contains unsupported claims, or mixes real and fabricated details.

Statement: """
        prompt += item.statement
        if item.context:
            prompt += f"\n\nContext: {item.context}"
        prompt += "\n\nClassification:"
        return prompt

    def extract_answer(self, item: HallucinationItem, response: str) -> str | None:
        """Extract the classification label from the response."""
        response_lower = response.strip().lower()
        # Check for explicit labels
        for label in ("true", "false", "hallucination"):
            if label in response_lower:
                # Prefer exact match or at word boundary
                words = response_lower.split()
                for w in words:
                    if w == label or w.startswith(label):
                        return label
                return label
        return None
