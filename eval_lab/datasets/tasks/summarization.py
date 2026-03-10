"""Short text summarization evaluation tasks."""

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from eval_lab.datasets.base import BaseTask
from eval_lab.datasets.registry import register


class SummarizationItem(BaseModel):
    """Text to summarize with reference summary."""

    id: str
    source_text: str
    reference_summary: str
    max_length: int = 50
    metadata: dict = Field(default_factory=dict)


SUMMARIZATION_ITEMS: list[SummarizationItem] = [
    SummarizationItem(
        id="s1",
        source_text="The company announced record profits for Q3, exceeding analyst expectations by 15%. CEO Jane Smith attributed the growth to strong demand in Asian markets and cost-cutting measures implemented earlier this year.",
        reference_summary="Company reports record Q3 profits, 15% above expectations, driven by Asian demand and cost cuts.",
    ),
    SummarizationItem(
        id="s2",
        source_text="Researchers at MIT have developed a new battery technology that could triple the range of electric vehicles. The prototype uses solid-state electrolytes and has shown promising results in laboratory tests.",
        reference_summary="MIT researchers create solid-state battery that may triple EV range.",
    ),
    SummarizationItem(
        id="s3",
        source_text="Local authorities have issued a boil water advisory for the downtown district after routine testing detected elevated levels of bacteria. Residents should boil water for at least one minute before drinking or cooking.",
        reference_summary="Boil water advisory issued for downtown due to bacteria in water supply.",
    ),
    SummarizationItem(
        id="s4",
        source_text="The new policy will take effect on January 1st. Employees will receive an additional 5 days of paid leave annually. Remote work options will be expanded for roles that don't require physical presence.",
        reference_summary="Policy changes: 5 extra paid days, expanded remote work, effective Jan 1.",
    ),
    SummarizationItem(
        id="s5",
        source_text="The concert sold out within hours of tickets going on sale. Organizers have added a second date to accommodate demand. Both shows will be held at the Riverside Arena in August.",
        reference_summary="Concert sold out quickly; second date added at Riverside Arena in August.",
    ),
    SummarizationItem(
        id="s6",
        source_text="Scientists have identified a new species of deep-sea fish that glows in the dark. The creature was discovered at a depth of 3,000 meters during an expedition in the Pacific. It uses bioluminescence to attract prey.",
        reference_summary="New glowing deep-sea fish species found at 3,000m in Pacific.",
    ),
    SummarizationItem(
        id="s7",
        source_text="The bridge closure will affect approximately 50,000 commuters daily. A detour adding 20 minutes to the typical route has been established. Repairs are expected to take six weeks.",
        reference_summary="Bridge closure impacts 50k commuters; 6-week repair, 20-min detour in place.",
    ),
]


@register()
class SummarizationTask(BaseTask):
    """Short text summarization with reference summaries."""

    name = "summarization"
    item_model = SummarizationItem

    async def get_items(self) -> AsyncIterator[SummarizationItem]:
        for item in SUMMARIZATION_ITEMS:
            yield item

    def format_prompt(self, item: SummarizationItem) -> str:
        return f"""Summarize the following text in one or two concise sentences. Capture the key facts and main point.

Text:
{item.source_text}

Summary:"""

    def extract_answer(self, item: SummarizationItem, response: str) -> str:
        """Return the model's summary as-is for downstream evaluation (e.g., ROUGE, BLEU)."""
        return response.strip()
