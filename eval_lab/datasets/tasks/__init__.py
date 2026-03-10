"""Evaluation tasks. Import to trigger registration."""

from eval_lab.datasets.tasks.hallucination import HallucinationTask
from eval_lab.datasets.tasks.instruction_following import InstructionFollowingTask
from eval_lab.datasets.tasks.reasoning import ReasoningTask
from eval_lab.datasets.tasks.summarization import SummarizationTask

__all__ = [
    "HallucinationTask",
    "InstructionFollowingTask",
    "ReasoningTask",
    "SummarizationTask",
]
