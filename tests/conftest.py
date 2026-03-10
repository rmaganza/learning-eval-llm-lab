"""Pytest configuration."""

import os

import pytest

# Use in-memory SQLite for tests to avoid schema migration issues
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture
def sample_eval_examples() -> list:
    """Sample EvalExample instances for tests."""
    from eval_lab.datasets.base import EvalExample

    return [
        EvalExample(
            example_id="1",
            input_prompt="What is 2+2?",
            expected_output="4",
        ),
        EvalExample(
            example_id="2",
            input_prompt="Capital of France?",
            expected_output="Paris",
        ),
    ]
