"""Storage protocol and types for evaluation result backends."""

from typing import Protocol

from eval_lab.runners.async_runner import EvalRunResult


class EvalStore(Protocol):
    """Protocol for evaluation result storage backends."""

    async def save_run(self, result: EvalRunResult) -> None:
        """Persist an evaluation run result."""
        ...

    async def get_run(self, run_id: str) -> EvalRunResult | None:
        """Retrieve a run by ID."""
        ...

    async def list_runs(
        self,
        dataset: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[EvalRunResult]:
        """List past runs with optional filters."""
        ...

    async def init_db(self) -> None:
        """Create tables if they don't exist."""
        ...
