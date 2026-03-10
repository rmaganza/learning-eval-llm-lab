"""Shared SQLAlchemy async storage backend for SQLite and PostgreSQL."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from eval_lab.runners.async_runner import EvalRunResult
from eval_lab.storage.base import Base, EvalRun


def _row_to_result(row: EvalRun) -> EvalRunResult:
    """Convert EvalRun ORM row to EvalRunResult."""
    return EvalRunResult(
        run_id=row.run_id,
        dataset_name=row.dataset,
        model_id=row.model_id,
        metric_scores=row.metric_scores or {},
        per_example_results=row.per_example or [],
        total_examples=row.total_examples,
        failed_examples=row.failed_examples,
        failed_errors=row.failed_errors or [],
    )


class AsyncEvalStore:
    """SQLAlchemy-backed storage for evaluation runs (SQLite or PostgreSQL)."""

    def __init__(self, url: str) -> None:
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_run(self, result: EvalRunResult) -> None:
        """Persist an evaluation run result. Retries with new run_id on duplicate."""
        run_id = result.run_id or EvalRun.new_run_id()
        run = EvalRun(
            run_id=run_id,
            dataset=result.dataset_name,
            model_id=result.model_id,
            metric_scores=result.metric_scores,
            per_example=result.per_example_results,
            failed_errors=getattr(result, "failed_errors", None) or [],
            total_examples=result.total_examples,
            failed_examples=result.failed_examples,
        )
        async with self._session_factory() as session:
            session.add(run)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                result.run_id = EvalRun.new_run_id()
                retry_run = EvalRun(
                    run_id=result.run_id,
                    dataset=result.dataset_name,
                    model_id=result.model_id,
                    metric_scores=result.metric_scores,
                    per_example=result.per_example_results,
                    failed_errors=getattr(result, "failed_errors", None) or [],
                    total_examples=result.total_examples,
                    failed_examples=result.failed_examples,
                )
                session.add(retry_run)
                await session.commit()

    async def get_run(self, run_id: str) -> EvalRunResult | None:
        """Retrieve a run by ID."""
        async with self._session_factory() as session:
            stmt = select(EvalRun).where(EvalRun.run_id == run_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _row_to_result(row)

    async def list_runs(
        self,
        dataset: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[EvalRunResult]:
        """List past runs with optional filters."""
        async with self._session_factory() as session:
            stmt = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
            if dataset:
                stmt = stmt.where(EvalRun.dataset == dataset)
            if model_id:
                stmt = stmt.where(EvalRun.model_id == model_id)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_row_to_result(r) for r in rows]
