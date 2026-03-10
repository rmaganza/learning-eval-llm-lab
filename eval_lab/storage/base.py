"""SQLAlchemy ORM models for evaluation run persistence."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base for all models."""

    pass


class EvalRun(Base):
    """Evaluation run metadata."""

    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(256), nullable=False)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    metric_scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    per_example: Mapped[list | None] = mapped_column(JSON, nullable=True)
    failed_errors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    total_examples: Mapped[int] = mapped_column(default=0)
    failed_examples: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    results: Mapped[list["EvalResult"]] = relationship(
        "EvalResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    @staticmethod
    def new_run_id() -> str:
        """Generate a new UUID for run_id."""
        return str(uuid4())


class EvalResult(Base):
    """Per-example evaluation result (optional granular storage)."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("eval_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    example_id: Mapped[str] = mapped_column(String(256), nullable=False)
    metric_scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="results")
