"""Storage factory—returns store based on DATABASE_URL environment variable."""

import os

from eval_lab.storage.async_store import AsyncEvalStore
from eval_lab.storage.types import EvalStore


def get_store() -> EvalStore:
    """
    Return storage backend based on DATABASE_URL environment variable.

    Supports sqlite+aiosqlite and postgresql+asyncpg URLs.
    """
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///eval_runs.sqlite")
    url = _normalize_url(url)
    return AsyncEvalStore(url)


def _normalize_url(url: str) -> str:
    """Ensure URL uses correct async driver for SQLAlchemy."""
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url
