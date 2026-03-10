"""FastAPI application for evaluation lab."""

from eval_lab.api.app import app, create_app

__all__ = ["create_app", "app"]


def run_server() -> None:
    """Entry point for eval-api script."""
    import uvicorn
    uvicorn.run("eval_lab.api.app:app", host="0.0.0.0", port=8000, reload=False)
