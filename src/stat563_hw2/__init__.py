"""STAT 563 HW2 automation package."""

from . import correlation, logistic

__all__ = ["logistic", "correlation", "main"]


def main() -> None:
    """Simple entry point that prints available pipelines."""

    print("STAT 563 HW2 package ready. Available modules: logistic, correlation")
    print("Use 'uv run python scripts/run_logistic_pipeline.py' for Q1-Q3")
    print("Use 'uv run python scripts/run_correlation_pipeline.py' for Q4")
