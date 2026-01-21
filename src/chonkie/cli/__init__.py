"""CLI module for Chonkie."""

from .cli_utils import app


def main() -> None:
    """Entry point for the CLI."""
    app()


__all__ = ["app", "main"]
