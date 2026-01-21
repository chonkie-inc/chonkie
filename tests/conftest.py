"""Pytest configuration."""

import os


def pytest_configure() -> None:
    """Global pytest configuration."""
    os.environ["CHONKIE_LOG"] = "unconfigured"
