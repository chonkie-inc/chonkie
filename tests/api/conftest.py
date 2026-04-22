"""Pytest configuration for OSS API tests.

``DATABASE_URL`` must be set before ``chonkie.api.database`` is first imported
(module-level ``DATABASE_URL`` and engine creation).
"""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def pytest_configure() -> None:
    """Point the API at an isolated SQLite file before any test imports the app."""
    if os.environ.get("_CHONKIE_API_TEST_DB_READY"):
        return

    import atexit
    import shutil

    db_dir = Path(tempfile.mkdtemp(prefix="chonkie_api_"))
    atexit.register(shutil.rmtree, db_dir, ignore_errors=True)

    db_path = db_dir / "test.db"
    url_path = db_path.resolve().as_posix()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{url_path}"
    os.environ["_CHONKIE_API_TEST_DB_READY"] = "1"


@pytest.fixture
def api_client() -> TestClient:
    """Sync TestClient; runs ASGI lifespan (DB init) on enter/exit."""
    from chonkie.api.main import app

    with TestClient(app) as client:
        yield client
