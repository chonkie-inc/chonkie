"""Pytest configuration."""

import logging
import os

import pytest
import requests

logger = logging.getLogger(__name__)


def pytest_configure() -> None:
    """Global pytest configuration."""
    os.environ["CHONKIE_LOG"] = "unconfigured"


@pytest.fixture(autouse=True, scope="session")
def configure_hf_session(tmp_path_factory):
    """Configure a caching HTTP session, scoped to the test run, to avoid redundant network calls during tests."""
    try:
        from huggingface_hub import configure_http_backend
    except ImportError:  # HF_Hub not installed, so no need to configure anything
        return

    try:
        import requests_cache

        requests_cache_db = tmp_path_factory.mktemp("cache") / "hf_requests_cache"

        def _construct_session() -> requests.Session:
            return requests_cache.CachedSession(
                requests_cache_db,
                allowable_methods=["GET", "POST", "HEAD"],
                # Only cache typical success/redirect/known-not-found responses.
                # Error responses (4xx/5xx other than 404) are intentionally not cached
                # so that transient or environment-specific failures are not persisted
                # across tests within a session.
                allowable_codes=[200, 302, 307, 404],
            )

        configure_http_backend(_construct_session)
        logger.info("Configured HF Hub HTTP session with requests_cache at %s", requests_cache_db)
    except ImportError:  # Failed to import requests_cache, I guess
        pass
