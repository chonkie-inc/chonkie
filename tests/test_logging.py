"""Tests for Chonkie logging functionality."""

import logging
from functools import cache
from unittest.mock import Mock

import pytest

from chonkie.logger import (
    configure,
    disable,
    enable,
    get_logger,
    is_enabled,
)


@pytest.fixture(autouse=True)
def patched_get_logger(monkeypatch):
    """Patch logging.getLogger to return an unwired logger for testing."""

    # We want to return the same logger instance for the same name
    # during the same test; `@cache` is a dead simple way to do that.
    @cache
    def _get_logger(name="root"):
        return logging.Logger(f"_fake_{name}")

    mock = Mock(side_effect=_get_logger)
    monkeypatch.setattr(logging, "getLogger", mock)
    return mock


def test_get_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_module")
    assert logger.name == "_fake_test_module"


def test_configure_levels():
    """Test configuring different log levels."""
    # Test each level
    for level in ["off", "error", "warning", "info", "debug", "1", "2", "3", "4"]:
        configure(level)
        # If not "off", logging should be enabled
        if level not in ("off", "false", "0", "disabled"):
            assert is_enabled()


def test_disable_enable():
    """Test disabling and re-enabling logging."""
    # Enable first
    enable("info")
    assert is_enabled()

    # Disable
    disable()
    assert not is_enabled()

    # Re-enable
    enable("debug")
    assert is_enabled()


def test_chonkie_log_env_var():
    """Test that CHONKIE_LOG environment variable behavior via configure."""
    # Test with debug level
    configure("debug")

    # Should be enabled
    assert is_enabled()


def test_chonkie_log_off_env_var():
    """Test that CHONKIE_LOG=off disables logging via configure."""
    # Test setting off via configure (which is what env var would do)
    configure("off")

    # Should be disabled
    assert not is_enabled()

    # Re-enable for other tests
    enable("info")
    assert is_enabled()


def test_numeric_log_levels():
    """Test numeric log level configuration."""
    # Test numeric levels
    configure("1")  # ERROR
    assert is_enabled()

    configure("2")  # WARNING
    assert is_enabled()

    configure("3")  # INFO
    assert is_enabled()

    configure("4")  # DEBUG
    assert is_enabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
