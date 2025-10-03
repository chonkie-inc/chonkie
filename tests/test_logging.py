"""Tests for Chonkie logging functionality."""


import pytest

from chonkie.logger import (
    configure_logging,
    disable_logging,
    enable_logging,
    get_logger,
    is_enabled,
)


def test_get_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_module")
    assert logger is not None


def test_configure_logging_levels():
    """Test configuring different log levels."""
    # Test each level
    for level in ["off", "error", "warning", "info", "debug", "1", "2", "3", "4"]:
        configure_logging(level)
        # If not "off", logging should be enabled
        if level not in ("off", "false", "0", "disabled"):
            assert is_enabled()


def test_disable_enable_logging():
    """Test disabling and re-enabling logging."""
    # Enable first
    enable_logging("info")
    assert is_enabled()

    # Disable
    disable_logging()
    assert not is_enabled()

    # Re-enable
    enable_logging("debug")
    assert is_enabled()


def test_logging_with_chunker():
    """Test that logging works with chunkers."""
    from chonkie import TokenChunker

    # Enable logging at INFO level
    configure_logging("info")

    # Create a chunker and process some text
    chunker = TokenChunker(chunk_size=10)
    chunks = chunker.chunk("This is a test sentence.")

    assert len(chunks) > 0


def test_logging_disabled_with_chunker():
    """Test that logging can be disabled."""
    from chonkie import TokenChunker

    # Disable logging
    disable_logging()
    assert not is_enabled()

    # Create a chunker - should work without logs
    chunker = TokenChunker(chunk_size=10)
    chunks = chunker.chunk("This is a test sentence.")

    assert len(chunks) > 0

    # Re-enable for other tests
    enable_logging("info")


def test_chonkie_log_env_var():
    """Test that CHONKIE_LOG environment variable behavior via configure_logging."""
    # Test with debug level
    configure_logging("debug")

    # Should be enabled
    assert is_enabled()


def test_chonkie_log_off_env_var():
    """Test that CHONKIE_LOG=off disables logging via configure_logging."""
    # Test setting off via configure_logging (which is what env var would do)
    configure_logging("off")

    # Should be disabled
    assert not is_enabled()

    # Re-enable for other tests
    enable_logging("info")
    assert is_enabled()


def test_numeric_log_levels():
    """Test numeric log level configuration."""
    # Test numeric levels
    configure_logging("1")  # ERROR
    assert is_enabled()

    configure_logging("2")  # WARNING
    assert is_enabled()

    configure_logging("3")  # INFO
    assert is_enabled()

    configure_logging("4")  # DEBUG
    assert is_enabled()


def test_batch_processing_logs():
    """Test that batch processing generates appropriate logs."""
    from chonkie import SentenceChunker

    configure_logging("info")

    chunker = SentenceChunker(chunk_size=50)
    texts = ["First sentence.", "Second sentence.", "Third sentence."]

    # Process batch
    results = chunker.chunk_batch(texts, show_progress=False)

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])