"""Centralized logging configuration for Chonkie.

This module provides a simple, extensible logging interface using loguru.
Logging is minimal by default (INFO level to stderr) but can be customized
via the CHONKIE_LOG environment variable or programmatic API.

Environment Variable:
    CHONKIE_LOG: Control logging behavior
        - off/false/0/disabled/none: Disable logging
        - error/1: ERROR level only
        - warning/2: WARNING and above
        - info/3: INFO and above (default)
        - debug/4: DEBUG and above (most verbose)

Example:
    >>> from chonkie.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing chunks")

    >>> # Configure programmatically
    >>> from chonkie.logger import configure_logging
    >>> configure_logging("debug")
    >>> configure_logging("off")  # Disable

"""

import os
import sys
from typing import Any, Optional, Tuple

from loguru import logger

# Track if we've configured the logger
_configured = False
_enabled = True
_handler_id: Optional[int] = None

# Default configuration
DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _parse_log_setting(value: Optional[str]) -> Tuple[bool, str]:
    """Parse CHONKIE_LOG environment variable.

    Args:
        value: The value of CHONKIE_LOG environment variable

    Returns:
        (enabled, level) tuple where enabled is bool and level is string

    """
    if not value:
        return True, DEFAULT_LOG_LEVEL

    value = value.lower().strip()

    # Handle disable cases
    if value in ("off", "false", "0", "disabled", "none"):
        return False, DEFAULT_LOG_LEVEL

    # Handle numeric levels
    level_map = {
        "1": "ERROR",
        "2": "WARNING",
        "3": "INFO",
        "4": "DEBUG",
    }
    if value in level_map:
        return True, level_map[value]

    # Handle string levels
    if value.upper() in ("ERROR", "WARNING", "INFO", "DEBUG"):
        return True, value.upper()

    # Default to INFO if value is unclear (e.g., "true", "on", etc.)
    return True, DEFAULT_LOG_LEVEL


def _configure_default() -> None:
    """Configure logger with default settings if not already configured."""
    global _configured, _enabled, _handler_id

    if _configured:
        return

    # Remove the library's specific handler if it exists
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:
            pass  # Handler already removed
    # NOTE: Do NOT call logger.remove() here as it would remove all handlers,
    # including those configured by the user. This was causing user's loguru
    # configuration to be overwritten when importing chonkie.

    # Parse CHONKIE_LOG environment variable
    enabled, level = _parse_log_setting(os.getenv("CHONKIE_LOG"))
    _enabled = enabled

    if not enabled:
        logger.disable("chonkie")
        _configured = True
        return

    # Add stderr handler with formatting and store its ID
    # Use filter to only process logs from chonkie modules to avoid
    # interfering with user's logging configuration
    _handler_id = logger.add(
        sys.stderr,
        format=DEFAULT_FORMAT,
        level=level,
        colorize=True,
        enqueue=True,  # Thread-safe
        filter=lambda record: record["extra"].get("name", "").startswith("chonkie"),
    )

    _configured = True


def get_logger(module_name: str) -> Any:
    """Get a logger instance for a specific module.

    This function returns a loguru logger bound with the module name as context.
    The logger is automatically configured on first use based on CHONKIE_LOG
    environment variable.

    Args:
        module_name: The name of the module requesting the logger (typically __name__)

    Returns:
        A loguru logger instance bound with the module name

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Detailed information", chunk_count=10)

    """
    _configure_default()

    # Bind the module name as context
    return logger.bind(name=module_name)


def configure_logging(
    level: Optional[str] = None,
    format: Optional[str] = None,
    enqueue: bool = True,
) -> None:
    """Configure Chonkie's logging system programmatically.

    This function allows you to override the CHONKIE_LOG environment variable.
    Can be called multiple times to reconfigure logging.

    Args:
        level: Log level or control string:
            - "off"/"false"/"0"/"disabled": Disable logging
            - "error"/"1": ERROR level only
            - "warning"/"2": WARNING and above
            - "info"/"3": INFO and above
            - "debug"/"4": DEBUG and above
            - None: Use CHONKIE_LOG env var or default to INFO
        format: Optional custom format string. Uses default if None.
        enqueue: Whether to make logging thread-safe. Defaults to True.

    Example:
        >>> configure_logging("debug")
        >>> configure_logging("off")  # Disable logging
        >>> configure_logging("info")  # Re-enable at INFO level

    """
    global _configured, _enabled, _handler_id

    # Remove the library's specific handler if it exists
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:
            pass  # Handler already removed

    # Parse the level setting
    enabled, log_level = _parse_log_setting(level)
    _enabled = enabled

    if not enabled:
        logger.disable("chonkie")
        _configured = True
        return

    # Use provided format or default
    log_format = format or DEFAULT_FORMAT

    # Add stderr handler and store its ID
    # Use filter to only process logs from chonkie modules to avoid
    # interfering with user's logging configuration
    _handler_id = logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        enqueue=enqueue,
        filter=lambda record: record["extra"].get("name", "").startswith("chonkie"),
    )

    # Re-enable if it was previously disabled
    logger.enable("chonkie")
    _configured = True


def disable_logging() -> None:
    """Disable all Chonkie logging.

    This is equivalent to configure_logging("off").
    Useful for suppressing logs in production or testing environments.

    Example:
        >>> disable_logging()
        >>> # No logs will be output

    """
    configure_logging("off")


def enable_logging(level: str = "INFO") -> None:
    """Re-enable Chonkie logging after it has been disabled.

    Args:
        level: The log level to enable. Defaults to INFO.

    Example:
        >>> disable_logging()
        >>> # ... do some work without logs ...
        >>> enable_logging("debug")
        >>> # Logs are back at DEBUG level

    """
    configure_logging(level)


def is_enabled() -> bool:
    """Check if logging is currently enabled.

    Returns:
        True if logging is enabled, False otherwise

    Example:
        >>> if is_enabled():
        ...     logger.debug("This will be logged")

    """
    return _enabled


# Export the main logger for direct use if needed
__all__ = [
    "get_logger",
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "is_enabled",
    "logger",
]