"""Centralized logging configuration for Chonkie.

This module provides a simple, extensible logging interface using Python's standard logging.
Logging is disabled by default (library pattern) but can be customized
via the CHONKIE_LOG environment variable or programmatic API.

Environment Variable:
    CHONKIE_LOG: Control logging behavior
        - off/false/0/disabled/none: Disable logging
        - error/1: ERROR level only
        - warning/2: WARNING and above (default if enabled)
        - info/3: INFO and above
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

import logging
import os
import sys
from typing import Optional, Tuple

# Track if we've configured the logger
_configured = False
_enabled = True
_handler: Optional[logging.Handler] = None

# Default configuration
DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"


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

    # Default to WARNING if value is unclear (e.g., "true", "on", etc.)
    return True, DEFAULT_LOG_LEVEL


def _configure_default() -> None:
    """Configure logger with default settings if not already configured.

    Following Python logging best practices for libraries:
    - By default, add NullHandler (no output unless user configures)
    - Only add handlers if user explicitly opts in via CHONKIE_LOG env var
    - Support hierarchical loggers (e.g., chonkie.chunker.base)
    """
    global _configured, _enabled, _handler

    if _configured:
        return

    # Get the chonkie logger (parent of all chonkie.* loggers)
    logger = logging.getLogger("chonkie")

    # Library pattern: Add NullHandler by default (no output)
    # This prevents "No handlers found" warnings
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    # Parse CHONKIE_LOG environment variable
    chonkie_log = os.getenv("CHONKIE_LOG")
    enabled, level = _parse_log_setting(chonkie_log)
    _enabled = enabled

    # Only configure if user explicitly wants logs (CHONKIE_LOG is set and not disabled)
    if enabled and chonkie_log is not None:
        # Remove NullHandler if present
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]

        # Create and configure a StreamHandler
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setLevel(getattr(logging, level))

        # Set formatter
        formatter = logging.Formatter(DEFAULT_FORMAT)
        _handler.setFormatter(formatter)

        # Add handler to chonkie logger
        logger.addHandler(_handler)
        logger.setLevel(getattr(logging, level))

        # Prevent propagation to avoid duplicate logs from Python's root logger
        logger.propagate = False

    _configured = True


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter to support loguru-style keyword arguments.

    This allows backwards compatibility with code that uses loguru's pattern:
        logger.debug("message", key=value)

    Standard logging doesn't support this, so we capture kwargs and ignore them.
    """

    def process(self, msg, kwargs):
        # Remove any extra keyword arguments that aren't supported by standard logging
        # Standard logging only supports: exc_info, stack_info, stacklevel, extra
        valid_keys = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return msg, filtered_kwargs


def get_logger(module_name: str) -> LoggerAdapter:
    """Get a logger instance for a specific module.

    This function returns a standard Python logger with hierarchical naming,
    wrapped in an adapter for loguru compatibility.
    The logger is automatically configured on first use based on CHONKIE_LOG
    environment variable.

    Args:
        module_name: The name of the module requesting the logger (typically __name__)

    Returns:
        A LoggerAdapter instance for the module

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Detailed information")

    """
    _configure_default()

    # Return a logger with the module name wrapped in our adapter
    # This creates a hierarchy: chonkie.chunker.base inherits from chonkie.chunker and chonkie
    base_logger = logging.getLogger(module_name)
    return LoggerAdapter(base_logger, {})


def configure_logging(
    level: Optional[str] = None,
    format: Optional[str] = None,
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
            - None: Use CHONKIE_LOG env var or default to WARNING
        format: Optional custom format string. Uses default if None.

    Example:
        >>> configure_logging("debug")
        >>> configure_logging("off")  # Disable logging
        >>> configure_logging("info")  # Re-enable at INFO level

    """
    global _configured, _enabled, _handler

    # Parse the level setting
    enabled, log_level = _parse_log_setting(level)
    _enabled = enabled

    # Get the chonkie logger
    logger = logging.getLogger("chonkie")

    # Remove existing handlers
    logger.handlers = []
    _handler = None

    if not enabled:
        # Add NullHandler to suppress logs
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)  # Effectively disable
        _configured = True
        return

    # Use provided format or default
    log_format = format or DEFAULT_FORMAT

    # Create and configure handler
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(getattr(logging, log_level))

    # Set formatter
    formatter = logging.Formatter(log_format)
    _handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, log_level))

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

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


# Export the main functions
__all__ = [
    "get_logger",
    "configure_logging",
    "disable_logging",
    "enable_logging",
    "is_enabled",
]
