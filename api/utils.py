"""Utility helpers for the Chonkie OSS API.

Provides a simple Timer for performance logging and a structured logger
that follows the same conventions as the cloud API.
"""

import logging
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class _StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports keyword-argument structured fields.

    Usage::

        log = get_logger("api.routes.chunking")
        log.info("Request received", endpoint="/v1/chunk/token", chunk_size=512)
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:  # type: ignore[override]
        extra = kwargs.pop("extra", {})
        # Pull any remaining kwargs out and append them as key=value pairs
        fields = {k: v for k, v in kwargs.items() if k not in {"exc_info", "stack_info", "stacklevel"}}
        for key in fields:
            kwargs.pop(key)
        if fields:
            field_str = "  " + "  ".join(f"{k}={v!r}" for k, v in fields.items())
            msg = msg + field_str
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> _StructuredLogger:
    """Return a structured logger for the given module name.

    Args:
        name: Module name (e.g. ``"api.routes.chunking"``).

    Returns:
        A :class:`_StructuredLogger` instance.
    """
    base = logging.getLogger(name)
    return _StructuredLogger(base, {})


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a human-readable format.

    Args:
        level: Log level string (``"DEBUG"``, ``"INFO"``, etc.).
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Lightweight wall-clock timer with named sub-timers.

    Usage::

        timer = Timer()
        timer.start()           # start the global timer

        timer.start("init")     # start a named timer
        ...
        ms = timer.end("init")  # stop and return elapsed ms

        total_ms = timer.end()  # stop global timer
    """

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}

    def start(self, name: str = "_global") -> None:
        """Start (or restart) a named timer.

        Args:
            name: Timer name.  Defaults to the global timer.
        """
        self._starts[name] = time.perf_counter()

    def end(self, name: str = "_global") -> float:
        """Stop a named timer and return elapsed milliseconds.

        Args:
            name: Timer name.  Defaults to the global timer.

        Returns:
            Elapsed time in milliseconds, or ``0.0`` if the timer was never
            started.
        """
        start = self._starts.pop(name, None)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000.0

    def elapsed(self, name: str = "_global") -> float:
        """Return elapsed milliseconds without stopping the timer.

        Args:
            name: Timer name.

        Returns:
            Elapsed time in milliseconds, or ``0.0`` if not started.
        """
        start = self._starts.get(name)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def fix_escaped_text(text: "str | list[str]") -> "str | list[str]":
    """Unescape common escape sequences that arrive double-escaped over JSON.

    Args:
        text: A string or list of strings to fix.

    Returns:
        The fixed string or list of strings.
    """
    if isinstance(text, list):
        return [_fix_single(t) for t in text]
    return _fix_single(text)


def _fix_single(text: str) -> str:
    replacements = [
        ("\\n", "\n"),
        ("\\t", "\t"),
        ("\\r", "\r"),
    ]
    for escaped, real in replacements:
        if escaped in text:
            text = text.replace(escaped, real)
    return text


def sanitize_text_encoding(text: str) -> str:
    """Replace invalid UTF-8 byte sequences with the replacement character.

    Args:
        text: Input string.

    Returns:
        Sanitized string safe for processing.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
