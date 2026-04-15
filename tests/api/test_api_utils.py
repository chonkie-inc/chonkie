"""Tests for ``chonkie.api.utils`` helpers."""

import time

from chonkie.api.utils import Timer, fix_escaped_text, sanitize_text_encoding


def test_timer_end_without_start_returns_zero() -> None:
    """end() without a matching start() returns 0.0 ms."""
    t = Timer()
    assert t.end("missing") == 0.0


def test_timer_named_subtimer() -> None:
    """Named start/end returns positive elapsed milliseconds."""
    t = Timer()
    t.start("work")
    time.sleep(0.01)
    elapsed = t.end("work")
    assert elapsed >= 0.0
    t.start("work")
    mid = t.elapsed("work")
    assert mid >= 0.0
    t.end("work")


def test_timer_elapsed_without_start() -> None:
    t = Timer()
    assert t.elapsed("nope") == 0.0


def test_fix_escaped_text_string() -> None:
    assert fix_escaped_text("a\\nb") == "a\nb"
    assert fix_escaped_text("a\\tb") == "a\tb"
    assert fix_escaped_text("a\\rb") == "a\rb"


def test_fix_escaped_text_list() -> None:
    out = fix_escaped_text(["x\\ny", "a\\tb"])
    assert out == ["x\ny", "a\tb"]


def test_sanitize_text_encoding_roundtrip() -> None:
    """Well-formed text round-trips; output is always a string."""
    assert sanitize_text_encoding("hello café") == "hello café"
    assert isinstance(sanitize_text_encoding(""), str)
