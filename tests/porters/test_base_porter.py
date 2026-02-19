"""Tests for BasePorter abstract interface."""

import pytest

from chonkie import Chunk
from chonkie.porters.base import BasePorter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(text: str = "hello") -> Chunk:
    return Chunk(text=text, start_index=0, end_index=len(text), token_count=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_base_porter_is_abstract():
    """Instantiating BasePorter directly raises TypeError."""
    with pytest.raises(TypeError):
        BasePorter()  # type: ignore[abstract]


def test_base_porter_subclass_without_export_raises():
    """A concrete subclass that omits export() raises TypeError on instantiation."""

    class Incomplete(BasePorter):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_base_porter_call_delegates_to_export():
    """__call__ on a concrete subclass invokes export() with the chunk list."""
    received: list = []

    class ConcretePorter(BasePorter):
        def export(self, chunks, **kwargs):
            received.extend(chunks)

    porter = ConcretePorter()
    chunks = [make_chunk("a"), make_chunk("b")]
    porter(chunks)
    assert received == chunks


def test_base_porter_export_receives_chunks():
    """export() in a concrete implementation receives the exact chunk list."""

    class RecordingPorter(BasePorter):
        def __init__(self):
            self.exported = []

        def export(self, chunks, **kwargs):
            self.exported = chunks

    porter = RecordingPorter()
    chunks = [make_chunk("x"), make_chunk("y"), make_chunk("z")]
    porter.export(chunks)
    assert porter.exported == chunks


def test_base_porter_export_returns_none():
    """export() implicitly returns None (no return value contract)."""

    class NullPorter(BasePorter):
        def export(self, chunks, **kwargs):
            pass

    porter = NullPorter()
    result = porter.export([make_chunk()])
    assert result is None
