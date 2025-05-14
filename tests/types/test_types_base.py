"""Tests for Base Types."""

from __future__ import annotations

import pytest

from chonkie import Chunk, Context


def test_context_init():
    """Test Context initialization."""
    context = Context(text="test", token_count=1)
    assert context.text == "test"
    assert context.token_count == 1
    assert context.start_index is None
    assert context.end_index is None


def test_context_raises_error():
    """Test Context raises error for illegal field values."""
    with pytest.raises(ValueError):
        Context(text=9000, token_count=1)

    with pytest.raises(ValueError):
        Context(text="test", token_count=-1, start_index="0")

    with pytest.raises(TypeError):
        Context(text="test", token_count="1")

    with pytest.raises(ValueError):
        Context(text="test", token_count=1, start_index=10, end_index=5)


def test_context_serialization():
    """Test Context serialization/deserialization."""
    context = Context(text="test", token_count=1, start_index=0, end_index=4)
    context_dict = context.to_dict()
    restored = Context.from_dict(context_dict)
    assert context.text == restored.text
    assert context.token_count == restored.token_count
    assert context.start_index == restored.start_index
    assert context.end_index == restored.end_index


# Chunk Tests
def test_chunk_init():
    """Test Chunk initialization."""
    chunk = Chunk(text="test chunk", start_index=0, end_index=10, token_count=2)
    assert chunk.text == "test chunk"
    assert chunk.start_index == 0
    assert chunk.end_index == 10
    assert chunk.token_count == 2
    assert chunk.context is None


def test_chunk_with_context():
    """Test Chunk with context."""
    context = Context(text="context", token_count=1)
    chunk = Chunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        context=context,
    )
    assert chunk.context == context


def test_chunk_serialization():
    """Test Chunk serialization/deserialization."""
    context = Context(text="context", token_count=1)
    chunk = Chunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        context=context,
    )
    chunk_dict = chunk.to_dict()
    restored = Chunk.from_dict(chunk_dict)
    assert chunk.text == restored.text
    assert chunk.token_count == restored.token_count
    assert chunk.context.text == restored.context.text
