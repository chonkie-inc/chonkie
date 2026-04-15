"""Tests for BaseHandshake helpers."""

from chonkie.handshakes.base import BaseHandshake
from chonkie.types import Chunk


def test_merge_chunk_metadata_precedence() -> None:
    """Handshake fields override keys present in chunk.metadata."""
    chunk = Chunk(
        text="hi",
        start_index=0,
        end_index=2,
        token_count=1,
        metadata={"filename": "a.txt", "token_count": 999},
    )
    merged = BaseHandshake._merge_chunk_metadata(
        chunk,
        {"text": chunk.text, "token_count": chunk.token_count},
    )
    assert merged["filename"] == "a.txt"
    assert merged["token_count"] == chunk.token_count


def test_coerce_flat_metadata() -> None:
    """Non-primitive metadata values become JSON strings."""
    merged = {"a": 1, "b": [1, 2], "c": "ok"}
    flat = BaseHandshake._coerce_flat_metadata(merged)
    assert flat["a"] == 1
    assert flat["c"] == "ok"
    assert flat["b"] == "[1, 2]"
