"""Tests for BaseHandshake helpers."""

from typing import Union

import pytest

from chonkie.handshakes.base import BaseHandshake
from chonkie.types import Chunk


class _DummyHandshake(BaseHandshake):
    """Minimal concrete handshake for exercising ``__call__`` / ``awrite``."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.last_chunks: Union[Chunk, list[Chunk], None] = None

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> str:
        self.last_chunks = chunks
        if self.fail:
            raise RuntimeError("write failed")
        return "ok"


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


def test_coerce_flat_metadata_skips_none_values() -> None:
    """None values are omitted from coerced metadata."""
    flat = BaseHandshake._coerce_flat_metadata({"keep": 1, "skip": None})
    assert flat == {"keep": 1}


def test_merge_chunk_metadata_non_dict_metadata() -> None:
    """Non-dict ``chunk.metadata`` is treated as empty."""
    chunk = Chunk(text="x", start_index=0, end_index=1, token_count=1)
    object.__setattr__(chunk, "metadata", "not-a-dict")
    merged = BaseHandshake._merge_chunk_metadata(chunk, {"text": "x"})
    assert merged == {"text": "x"}


def test_generate_id_is_deterministic() -> None:
    """Same input string yields the same UUID string."""
    a = BaseHandshake._generate_id("same")
    b = BaseHandshake._generate_id("same")
    c = BaseHandshake._generate_id("different")
    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_awrite_delegates_to_write() -> None:
    """Awrite runs write in a thread."""
    h = _DummyHandshake()
    chunk = Chunk(text="a", start_index=0, end_index=1, token_count=1)
    out = await h.awrite(chunk)
    assert out == "ok"
    assert h.last_chunks == chunk


def test_call_invokes_write_success() -> None:
    """__call__ forwards to write and returns its result."""
    h = _DummyHandshake()
    chunk = Chunk(text="a", start_index=0, end_index=1, token_count=1)
    assert h(chunk) == "ok"
    assert h.last_chunks == chunk


def test_call_invokes_write_list() -> None:
    h = _DummyHandshake()
    chunks = [
        Chunk(text="a", start_index=0, end_index=1, token_count=1),
        Chunk(text="b", start_index=1, end_index=2, token_count=1),
    ]
    assert h(chunks) == "ok"
    assert h.last_chunks == chunks


def test_call_propagates_write_errors() -> None:
    """__call__ re-raises exceptions from write."""
    h = _DummyHandshake(fail=True)
    chunk = Chunk(text="a", start_index=0, end_index=1, token_count=1)
    with pytest.raises(RuntimeError, match="write failed"):
        h(chunk)


def test_call_rejects_invalid_input_type() -> None:
    """__call__ requires Chunk or sequence of Chunks."""
    h = _DummyHandshake()
    with pytest.raises(TypeError, match="Chunk or a sequence"):
        h(123)  # type: ignore[arg-type]
