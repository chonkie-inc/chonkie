"""Tests for Base Types."""

from __future__ import annotations

import pytest

from chonkie import Chunk

try:
    import numpy as np  # noqa: F401
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# Context tests removed - context is now a simple string field in Chunk


# Context error tests removed - context is now a simple string field


# Context serialization tests removed - context is now a simple string field


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
    chunk = Chunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        context="context string",
    )
    assert chunk.context == "context string"


def test_chunk_serialization():
    """Test Chunk serialization/deserialization."""
    chunk = Chunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        context="context string",
    )
    chunk_dict = chunk.to_dict()
    restored = Chunk.from_dict(chunk_dict)
    assert chunk.text == restored.text
    assert chunk.token_count == restored.token_count
    assert chunk.context == restored.context


def test_chunk_with_embedding():
    """Test Chunk with embedding attribute."""
    # Test initialization without embedding
    chunk1 = Chunk(text="test", start_index=0, end_index=4, token_count=1)
    assert chunk1.embedding is None

    # Test initialization with list embedding
    chunk2 = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=[0.1, 0.2, 0.3]
    )
    assert chunk2.embedding == [0.1, 0.2, 0.3]


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
def test_chunk_with_numpy_embedding():
    """Test Chunk with numpy array embedding."""
    import numpy as np

    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    chunk = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=embedding
    )
    assert chunk.embedding is not None
    assert isinstance(chunk.embedding, np.ndarray)
    assert len(chunk.embedding) == 5


def test_chunk_embedding_repr():
    """Test Chunk __repr__ with embedding."""
    # Test without embedding
    chunk1 = Chunk(text="test", start_index=0, end_index=4, token_count=1)
    repr1 = repr(chunk1)
    assert "embedding" not in repr1

    # Test with short list embedding
    chunk2 = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=[0.1, 0.2, 0.3]
    )
    repr2 = repr(chunk2)
    assert "embedding" in repr2
    assert "0.1000" in repr2
    assert "..." not in repr2  # Should show all values for short embedding

    # Test with long list embedding
    chunk3 = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=[i * 0.1 for i in range(100)]
    )
    repr3 = repr(chunk3)
    assert "embedding" in repr3
    assert "..." in repr3  # Should truncate long embedding
    assert "0.0000" in repr3  # First value
    assert "9.8000" in repr3  # Second to last value


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
def test_chunk_numpy_embedding_repr():
    """Test Chunk __repr__ with numpy embedding shows shape."""
    import numpy as np

    embedding = np.random.randn(768)  # Typical embedding size
    chunk = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=embedding
    )
    repr_str = repr(chunk)
    assert "embedding" in repr_str
    assert "..." in repr_str  # Should truncate
    assert "shape=(768,)" in repr_str  # Should show shape


def test_chunk_embedding_serialization():
    """Test Chunk serialization with embedding."""
    # Test with list embedding
    chunk1 = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=[0.1, 0.2, 0.3]
    )
    dict1 = chunk1.to_dict()
    assert "embedding" in dict1
    assert dict1["embedding"] == [0.1, 0.2, 0.3]

    restored1 = Chunk.from_dict(dict1)
    assert restored1.embedding == [0.1, 0.2, 0.3]


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
def test_chunk_numpy_embedding_serialization():
    """Test Chunk serialization with numpy embedding."""
    import numpy as np

    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    chunk = Chunk(
        text="test",
        start_index=0,
        end_index=4,
        token_count=1,
        embedding=embedding
    )

    # to_dict should convert numpy to list
    dict_repr = chunk.to_dict()
    assert "embedding" in dict_repr
    assert isinstance(dict_repr["embedding"], list)
    assert dict_repr["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    # from_dict keeps as list (numpy conversion happens elsewhere if needed)
    restored = Chunk.from_dict(dict_repr)
    assert restored.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
