"""Unit tests for Sentence class."""

import pytest

from chonkie import Sentence

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def test_sentence_init():
    """Test Sentence initialization."""
    sentence = Sentence(text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=3)
    assert sentence.text == "Ratatouille is a movie."
    assert sentence.start_index == 0
    assert sentence.end_index == 14
    assert sentence.token_count == 3
    assert sentence.embedding is None


def test_sentence_invalid_text_type():
    """Test Sentence raises ValueError for non-string text."""
    with pytest.raises(ValueError):
        Sentence(text=9000, start_index=0, end_index=14, token_count=3)


def test_sentence_invalid_negative_start_index():
    """Test Sentence raises ValueError for negative start_index."""
    with pytest.raises(ValueError):
        Sentence(text="hello", start_index=-1, end_index=5, token_count=1)


def test_sentence_invalid_negative_end_index():
    """Test Sentence raises ValueError for negative end_index."""
    with pytest.raises(ValueError):
        Sentence(text="hello", start_index=0, end_index=-1, token_count=1)


def test_sentence_invalid_negative_token_count():
    """Test Sentence raises ValueError for negative token_count."""
    with pytest.raises(ValueError):
        Sentence(text="hello", start_index=0, end_index=5, token_count=-1)


def test_sentence_invalid_index_order():
    """Test Sentence raises ValueError when start_index > end_index."""
    with pytest.raises(ValueError):
        Sentence(text="hello", start_index=10, end_index=5, token_count=1)


def test_sentence_equal_indices_allowed():
    """Test that start_index == end_index is valid (empty span)."""
    sentence = Sentence(text="", start_index=5, end_index=5, token_count=0)
    assert sentence.start_index == sentence.end_index


def test_sentence_repr_contains_key_fields():
    """Test Sentence __repr__ contains text, indices, and token_count."""
    sentence = Sentence(text="Hello.", start_index=0, end_index=6, token_count=1)
    r = repr(sentence)
    assert "Hello." in r
    assert "start_index=0" in r
    assert "end_index=6" in r
    assert "token_count=1" in r


def test_sentence_repr_without_embedding_omits_embedding():
    """Test Sentence __repr__ does not show embedding when None."""
    sentence = Sentence(text="Hello.", start_index=0, end_index=6, token_count=1)
    assert "embedding" not in repr(sentence)


def test_sentence_repr_with_embedding_includes_it():
    """Test Sentence __repr__ includes embedding when set."""
    sentence = Sentence(
        text="Hello.", start_index=0, end_index=6, token_count=1, embedding=[0.1, 0.2]
    )
    assert "embedding" in repr(sentence)


def test_sentence_serialization_round_trip():
    """Test Sentence to_dict / from_dict round-trip preserves all fields."""
    sentence = Sentence(text="Ratatouille is a movie.", start_index=0, end_index=23, token_count=4)
    d = sentence.to_dict()
    restored = Sentence.from_dict(d)
    assert restored.text == sentence.text
    assert restored.start_index == sentence.start_index
    assert restored.end_index == sentence.end_index
    assert restored.token_count == sentence.token_count
    assert restored.embedding == sentence.embedding


def test_sentence_to_dict_with_list_embedding():
    """Test that to_dict preserves a list embedding as-is."""
    sentence = Sentence(
        text="hello", start_index=0, end_index=5, token_count=1, embedding=[0.1, 0.2, 0.3]
    )
    d = sentence.to_dict()
    assert d["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
def test_sentence_to_dict_converts_numpy_embedding():
    """Test that to_dict converts a numpy embedding to a list."""
    embedding = np.array([0.4, 0.5, 0.6])
    sentence = Sentence(
        text="hello", start_index=0, end_index=5, token_count=1, embedding=embedding
    )
    d = sentence.to_dict()
    assert isinstance(d["embedding"], list)
    assert d["embedding"] == pytest.approx([0.4, 0.5, 0.6])


def test_sentence_from_dict_with_embedding():
    """Test from_dict restores embedding field."""
    d = {
        "text": "hi",
        "start_index": 0,
        "end_index": 2,
        "token_count": 1,
        "embedding": [0.7, 0.8],
    }
    sentence = Sentence.from_dict(d)
    assert sentence.embedding == [0.7, 0.8]


def test_sentence_from_dict_without_embedding():
    """Test from_dict sets embedding to None when absent."""
    d = {"text": "hi", "start_index": 0, "end_index": 2, "token_count": 1}
    sentence = Sentence.from_dict(d)
    assert sentence.embedding is None
