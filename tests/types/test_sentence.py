"""Unit tests for Sentence class."""

import pytest

from chonkie import Sentence


def test_sentence_init():
    """Test Sentence initialization."""
    sentence = Sentence(text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=3)
    assert sentence.text == "Ratatouille is a movie."
    assert sentence.start_index == 0
    assert sentence.end_index == 14
    assert sentence.token_count == 3


def test_sentence_raises_error():
    """Test Sentence raises error for illegal field values."""
    with pytest.raises(ValueError):
        Sentence(text=9000, start_index=0, end_index=14, token_count=3)

    with pytest.raises(ValueError):
        Sentence(text="Ratatouille is a movie.", start_index=-1, end_index=14, token_count=3)

    with pytest.raises(ValueError):
        Sentence(text="Ratatouille is a movie.", start_index=0, end_index=-1, token_count=3)

    with pytest.raises(ValueError):
        Sentence(text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=-1)

    with pytest.raises(ValueError):
        Sentence(
            text="Ratatouille is a movie.",
            start_index=10,
            end_index=5,
            token_count=3,
        )


def test_sentence_serialization():
    """Test Sentence serialization/deserialization."""
    sentence = Sentence(text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=3)
    sentence_dict = sentence.to_dict()
    restored = Sentence.from_dict(sentence_dict)
    assert sentence.text == restored.text
    assert sentence.token_count == restored.token_count
