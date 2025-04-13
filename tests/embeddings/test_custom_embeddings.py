"""Contains test cases for the CustomEmbeddings class.

The tests verify:

- Initialization with a specified dimension
- Embedding a single text string
- Embedding a batch of text strings
- Token counting
- Similarity calculation
"""

from typing import Callable

import numpy as np
import pytest

from chonkie.embeddings.base import BaseEmbeddings


class CustomEmbeddings(BaseEmbeddings):
    """Custom embeddings class."""

    def __init__(self, dimension: int = 4) -> None:
        """Initialize the CustomEmbeddings class."""
        super().__init__()
        self._dimension = dimension

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text string into a vector representation."""
        # For demonstration, returns a random vector
        return np.random.rand(self._dimension)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        # Very naive token counting—split by whitespace
        return len(text.split())

    def get_tokenizer_or_token_counter(self) -> Callable[[str], int]:
        """Return the tokenizer or token counter."""
        return self.count_tokens

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> float:
        """Calculate the cosine similarity between two vectors."""
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


def test_custom_embeddings_initialization() -> None:
    """Test the initialization of the CustomEmbeddings class."""
    embeddings = CustomEmbeddings(dimension=4)
    assert isinstance(embeddings, BaseEmbeddings)
    assert embeddings.dimension == 4


def test_custom_embeddings_single_text() -> None:
    """Test the embedding of a single text string."""
    embeddings = CustomEmbeddings(dimension=4)
    text = "Test string"
    vector = embeddings.embed(text)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (4,)


def test_custom_embeddings_batch_text() -> None:
    """Test the embedding of a batch of text strings."""
    embeddings = CustomEmbeddings(dimension=4)
    texts = ["Test string one", "Test string two"]
    vectors = embeddings.embed_batch(texts)
    assert len(vectors) == 2
    for vec in vectors:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4,)


def test_custom_embeddings_token_count() -> None:
    """Test the token counting functionality."""
    embeddings = CustomEmbeddings()
    text = "Test string for counting tokens"
    count = embeddings.count_tokens(text)
    assert isinstance(count, int)
    assert count == len(text.split())


def test_custom_embeddings_similarity() -> None:
    """Test the similarity calculation."""
    embeddings = CustomEmbeddings(dimension=4)
    vec1 = embeddings.embed("Text A")
    vec2 = embeddings.embed("Text B")
    sim = embeddings.similarity(vec1, vec2)
    # Cosine similarity is in [-1, 1]—random vectors often produce a small positive or negative value
    assert -1.0 <= sim <= 1.0


if __name__ == "__main__":
    pytest.main()
