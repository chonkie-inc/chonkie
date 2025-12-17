"""Test suite for OpenAIEmbeddings."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from chonkie.embeddings.openai import OpenAIEmbeddings


@pytest.fixture
def embedding_model() -> OpenAIEmbeddings:
    """Fixture to create an OpenAIEmbeddings instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def sample_text() -> str:
    """Fixture to create a sample text for testing."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> list[str]:
    """Fixture to create a list of sample texts for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_initialization_with_model_name(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings initializes with a model name."""
    assert embedding_model.model == "text-embedding-3-small"
    assert embedding_model.client is not None


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed")
def test_embed_single_text(
    mock_embed,
    embedding_model: OpenAIEmbeddings,
    sample_text: str,
) -> None:
    """Test that OpenAIEmbeddings correctly embeds a single text."""
    mock_embed.return_value = np.zeros(embedding_model.dimension)
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed_batch")
def test_embed_batch_texts(
    mock_embed_batch,
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that OpenAIEmbeddings correctly embeds a batch of texts."""
    mock_embed_batch.return_value = [np.zeros(embedding_model.dimension) for _ in sample_texts]
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(embedding.shape == (embedding_model.dimension,) for embedding in embeddings)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed_batch")
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.similarity")
def test_similarity(
    mock_similarity,
    mock_embed_batch,
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that OpenAIEmbeddings correctly calculates similarity between two embeddings."""
    mock_embed_batch.return_value = [np.zeros(embedding_model.dimension) for _ in sample_texts]
    mock_similarity.return_value = np.float32(0.5)
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_dimension_property(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings correctly calculates the dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_is_available() -> None:
    """Test that OpenAIEmbeddings correctly checks if it is available."""
    assert OpenAIEmbeddings()._is_available() is True


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_repr(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings correctly returns a string representation."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("OpenAIEmbeddings")


if __name__ == "__main__":
    pytest.main()
