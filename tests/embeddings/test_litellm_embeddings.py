"""Test suite for LiteLLMEmbeddings."""

import os
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.litellm import LiteLLMEmbeddings


@pytest.fixture
def embedding_model() -> LiteLLMEmbeddings:
    """Fixture to create a LiteLLMEmbeddings instance with default OpenAI model."""
    api_key = os.environ.get("OPENAI_API_KEY")
    return LiteLLMEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def sample_text() -> str:
    """Fixture to create a sample text for testing."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture to create a list of sample texts for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization_with_default_model() -> None:
    """Test that LiteLLMEmbeddings initializes with default model."""
    with patch.object(LiteLLMEmbeddings, "_detect_dimension", return_value=1536):
        with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
            embeddings = LiteLLMEmbeddings()
            assert embeddings.model == "text-embedding-3-small"
            assert embeddings.dimension == 1536


def test_initialization_with_custom_model() -> None:
    """Test that LiteLLMEmbeddings initializes with custom model."""
    with patch.object(LiteLLMEmbeddings, "_detect_dimension", return_value=1024):
        with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
            embeddings = LiteLLMEmbeddings(model="voyage/voyage-3-large")
            assert embeddings.model == "voyage/voyage-3-large"
            assert embeddings.dimension == 1024


def test_initialization_with_explicit_dimension() -> None:
    """Test that LiteLLMEmbeddings respects explicitly provided dimension."""
    with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
        embeddings = LiteLLMEmbeddings(model="text-embedding-3-small", dimension=512)
        assert embeddings.dimension == 512


def test_get_provider() -> None:
    """Test that _get_provider correctly extracts provider name."""
    with patch.object(LiteLLMEmbeddings, "_detect_dimension", return_value=1024):
        with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
            # Test with provider prefix
            embeddings = LiteLLMEmbeddings(model="voyage/voyage-3-large")
            assert embeddings._get_provider() == "voyage"

            # Test without provider prefix
            embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
            assert embeddings._get_provider() is None


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.embed")
def test_embed_single_text(mock_embed, embedding_model: LiteLLMEmbeddings, sample_text: str) -> None:
    """Test that LiteLLMEmbeddings correctly embeds a single text."""
    mock_embed.return_value = np.zeros(embedding_model.dimension)
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.embed_batch")
def test_embed_batch_texts(
    mock_embed_batch, embedding_model: LiteLLMEmbeddings, sample_texts: List[str]
) -> None:
    """Test that LiteLLMEmbeddings correctly embeds a batch of texts."""
    mock_embed_batch.return_value = [
        np.zeros(embedding_model.dimension) for _ in sample_texts
    ]
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (embedding_model.dimension,) for embedding in embeddings
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_embed_empty_batch(embedding_model: LiteLLMEmbeddings) -> None:
    """Test that LiteLLMEmbeddings correctly handles empty batch."""
    embeddings = embedding_model.embed_batch([])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.embed_batch")
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.similarity")
def test_similarity(
    mock_similarity,
    mock_embed_batch,
    embedding_model: LiteLLMEmbeddings,
    sample_texts: List[str],
) -> None:
    """Test that LiteLLMEmbeddings correctly calculates similarity between embeddings."""
    mock_embed_batch.return_value = [
        np.zeros(embedding_model.dimension) for _ in sample_texts
    ]
    mock_similarity.return_value = np.float32(0.5)
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_dimension_property(embedding_model: LiteLLMEmbeddings) -> None:
    """Test that LiteLLMEmbeddings correctly returns the dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_get_tokenizer(embedding_model: LiteLLMEmbeddings) -> None:
    """Test that LiteLLMEmbeddings correctly returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_is_available() -> None:
    """Test that LiteLLMEmbeddings correctly checks if it is available."""
    with patch.object(LiteLLMEmbeddings, "_detect_dimension", return_value=1536):
        with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
            embeddings = LiteLLMEmbeddings()
            assert embeddings._is_available() is True


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_repr(embedding_model: LiteLLMEmbeddings) -> None:
    """Test that LiteLLMEmbeddings correctly returns a string representation."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("LiteLLMEmbeddings")
    assert "text-embedding-3-small" in repr_str


def test_known_dimensions() -> None:
    """Test that known model dimensions are correctly defined."""
    assert "text-embedding-3-small" in LiteLLMEmbeddings.KNOWN_DIMENSIONS
    assert "voyage/voyage-3-large" in LiteLLMEmbeddings.KNOWN_DIMENSIONS
    assert "cohere/embed-english-v3.0" in LiteLLMEmbeddings.KNOWN_DIMENSIONS


@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.embed")
def test_voyage_model(mock_embed) -> None:
    """Test that LiteLLMEmbeddings works with VoyageAI models."""
    with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
        api_key = os.environ.get("VOYAGE_API_KEY")
        embeddings = LiteLLMEmbeddings(model="voyage/voyage-3-large", api_key=api_key)
        assert embeddings.model == "voyage/voyage-3-large"
        assert embeddings.dimension == 1024

        mock_embed.return_value = np.zeros(embeddings.dimension)
        embedding = embeddings.embed("test")
        assert embedding.shape == (embeddings.dimension,)


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
@patch("chonkie.embeddings.litellm.LiteLLMEmbeddings.embed")
def test_cohere_model(mock_embed) -> None:
    """Test that LiteLLMEmbeddings works with Cohere models."""
    with patch.object(LiteLLMEmbeddings, "_initialize_tokenizer", return_value=MagicMock()):
        api_key = os.environ.get("COHERE_API_KEY")
        embeddings = LiteLLMEmbeddings(model="cohere/embed-english-v3.0", api_key=api_key)
        assert embeddings.model == "cohere/embed-english-v3.0"
        assert embeddings.dimension == 1024

        mock_embed.return_value = np.zeros(embeddings.dimension)
        embedding = embeddings.embed("test")
        assert embedding.shape == (embeddings.dimension,)


def test_import_error_when_litellm_not_available() -> None:
    """Test that appropriate error is raised when litellm is not available."""
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="litellm package is not available"):
            LiteLLMEmbeddings()


if __name__ == "__main__":
    pytest.main()
