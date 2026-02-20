"""Test suite for CohereEmbeddings (CatsuEmbeddings wrapper)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.cohere import CohereEmbeddings


@pytest.fixture
def mock_catsu_client():
    """Mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, 384).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = "embed-english-light-v3.0"
    mock_model_info.dimensions = 384
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create a CohereEmbeddings instance."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        return CohereEmbeddings(model="embed-english-light-v3.0", api_key="test-key")


@pytest.fixture
def sample_text() -> str:
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> list:
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization(embedding_model: CohereEmbeddings) -> None:
    """Test that CohereEmbeddings initializes correctly."""
    assert embedding_model.model == "embed-english-light-v3.0"
    assert embedding_model._catsu is not None


def test_initialization_with_env_var(mock_catsu_client) -> None:
    """Test initialization using environment variable for API key."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {"COHERE_API_KEY": "env-test-key"}):
            embeddings = CohereEmbeddings()
            assert embeddings.model == "embed-english-light-v3.0"


def test_default_model() -> None:
    """Test that CohereEmbeddings has the correct default model."""
    assert CohereEmbeddings.DEFAULT_MODEL == "embed-english-light-v3.0"


def test_embed_single_text(embedding_model: CohereEmbeddings, sample_text: str, mock_catsu_client) -> None:
    """Test that embed delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 384).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    result = embedding_model.embed(sample_text)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_embed_batch_texts(embedding_model: CohereEmbeddings, sample_texts: list, mock_catsu_client) -> None:
    """Test that embed_batch delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 384).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    results = embedding_model.embed_batch(sample_texts)
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)


def test_embed_batch_empty(embedding_model: CohereEmbeddings) -> None:
    """Test embed_batch with empty list."""
    results = embedding_model.embed_batch([])
    assert results == []


def test_dimension_property(embedding_model: CohereEmbeddings) -> None:
    """Test that dimension property works."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 384


def test_get_tokenizer(embedding_model: CohereEmbeddings) -> None:
    """Test that get_tokenizer returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


def test_similarity(embedding_model: CohereEmbeddings) -> None:
    """Test similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = embedding_model.similarity(v1, v2)
    assert isinstance(sim, np.float32)
    assert abs(sim - 1.0) < 1e-6


def test_repr(embedding_model: CohereEmbeddings) -> None:
    """Test string representation."""
    repr_str = repr(embedding_model)
    assert "CohereEmbeddings" in repr_str
    assert "embed-english-light-v3.0" in repr_str


def test_is_available() -> None:
    """Test _is_available method returns a bool."""
    assert isinstance(CohereEmbeddings._is_available(), bool)


def test_cohere_embeddings_missing_dependencies() -> None:
    """Test that ImportError is raised when catsu is not available."""
    with patch.object(CohereEmbeddings, "_is_available", return_value=False):
        with pytest.raises(
            ImportError, match=r"One \(or more\) of the following packages is not available"
        ):
            CohereEmbeddings(api_key="test-key")


def test_catsu_initialized_with_correct_provider(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings is initialized with cohere provider."""
    with patch("catsu.Client", return_value=mock_catsu_client) as mock_client_class:
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0", api_key="my-key")
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs.get("api_keys") == {"cohere": "my-key"}


def test_batch_size_capped_at_96(mock_catsu_client) -> None:
    """Test that batch size is capped at 96 for Cohere."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = CohereEmbeddings(api_key="test-key", batch_size=200)
        assert embeddings._catsu._batch_size <= 96


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping integration test - requires COHERE_API_KEY",
)
def test_real_api_integration():
    """Integration test with real API (requires COHERE_API_KEY)."""
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    text = "This is a real integration test."
    embedding = embeddings.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


if __name__ == "__main__":
    pytest.main()
