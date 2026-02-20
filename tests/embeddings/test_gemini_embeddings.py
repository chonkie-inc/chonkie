"""Test suite for GeminiEmbeddings (CatsuEmbeddings wrapper)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.gemini import GeminiEmbeddings


@pytest.fixture
def mock_catsu_client():
    """Mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, 3072).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = "gemini-embedding-exp-03-07"
    mock_model_info.dimensions = 3072
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create a GeminiEmbeddings instance."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        return GeminiEmbeddings(api_key="test-key")


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


def test_initialization(embedding_model: GeminiEmbeddings) -> None:
    """Test that GeminiEmbeddings initializes correctly."""
    assert embedding_model.model == "gemini-embedding-exp-03-07"
    assert embedding_model.task_type == "SEMANTIC_SIMILARITY"
    assert embedding_model._catsu is not None


def test_initialization_with_env_var(mock_catsu_client) -> None:
    """Test initialization using environment variable for API key."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-test-key"}):
            embeddings = GeminiEmbeddings()
            assert embeddings.model == "gemini-embedding-exp-03-07"


def test_initialization_without_api_key(mock_catsu_client) -> None:
    """Test that GeminiEmbeddings can initialize without API key (validation deferred to API call)."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {}, clear=True):
            # With catsu wrapper, API key validation is deferred to the actual API call
            embeddings = GeminiEmbeddings()
            assert embeddings.model == "gemini-embedding-exp-03-07"


def test_initialization_with_custom_model(mock_catsu_client) -> None:
    """Test initialization with a custom model."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = GeminiEmbeddings(model="text-embedding-004", api_key="test-key")
        assert embeddings.model == "text-embedding-004"


def test_default_model() -> None:
    """Test that GeminiEmbeddings has the correct default model."""
    assert GeminiEmbeddings.DEFAULT_MODEL == "gemini-embedding-exp-03-07"


def test_embed_single_text(embedding_model: GeminiEmbeddings, sample_text: str, mock_catsu_client) -> None:
    """Test that embed delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 3072).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    result = embedding_model.embed(sample_text)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_embed_batch_texts(embedding_model: GeminiEmbeddings, sample_texts: list, mock_catsu_client) -> None:
    """Test that embed_batch delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 3072).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    results = embedding_model.embed_batch(sample_texts)
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)


def test_embed_batch_empty(embedding_model: GeminiEmbeddings) -> None:
    """Test embed_batch with empty list."""
    results = embedding_model.embed_batch([])
    assert results == []


def test_dimension_property(embedding_model: GeminiEmbeddings) -> None:
    """Test that dimension property works."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 3072


def test_get_tokenizer(embedding_model: GeminiEmbeddings) -> None:
    """Test that get_tokenizer returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


def test_similarity(embedding_model: GeminiEmbeddings) -> None:
    """Test similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = embedding_model.similarity(v1, v2)
    assert isinstance(sim, np.float32)
    assert abs(sim - 1.0) < 1e-6


def test_repr(embedding_model: GeminiEmbeddings) -> None:
    """Test string representation."""
    repr_str = repr(embedding_model)
    assert "GeminiEmbeddings" in repr_str
    assert "gemini-embedding-exp-03-07" in repr_str
    assert "SEMANTIC_SIMILARITY" in repr_str


def test_is_available() -> None:
    """Test _is_available method returns a bool."""
    assert isinstance(GeminiEmbeddings._is_available(), bool)


def test_import_dependencies_failure() -> None:
    """Test that ImportError is raised when catsu is not available."""
    with patch.object(GeminiEmbeddings, "_is_available", return_value=False):
        with pytest.raises(
            ImportError, match=r"One \(or more\) of the following packages is not available"
        ):
            GeminiEmbeddings(api_key="test-key")


def test_catsu_initialized_with_correct_provider(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings is initialized with gemini provider."""
    with patch("catsu.Client", return_value=mock_catsu_client) as mock_client_class:
        embeddings = GeminiEmbeddings(api_key="my-key")
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs.get("api_keys") == {"gemini": "my-key"}


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Skipping integration test - requires GEMINI_API_KEY",
)
def test_real_embed_integration():
    """Integration test with real API (requires GEMINI_API_KEY)."""
    embeddings = GeminiEmbeddings()
    text = "This is a real integration test."
    embedding = embeddings.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


if __name__ == "__main__":
    pytest.main()
