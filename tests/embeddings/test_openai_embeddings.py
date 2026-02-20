"""Test suite for OpenAIEmbeddings (CatsuEmbeddings wrapper)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.openai import OpenAIEmbeddings


@pytest.fixture
def mock_catsu_client():
    """Mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, 1536).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = "text-embedding-3-small"
    mock_model_info.dimensions = 1536
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create an OpenAIEmbeddings instance."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key="test-key")


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


def test_initialization(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings initializes correctly."""
    assert embedding_model.model == "text-embedding-3-small"
    assert embedding_model._catsu is not None


def test_initialization_with_env_var(mock_catsu_client) -> None:
    """Test initialization using environment variable for API key."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            embeddings = OpenAIEmbeddings()
            assert embeddings.model == "text-embedding-3-small"


def test_default_model() -> None:
    """Test that OpenAIEmbeddings has the correct default model."""
    assert OpenAIEmbeddings.DEFAULT_MODEL == "text-embedding-3-small"


def test_embed_single_text(embedding_model: OpenAIEmbeddings, sample_text: str, mock_catsu_client) -> None:
    """Test that embed delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 1536).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    result = embedding_model.embed(sample_text)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_embed_batch_texts(embedding_model: OpenAIEmbeddings, sample_texts: list, mock_catsu_client) -> None:
    """Test that embed_batch delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 1536).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    results = embedding_model.embed_batch(sample_texts)
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)


def test_embed_batch_empty(embedding_model: OpenAIEmbeddings) -> None:
    """Test embed_batch with empty list."""
    results = embedding_model.embed_batch([])
    assert results == []


def test_dimension_property(embedding_model: OpenAIEmbeddings) -> None:
    """Test that dimension property works."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 1536


def test_get_tokenizer(embedding_model: OpenAIEmbeddings) -> None:
    """Test that get_tokenizer returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


def test_similarity(embedding_model: OpenAIEmbeddings) -> None:
    """Test similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = embedding_model.similarity(v1, v2)
    assert isinstance(sim, np.float32)
    assert abs(sim - 1.0) < 1e-6


def test_repr(embedding_model: OpenAIEmbeddings) -> None:
    """Test string representation."""
    repr_str = repr(embedding_model)
    assert "OpenAIEmbeddings" in repr_str
    assert "text-embedding-3-small" in repr_str


def test_is_available() -> None:
    """Test _is_available method returns a bool."""
    assert isinstance(OpenAIEmbeddings._is_available(), bool)


def test_openai_embeddings_missing_dependencies() -> None:
    """Test that ImportError is raised when catsu is not available."""
    with patch.object(OpenAIEmbeddings, "_is_available", return_value=False):
        with pytest.raises(
            ImportError, match=r"One \(or more\) of the following packages is not available"
        ):
            OpenAIEmbeddings(api_key="test-key")


def test_catsu_initialized_with_correct_provider(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings is initialized with openai provider."""
    with patch("catsu.Client", return_value=mock_catsu_client) as mock_client_class:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="my-key") # noqa: F841
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs.get("api_keys") == {"openai": "my-key"}


def test_backward_compat_signature(mock_catsu_client) -> None:
    """Test that old __init__ signature parameters are accepted."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        # These old params should be accepted without error
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            tokenizer=None,
            dimension=None,
            max_tokens=None,
            base_url=None,
            api_key="test-key",
            organization=None,
            max_retries=3,
            timeout=60.0,
            batch_size=128,
        )
        assert embeddings.model == "text-embedding-3-small"


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping integration test - requires OPENAI_API_KEY",
)
def test_real_embed_integration():
    """Integration test with real API (requires OPENAI_API_KEY)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text = "This is a real integration test."
    embedding = embeddings.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


if __name__ == "__main__":
    pytest.main()
