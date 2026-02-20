"""Test suite for JinaEmbeddings (CatsuEmbeddings wrapper)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.jina import JinaEmbeddings


@pytest.fixture
def mock_catsu_client():
    """Mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, 2048).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = "jina-embeddings-v4"
    mock_model_info.dimensions = 2048
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create a JinaEmbeddings instance."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        return JinaEmbeddings(api_key="test-key")


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


def test_initialization(embedding_model: JinaEmbeddings) -> None:
    """Test that JinaEmbeddings initializes correctly."""
    assert embedding_model.model == "jina-embeddings-v4"
    assert embedding_model.task == "text-matching"
    assert embedding_model._catsu is not None


def test_initialization_with_env_var(mock_catsu_client) -> None:
    """Test initialization using environment variable for API key."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {"JINA_API_KEY": "env-test-key"}):
            embeddings = JinaEmbeddings()
            assert embeddings.model == "jina-embeddings-v4"


def test_initialization_with_custom_model(mock_catsu_client) -> None:
    """Test initialization with a custom model."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = JinaEmbeddings(model="jina-embeddings-v3", api_key="test-key")
        assert embeddings.model == "jina-embeddings-v3"


def test_default_model() -> None:
    """Test that JinaEmbeddings has the correct default model."""
    assert JinaEmbeddings.DEFAULT_MODEL == "jina-embeddings-v4"


def test_embed_single_text(embedding_model: JinaEmbeddings, sample_text: str, mock_catsu_client) -> None:
    """Test that embed delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 2048).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    result = embedding_model.embed(sample_text)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_embed_batch_texts(embedding_model: JinaEmbeddings, sample_texts: list, mock_catsu_client) -> None:
    """Test that embed_batch delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 2048).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    results = embedding_model.embed_batch(sample_texts)
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)


def test_embed_batch_empty(embedding_model: JinaEmbeddings) -> None:
    """Test embed_batch with empty list."""
    results = embedding_model.embed_batch([])
    assert results == []


def test_dimension_property(embedding_model: JinaEmbeddings) -> None:
    """Test that dimension property works."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 2048


def test_get_tokenizer(embedding_model: JinaEmbeddings) -> None:
    """Test that get_tokenizer returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


def test_similarity(embedding_model: JinaEmbeddings) -> None:
    """Test similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = embedding_model.similarity(v1, v2)
    assert isinstance(sim, np.float32)
    assert abs(sim - 1.0) < 1e-6


def test_repr(embedding_model: JinaEmbeddings) -> None:
    """Test string representation."""
    repr_str = repr(embedding_model)
    assert "JinaEmbeddings" in repr_str
    assert "jina-embeddings-v4" in repr_str


def test_is_available() -> None:
    """Test _is_available method returns a bool."""
    assert isinstance(JinaEmbeddings._is_available(), bool)


def test_import_dependencies_failure() -> None:
    """Test that ImportError is raised when catsu is not available."""
    with patch.object(JinaEmbeddings, "_is_available", return_value=False):
        with pytest.raises(
            ImportError, match=r"One \(or more\) of the following packages is not available"
        ):
            JinaEmbeddings(api_key="test-key")


def test_catsu_initialized_with_correct_provider(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings is initialized with jina provider."""
    with patch("catsu.Client", return_value=mock_catsu_client) as mock_client_class:
        embeddings = JinaEmbeddings(api_key="my-key") # noqa: F841
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs.get("api_keys") == {"jinaai": "my-key"}


def test_backward_compat_signature(mock_catsu_client) -> None:
    """Test that old __init__ signature parameters are accepted."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = JinaEmbeddings(
            model="jina-embeddings-v4",
            task="text-matching",
            batch_size=32,
            max_retries=3,
            api_key="test-key",
        )
        assert embeddings.model == "jina-embeddings-v4"
        assert embeddings.task == "text-matching"


@pytest.mark.skipif(
    "JINA_API_KEY" not in os.environ,
    reason="Skipping integration test - requires JINA_API_KEY",
)
@pytest.mark.xfail(
    reason="catsu Jina provider key name may differ between published versions; xfail until stabilised",
    strict=False,
)
def test_real_embed_integration():
    """Integration test with real API (requires JINA_API_KEY)."""
    embeddings = JinaEmbeddings()
    text = "This is a real integration test."
    embedding = embeddings.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


if __name__ == "__main__":
    pytest.main()
