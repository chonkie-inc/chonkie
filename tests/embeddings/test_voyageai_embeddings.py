"""Test suite for VoyageAIEmbeddings (CatsuEmbeddings wrapper)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.voyageai import VoyageAIEmbeddings


@pytest.fixture
def mock_catsu_client():
    """Mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, 1024).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = "voyage-3"
    mock_model_info.dimensions = 1024
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create a VoyageAIEmbeddings instance."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        return VoyageAIEmbeddings(api_key="test-key")


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


def test_initialization(embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings initializes correctly."""
    assert embedding_model.model == "voyage-3"
    assert embedding_model._catsu is not None


def test_initialization_with_env_var(mock_catsu_client) -> None:
    """Test initialization using environment variable for API key."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "env-test-key"}):
            embeddings = VoyageAIEmbeddings()
            assert embeddings.model == "voyage-3"


def test_initialization_with_custom_model(mock_catsu_client) -> None:
    """Test initialization with a custom model."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = VoyageAIEmbeddings(model="voyage-3-large", api_key="test-key")
        assert embeddings.model == "voyage-3-large"


def test_default_model() -> None:
    """Test that VoyageAIEmbeddings has the correct default model."""
    assert VoyageAIEmbeddings.DEFAULT_MODEL == "voyage-3"


def test_embed_single_text(
    embedding_model: VoyageAIEmbeddings, sample_text: str, mock_catsu_client
) -> None:
    """Test that embed delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 1024).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    result = embedding_model.embed(sample_text)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_embed_batch_texts(
    embedding_model: VoyageAIEmbeddings, sample_texts: list, mock_catsu_client
) -> None:
    """Test that embed_batch delegates to CatsuEmbeddings."""
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 1024).astype(
        np.float32
    )
    mock_catsu_client.embed.return_value = mock_response

    results = embedding_model.embed_batch(sample_texts)
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)


def test_embed_batch_empty(embedding_model: VoyageAIEmbeddings) -> None:
    """Test embed_batch with empty list."""
    results = embedding_model.embed_batch([])
    assert results == []


def _mock_contextual_response(embeddings_by_input: list[list[list[float]]]):
    response = MagicMock()
    response.embeddings = embeddings_by_input
    return response


def test_contextual_embed_batch_uses_contextual_endpoint() -> None:
    """Contextual models embed chunks together as one document."""
    with patch("catsu.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.contextualized_embed.return_value = _mock_contextual_response(
            [[[0.1] * 1024, [0.2] * 1024]],
        )
        embeddings = VoyageAIEmbeddings(model="voyage-context-3", api_key="test-key")

        results = embeddings.embed_batch(["chunk 1", "chunk 2"])

    mock_client.contextualized_embed.assert_called_once_with(
        model="voyage-context-3",
        inputs=[["chunk 1", "chunk 2"]],
        provider="voyageai",
        input_type="document",
        dimensions=1024,
    )
    assert len(results) == 2
    assert np.allclose(results[0], [0.1] * 1024)


def test_contextual_embed_documents_preserves_document_groups() -> None:
    """Contextual models can embed multiple document chunk groups together."""
    with patch("catsu.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.contextualized_embed.return_value = _mock_contextual_response(
            [[[0.1] * 512, [0.2] * 512], [[0.3] * 512]],
        )
        embeddings = VoyageAIEmbeddings(
            model="voyage-context-3",
            api_key="test-key",
            output_dimension=512,
        )

        results = embeddings.embed_documents(
            [["doc 1 chunk 1", "doc 1 chunk 2"], ["doc 2 chunk 1"]],
        )

    mock_client.contextualized_embed.assert_called_once_with(
        model="voyage-context-3",
        inputs=[["doc 1 chunk 1", "doc 1 chunk 2"], ["doc 2 chunk 1"]],
        provider="voyageai",
        input_type="document",
        dimensions=512,
    )
    assert embeddings.dimension == 512
    assert len(results) == 3


def test_contextual_embed_single_text_uses_query_input_type() -> None:
    """Single contextual embeds are treated as query embeddings."""
    with patch("catsu.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.contextualized_embed.return_value = _mock_contextual_response(
            [[[0.1] * 1024]],
        )
        embeddings = VoyageAIEmbeddings(model="voyage-context-3", api_key="test-key")

        result = embeddings.embed("find this")

    mock_client.contextualized_embed.assert_called_once_with(
        model="voyage-context-3",
        inputs=[["find this"]],
        provider="voyageai",
        input_type="query",
        dimensions=1024,
    )
    assert result.shape == (1024,)


def test_contextual_invalid_output_dimension_raises() -> None:
    """Contextual models validate their supported output dimensions."""
    with pytest.raises(ValueError, match="Invalid output_dimension"):
        VoyageAIEmbeddings(
            model="voyage-context-3",
            api_key="test-key",
            output_dimension=300,  # type: ignore[arg-type]
        )


def test_dimension_property(embedding_model: VoyageAIEmbeddings) -> None:
    """Test that dimension property works."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 1024


def test_get_tokenizer(embedding_model: VoyageAIEmbeddings) -> None:
    """Test that get_tokenizer returns a tokenizer."""
    tokenizer = embedding_model.get_tokenizer()
    assert tokenizer is not None


def test_similarity(embedding_model: VoyageAIEmbeddings) -> None:
    """Test similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = embedding_model.similarity(v1, v2)
    assert isinstance(sim, np.float32)
    assert abs(sim - 1.0) < 1e-6


def test_repr(embedding_model: VoyageAIEmbeddings) -> None:
    """Test string representation."""
    repr_str = repr(embedding_model)
    assert "VoyageAIEmbeddings" in repr_str
    assert "voyage-3" in repr_str


def test_is_available() -> None:
    """Test _is_available method returns a bool."""
    assert isinstance(VoyageAIEmbeddings._is_available(), bool)


def test_voyageai_embeddings_missing_dependencies() -> None:
    """Test that ImportError is raised when catsu is not available."""
    with patch.object(VoyageAIEmbeddings, "_is_available", return_value=False):
        with pytest.raises(
            ImportError, match=r"One \(or more\) of the following packages is not available"
        ):
            VoyageAIEmbeddings(api_key="test-key")


def test_catsu_initialized_with_correct_provider(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings is initialized with voyageai provider."""
    with patch("catsu.Client", return_value=mock_catsu_client) as mock_client_class:
        embeddings = VoyageAIEmbeddings(api_key="my-key")  # noqa: F841
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs.get("api_keys") == {"voyageai": "my-key"}


def test_batch_size_capped_at_128(mock_catsu_client) -> None:
    """Test that batch size is capped at 128 for VoyageAI."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = VoyageAIEmbeddings(api_key="test-key", batch_size=256)
        assert embeddings._catsu._batch_size <= 128


def test_backward_compat_signature(mock_catsu_client) -> None:
    """Test that old __init__ signature parameters are accepted."""
    with patch("catsu.Client", return_value=mock_catsu_client):
        embeddings = VoyageAIEmbeddings(
            model="voyage-3",
            api_key="test-key",
            max_retries=3,
            timeout=60.0,
            output_dimension=None,
            batch_size=128,
            truncation=True,
        )
        assert embeddings.model == "voyage-3"


@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping integration test - requires VOYAGE_API_KEY",
)
def test_real_embed_integration():
    """Integration test with real API (requires VOYAGE_API_KEY)."""
    embeddings = VoyageAIEmbeddings(model="voyage-3")
    text = "This is a real integration test."
    embedding = embeddings.embed(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


if __name__ == "__main__":
    pytest.main()
