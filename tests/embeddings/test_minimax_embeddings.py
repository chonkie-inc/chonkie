"""Test suite for MiniMaxEmbeddings."""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from chonkie.embeddings.minimax import MiniMaxEmbeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_api_response(vectors=None, n=1):
    """Helper to create a mock API response."""
    if vectors is None:
        vectors = [np.random.rand(1536).tolist() for _ in range(n)]
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "vectors": vectors,
        "total_tokens": 5 * n,
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_api_response():
    """Create a mock API response with a single embedding vector."""
    return _make_api_response(n=1)


@pytest.fixture
def embedding_model(mock_api_response):
    """Create a MiniMaxEmbeddings instance with a mocked httpx client."""
    mock_client = MagicMock()
    mock_client.post.return_value = mock_api_response
    with patch("httpx.Client", return_value=mock_client):
        model = MiniMaxEmbeddings(api_key="test-minimax-key")
    # Replace the real client with our mock
    model._client = mock_client
    return model


@pytest.fixture
def sample_text() -> str:
    return "This is a sample text for testing MiniMax embeddings."


@pytest.fixture
def sample_texts() -> list:
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestMiniMaxEmbeddingsImportAndConstruction:
    """Test import and basic construction."""

    def test_import(self) -> None:
        """Test that MiniMaxEmbeddings can be imported."""
        from chonkie import MiniMaxEmbeddings as ME

        assert ME is not None

    def test_is_subclass_of_base(self) -> None:
        """Test that MiniMaxEmbeddings is a subclass of BaseEmbeddings."""
        from chonkie import BaseEmbeddings

        assert issubclass(MiniMaxEmbeddings, BaseEmbeddings)

    def test_has_required_methods(self) -> None:
        """Test that MiniMaxEmbeddings has all required methods."""
        assert hasattr(MiniMaxEmbeddings, "embed")
        assert hasattr(MiniMaxEmbeddings, "embed_batch")
        assert hasattr(MiniMaxEmbeddings, "dimension")
        assert hasattr(MiniMaxEmbeddings, "get_tokenizer")
        assert hasattr(MiniMaxEmbeddings, "_is_available")

    def test_default_model(self) -> None:
        """Test the default model name."""
        assert MiniMaxEmbeddings.DEFAULT_MODEL == "embo-01"

    def test_available_models(self) -> None:
        """Test available models dict."""
        assert "embo-01" in MiniMaxEmbeddings.AVAILABLE_MODELS
        assert MiniMaxEmbeddings.AVAILABLE_MODELS["embo-01"] == 1536


class TestMiniMaxEmbeddingsErrorHandling:
    """Test error handling."""

    def test_missing_api_key(self) -> None:
        """Test that ValueError is raised without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MiniMaxEmbeddings requires an API key"):
                MiniMaxEmbeddings()

    def test_missing_dependencies(self) -> None:
        """Test that ImportError is raised when httpx is not available."""
        with patch.object(MiniMaxEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match="httpx"):
                MiniMaxEmbeddings(api_key="test-key")

    def test_invalid_embedding_type(self) -> None:
        """Test that ValueError is raised with invalid embedding_type."""
        with pytest.raises(ValueError, match="embedding_type must be"):
            MiniMaxEmbeddings(api_key="test-key", embedding_type="invalid")


class TestMiniMaxEmbeddingsInitialization:
    """Test initialization with various parameters."""

    def test_initialization_with_api_key(self, embedding_model) -> None:
        """Test initialization with explicit API key."""
        assert embedding_model is not None
        assert embedding_model.model == "embo-01"
        assert embedding_model.api_key == "test-minimax-key"

    def test_initialization_with_env_var(self) -> None:
        """Test initialization using MINIMAX_API_KEY env var."""
        with patch("httpx.Client", return_value=MagicMock()):
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-test-key"}):
                model = MiniMaxEmbeddings()
                assert model.api_key == "env-test-key"

    def test_initialization_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("httpx.Client", return_value=MagicMock()):
            model = MiniMaxEmbeddings(
                api_key="test-key",
                embedding_type="query",
                batch_size=32,
                timeout=60.0,
            )
            assert model.embedding_type == "query"
            assert model._batch_size == 32
            assert model._timeout == 60.0

    def test_db_embedding_type(self, embedding_model) -> None:
        """Test default embedding_type is 'db'."""
        assert embedding_model.embedding_type == "db"


class TestMiniMaxEmbeddingsFunctionality:
    """Test core embedding functionality."""

    def test_embed_single_text(self, embedding_model, sample_text, mock_api_response) -> None:
        """Test embedding a single text."""
        embedding_model._client.post.return_value = mock_api_response
        result = embedding_model.embed(sample_text)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == 1536

    def test_embed_batch_texts(self, embedding_model, sample_texts) -> None:
        """Test embedding multiple texts in a batch."""
        embedding_model._client.post.return_value = _make_api_response(n=len(sample_texts))

        results = embedding_model.embed_batch(sample_texts)
        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        for emb in results:
            assert isinstance(emb, np.ndarray)
            assert emb.dtype == np.float32
            assert len(emb) == 1536

    def test_embed_batch_empty(self, embedding_model) -> None:
        """Test embed_batch with empty list."""
        results = embedding_model.embed_batch([])
        assert results == []

    def test_embed_batch_large_batch(self, embedding_model) -> None:
        """Test that large batches are split according to batch_size."""
        embedding_model._batch_size = 2
        texts = ["text1", "text2", "text3", "text4", "text5"]

        # 5 texts with batch_size=2 -> 3 API calls (2, 2, 1)
        embedding_model._client.post.side_effect = [
            _make_api_response(n=2),
            _make_api_response(n=2),
            _make_api_response(n=1),
        ]

        results = embedding_model.embed_batch(texts)
        assert len(results) == 5
        assert embedding_model._client.post.call_count == 3


class TestMiniMaxEmbeddingsProperties:
    """Test property methods."""

    def test_dimension_property(self, embedding_model) -> None:
        """Test dimension property returns 1536."""
        assert embedding_model.dimension == 1536
        assert isinstance(embedding_model.dimension, int)

    def test_get_tokenizer(self, embedding_model) -> None:
        """Test get_tokenizer returns a tokenizer."""
        tokenizer = embedding_model.get_tokenizer()
        assert tokenizer is not None

    def test_similarity(self, embedding_model) -> None:
        """Test cosine similarity calculation."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = embedding_model.similarity(v1, v2)
        assert isinstance(sim, np.float32)
        assert abs(sim - 1.0) < 1e-6

    def test_similarity_orthogonal(self, embedding_model) -> None:
        """Test cosine similarity for orthogonal vectors."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        sim = embedding_model.similarity(v1, v2)
        assert abs(sim) < 1e-6


class TestMiniMaxEmbeddingsUtilities:
    """Test utility methods."""

    def test_repr(self, embedding_model) -> None:
        """Test string representation."""
        repr_str = repr(embedding_model)
        assert "MiniMaxEmbeddings" in repr_str
        assert "embo-01" in repr_str

    def test_is_available_true(self) -> None:
        """Test _is_available returns True when httpx is installed."""
        with patch("chonkie.embeddings.minimax.importutil.find_spec") as mock_find_spec:
            mock_find_spec.return_value = Mock()
            assert MiniMaxEmbeddings._is_available()

    def test_is_available_false(self) -> None:
        """Test _is_available returns False when httpx is missing."""
        with patch("chonkie.embeddings.minimax.importutil.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None
            assert not MiniMaxEmbeddings._is_available()

    def test_callable_single_text(self, embedding_model, mock_api_response) -> None:
        """Test that the embeddings object is callable with a single string."""
        embedding_model._client.post.return_value = mock_api_response
        result = embedding_model("Hello world")
        assert isinstance(result, np.ndarray)

    def test_callable_batch(self, embedding_model) -> None:
        """Test that the embeddings object is callable with a list of strings."""
        embedding_model._client.post.return_value = _make_api_response(n=2)
        result = embedding_model(["Hello", "World"])
        assert isinstance(result, list)
        assert len(result) == 2


class TestMiniMaxEmbeddingsAPIError:
    """Test API error handling."""

    def test_api_error_response(self, embedding_model) -> None:
        """Test handling of API error status."""
        error_response = MagicMock()
        error_response.status_code = 200
        error_response.json.return_value = {
            "base_resp": {"status_code": 1001, "status_msg": "invalid api key"},
        }
        error_response.raise_for_status = MagicMock()
        embedding_model._client.post.return_value = error_response

        with pytest.raises(Exception):
            embedding_model.embed("test")

    def test_missing_vectors_field(self, embedding_model) -> None:
        """Test handling of missing vectors in response."""
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        bad_response.raise_for_status = MagicMock()
        embedding_model._client.post.return_value = bad_response

        with pytest.raises(Exception):
            embedding_model.embed("test")


class TestMiniMaxEmbeddingsRegistry:
    """Test that MiniMax embeddings are registered in the EmbeddingsRegistry."""

    def test_registry_provider(self) -> None:
        """Test that 'minimax' provider is registered."""
        from chonkie.embeddings.registry import EmbeddingsRegistry

        cls = EmbeddingsRegistry.get_provider("minimax")
        assert cls is MiniMaxEmbeddings

    def test_registry_model(self) -> None:
        """Test that 'embo-01' model is registered."""
        from chonkie.embeddings.registry import EmbeddingsRegistry

        cls = EmbeddingsRegistry.match("embo-01")
        assert cls is MiniMaxEmbeddings

    def test_registry_pattern(self) -> None:
        """Test that embo- pattern matches MiniMax."""
        from chonkie.embeddings.registry import EmbeddingsRegistry

        cls = EmbeddingsRegistry.match("embo-02")
        assert cls is MiniMaxEmbeddings

    def test_registry_provider_prefix(self) -> None:
        """Test minimax:// provider prefix matching."""
        from chonkie.embeddings.registry import EmbeddingsRegistry

        cls = EmbeddingsRegistry.match("minimax://embo-01")
        assert cls is MiniMaxEmbeddings


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "MINIMAX_API_KEY" not in os.environ,
    reason="Skipping integration test — requires MINIMAX_API_KEY",
)
@pytest.mark.xfail(
    reason="MiniMax embedding API has strict rate limits; xfail until stabilised",
    strict=False,
)
class TestMiniMaxEmbeddingsIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_real_embed_single(self) -> None:
        """Integration: embed a single text string."""
        embeddings = MiniMaxEmbeddings()
        result = embeddings.embed("Hello, world!")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 1536

    def test_real_embed_batch(self) -> None:
        """Integration: embed multiple texts."""
        embeddings = MiniMaxEmbeddings()
        results = embeddings.embed_batch(["Hello", "World", "MiniMax embeddings test"])
        assert len(results) == 3
        for emb in results:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == 1536

    def test_real_query_type(self) -> None:
        """Integration: embed with type='query'."""
        embeddings = MiniMaxEmbeddings(embedding_type="query")
        result = embeddings.embed("search query text")
        assert isinstance(result, np.ndarray)
        assert len(result) == 1536


if __name__ == "__main__":
    pytest.main()
