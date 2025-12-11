"""Test suite for CatsuEmbeddings."""

import os
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Try to import CatsuEmbeddings, skip all tests if not available
try:
    from chonkie.embeddings.catsu import CatsuEmbeddings, CatsuTokenizerWrapper
    CATSU_AVAILABLE = True
except ImportError:
    CATSU_AVAILABLE = False


@pytest.fixture
def mock_catsu_client():
    """Fixture to create a mock Catsu client."""
    mock_client = MagicMock()

    # Mock embed response
    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(2, 1024).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    # Mock list_models response
    mock_model_info = MagicMock()
    mock_model_info.name = "voyage-3"
    mock_model_info.dimensions = 1024
    mock_client.list_models.return_value = [mock_model_info]

    # Mock tokenize response
    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


@pytest.fixture
def embedding_model(mock_catsu_client):
    """Fixture to create a CatsuEmbeddings instance with mocked client."""
    if not CATSU_AVAILABLE:
        pytest.skip("Catsu not available")

    with patch('catsu.Client', return_value=mock_catsu_client):
        return CatsuEmbeddings(model="voyage-3", provider="voyageai")


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


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_catsu_not_available_error():
    """Test that proper error is raised when Catsu is not available."""
    with patch('chonkie.embeddings.catsu.importutil.find_spec', return_value=None):
        with pytest.raises(ImportError, match="catsu package is not available"):
            CatsuEmbeddings(model="voyage-3")


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_initialization_with_model_name(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings initializes with a model name."""
    assert embedding_model.model == "voyage-3"
    assert embedding_model.provider == "voyageai"
    assert embedding_model.client is not None


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_initialization_with_api_keys(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings initializes with API keys."""
    api_keys = {"voyageai": "test-key-123"}

    with patch('catsu.Client', return_value=mock_catsu_client) as mock_client_class:
        CatsuEmbeddings(
            model="voyage-3",
            provider="voyageai",
            api_keys=api_keys
        )

        # Verify Client was initialized with correct api_keys
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs['api_keys'] == api_keys


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_initialization_with_config_params(mock_catsu_client) -> None:
    """Test that CatsuEmbeddings initializes with custom config parameters."""
    with patch('catsu.Client', return_value=mock_catsu_client) as mock_client_class:
        embeddings = CatsuEmbeddings(
            model="voyage-3",
            max_retries=5,
            timeout=60,
            verbose=True,
            batch_size=64
        )

        # Verify Client was initialized with correct parameters
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs['max_retries'] == 5
        assert call_kwargs['timeout'] == 60
        assert call_kwargs['verbose'] is True
        assert embeddings._batch_size == 64


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_embed_single_text(embedding_model: CatsuEmbeddings, sample_text: str) -> None:
    """Test that CatsuEmbeddings correctly embeds a single text."""
    embedding = embedding_model.embed(sample_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Should be 1D array
    assert len(embedding) == 1024  # voyage-3 dimension


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_embed_batch_texts(embedding_model: CatsuEmbeddings, sample_texts: List[str]) -> None:
    """Test that CatsuEmbeddings correctly embeds a batch of texts."""
    # Mock the client to return correct number of embeddings
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 1024).astype(np.float32)
    embedding_model.client.embed.return_value = mock_response

    embeddings = embedding_model.embed_batch(sample_texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(embedding.ndim == 1 for embedding in embeddings)
    assert all(len(embedding) == 1024 for embedding in embeddings)


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_embed_batch_empty_list(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings handles empty batch correctly."""
    embeddings = embedding_model.embed_batch([])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_embed_batch_with_batching(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings correctly handles batching."""
    # Set small batch size
    embedding_model._batch_size = 2

    # Create 5 texts to force multiple batches
    texts = [f"Text {i}" for i in range(5)]

    # Mock responses for each batch
    def mock_embed_side_effect(*args, **kwargs):
        input_texts = kwargs.get('input', args[1] if len(args) > 1 else [])
        batch_size = len(input_texts) if isinstance(input_texts, list) else 1
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(batch_size, 1024).astype(np.float32)
        return mock_response

    embedding_model.client.embed.side_effect = mock_embed_side_effect

    embeddings = embedding_model.embed_batch(texts)

    assert len(embeddings) == 5
    # Should have made 3 calls: batch of 2, batch of 2, batch of 1
    assert embedding_model.client.embed.call_count >= 3


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_embed_batch_fallback_on_error(embedding_model: CatsuEmbeddings, sample_texts: List[str]) -> None:
    """Test that CatsuEmbeddings falls back to individual embeds on batch failure."""
    # Mock batch embed to fail, individual embeds to succeed
    call_count = [0]

    def mock_embed_side_effect(*args, **kwargs):
        call_count[0] += 1
        input_texts = kwargs.get('input', args[1] if len(args) > 1 else [])

        # First call (batch) fails
        if call_count[0] == 1:
            raise Exception("Batch embedding failed")

        # Subsequent calls (individual) succeed
        batch_size = len(input_texts) if isinstance(input_texts, list) else 1
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(batch_size, 1024).astype(np.float32)
        return mock_response

    embedding_model.client.embed.side_effect = mock_embed_side_effect

    with pytest.warns(UserWarning, match="Batch embedding failed"):
        embeddings = embedding_model.embed_batch(sample_texts)

    assert len(embeddings) == len(sample_texts)


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_similarity(embedding_model: CatsuEmbeddings, sample_texts: List[str]) -> None:
    """Test that CatsuEmbeddings correctly calculates similarity between two embeddings."""
    # Create two embeddings
    emb1 = np.random.rand(1024).astype(np.float32)
    emb2 = np.random.rand(1024).astype(np.float32)

    similarity_score = embedding_model.similarity(emb1, emb2)

    assert isinstance(similarity_score, np.float32)
    assert -1.0 <= similarity_score <= 1.0  # Cosine similarity range


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_dimension_property(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings correctly returns the dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 1024


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_dimension_property_fallback(mock_catsu_client) -> None:
    """Test that dimension property falls back to test embedding if not in catalog."""
    # Mock list_models to return empty list
    mock_catsu_client.list_models.return_value = []

    # Mock embed to return a test embedding
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(1, 512).astype(np.float32)
    mock_catsu_client.embed.return_value = mock_response

    with patch('catsu.Client', return_value=mock_catsu_client):
        embeddings = CatsuEmbeddings(model="unknown-model")

        # Should infer dimension from test embedding
        assert embeddings.dimension == 512


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_get_tokenizer(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings returns a tokenizer wrapper."""
    tokenizer = embedding_model.get_tokenizer()

    assert isinstance(tokenizer, CatsuTokenizerWrapper)
    assert tokenizer.model == "voyage-3"
    assert tokenizer.provider == "voyageai"


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_tokenizer_count(embedding_model: CatsuEmbeddings, sample_text: str) -> None:
    """Test that tokenizer wrapper correctly counts tokens."""
    tokenizer = embedding_model.get_tokenizer()
    token_count = tokenizer.count(sample_text)

    assert isinstance(token_count, int)
    assert token_count == 10  # From mock


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_tokenizer_encode_warning(embedding_model: CatsuEmbeddings) -> None:
    """Test that tokenizer encode method warns about unsupported functionality."""
    tokenizer = embedding_model.get_tokenizer()

    with pytest.warns(UserWarning, match="Token encoding not supported"):
        tokens = tokenizer.encode("test text")

    assert tokens == []


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_repr(embedding_model: CatsuEmbeddings) -> None:
    """Test that CatsuEmbeddings has a proper string representation."""
    repr_str = repr(embedding_model)
    assert "CatsuEmbeddings" in repr_str
    assert "voyage-3" in repr_str
    assert "voyageai" in repr_str


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_repr_without_provider(mock_catsu_client) -> None:
    """Test repr without explicit provider."""
    with patch('catsu.Client', return_value=mock_catsu_client):
        embeddings = CatsuEmbeddings(model="voyage-3")
        repr_str = repr(embeddings)
        assert "CatsuEmbeddings" in repr_str
        assert "voyage-3" in repr_str


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_call_method_single_text(embedding_model: CatsuEmbeddings, sample_text: str) -> None:
    """Test that CatsuEmbeddings can be called directly with a single text."""
    embedding = embedding_model(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1


@pytest.mark.skipif(
    not CATSU_AVAILABLE,
    reason="Skipping test because Catsu is not installed",
)
def test_call_method_batch(embedding_model: CatsuEmbeddings, sample_texts: List[str]) -> None:
    """Test that CatsuEmbeddings can be called directly with a list of texts."""
    # Mock the client to return correct number of embeddings
    mock_response = MagicMock()
    mock_response.to_numpy.return_value = np.random.rand(len(sample_texts), 1024).astype(np.float32)
    embedding_model.client.embed.return_value = mock_response

    embeddings = embedding_model(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)


# Integration test (requires actual API key)
@pytest.mark.skipif(
    not CATSU_AVAILABLE or "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping integration test - requires Catsu and VOYAGE_API_KEY",
)
def test_real_embed_integration():
    """Integration test with real Catsu client (requires API key)."""
    embeddings = CatsuEmbeddings(model="voyage-3", provider="voyageai")

    text = "This is a real integration test."
    embedding = embeddings.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0

    # Test dimension property
    assert embeddings.dimension == len(embedding)
