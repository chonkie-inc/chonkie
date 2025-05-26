"""Test suite for GeminiEmbeddings."""

import os
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.gemini import GeminiEmbeddings


@pytest.fixture
def embedding_model() -> GeminiEmbeddings:
    """Fixture to create a GeminiEmbeddings instance."""
    api_key = os.environ.get("GEMINI_API_KEY")
    return GeminiEmbeddings(api_key=api_key)  # Use default model


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


def test_initialization_without_api_key() -> None:
    """Test that GeminiEmbeddings raises ValueError when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Gemini API key not found"):
            GeminiEmbeddings()


def test_initialization_with_api_key() -> None:
    """Test that GeminiEmbeddings initializes correctly with API key."""
    with patch("google.genai.Client") as mock_client:
        embeddings = GeminiEmbeddings(api_key="test-key")
        assert embeddings.model == "gemini-embedding-exp-03-07"  # Updated default model
        assert embeddings.task_type == "SEMANTIC_SIMILARITY"
        assert embeddings._api_key == "test-key"
        mock_client.assert_called_once_with(api_key="test-key")


def test_initialization_with_env_var() -> None:
    """Test that GeminiEmbeddings uses environment variable for API key."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
        with patch("google.genai.Client") as mock_client:
            embeddings = GeminiEmbeddings()
            assert embeddings._api_key == "env-key"
            mock_client.assert_called_once_with(api_key="env-key")


def test_initialization_with_custom_model() -> None:
    """Test that GeminiEmbeddings initializes with custom model."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(model="embedding-001", api_key="test-key")
        assert embeddings.model == "embedding-001"
        assert embeddings.dimension == 768
        
        # Test experimental model with different dimensions
        embeddings_exp = GeminiEmbeddings(model="gemini-embedding-exp-03-07", api_key="test-key")
        assert embeddings_exp.model == "gemini-embedding-exp-03-07"
        assert embeddings_exp.dimension == 3072


def test_dimension_property() -> None:
    """Test that the dimension property returns correct value."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(api_key="test-key")
        assert embeddings.dimension == 3072  # Updated for new default model


def test_count_tokens() -> None:
    """Test token counting functionality."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.count_tokens.return_value = 10
        
        embeddings = GeminiEmbeddings(api_key="test-key")
        
        # Test API-based token counting
        text = "This is a test sentence"
        token_count = embeddings.count_tokens(text)
        assert token_count == 10
        mock_client.models.count_tokens.assert_called_once_with(model="gemini-embedding-exp-03-07", contents=text)


def test_count_tokens_batch() -> None:
    """Test batch token counting functionality."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(api_key="test-key")
        
        texts = ["Hello world", "This is a test"]
        token_counts = embeddings.count_tokens_batch(texts)
        
        assert len(token_counts) == 2
        assert all(isinstance(count, int) for count in token_counts)


def test_similarity() -> None:
    """Test similarity computation."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(api_key="test-key")
        
        # Create sample embeddings
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        embedding3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Test orthogonal vectors (similarity should be 0)
        sim1 = embeddings.similarity(embedding1, embedding2)
        assert abs(sim1) < 1e-6
        
        # Test identical vectors (similarity should be 1)
        sim2 = embeddings.similarity(embedding1, embedding3)
        assert abs(sim2 - 1.0) < 1e-6


def test_get_tokenizer_or_token_counter() -> None:
    """Test that get_tokenizer_or_token_counter returns the count_tokens method."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(api_key="test-key")
        
        tokenizer = embeddings.get_tokenizer_or_token_counter()
        assert callable(tokenizer)
        assert tokenizer == embeddings.count_tokens


def test_repr() -> None:
    """Test string representation."""
    with patch("google.genai.Client"):
        embeddings = GeminiEmbeddings(api_key="test-key")
        repr_str = repr(embeddings)
        assert "GeminiEmbeddings" in repr_str
        assert "gemini-embedding-exp-03-07" in repr_str
        assert "SEMANTIC_SIMILARITY" in repr_str


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Skipping test because GEMINI_API_KEY is not defined",
)
def test_initialization_with_real_api_key(embedding_model: GeminiEmbeddings) -> None:
    """Test that GeminiEmbeddings initializes with a real API key."""
    assert embedding_model.model == "gemini-embedding-exp-03-07"  # Updated default model
    assert embedding_model.client is not None


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Skipping test because GEMINI_API_KEY is not defined",
)
def test_embed_single_text_real(embedding_model: GeminiEmbeddings, sample_text: str) -> None:
    """Test that GeminiEmbeddings correctly embeds a single text with real API."""
    embedding = embedding_model.embed(sample_text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (3072,)  # Expected dimension for gemini-embedding-exp-03-07
    assert not np.allclose(embedding, 0)  # Embedding should not be all zeros


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Skipping test because GEMINI_API_KEY is not defined",
)
def test_embed_batch_real(embedding_model: GeminiEmbeddings, sample_texts: List[str]) -> None:
    """Test that GeminiEmbeddings correctly embeds multiple texts with real API."""
    embeddings = embedding_model.embed_batch(sample_texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    
    for embedding in embeddings:
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (3072,)
        assert not np.allclose(embedding, 0)


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ,
    reason="Skipping test because GEMINI_API_KEY is not defined",
)
def test_call_method_real(embedding_model: GeminiEmbeddings, sample_text: str, sample_texts: List[str]) -> None:
    """Test that GeminiEmbeddings correctly handles __call__ method with real API."""
    # Test single text
    single_embedding = embedding_model(sample_text)
    assert isinstance(single_embedding, np.ndarray)
    assert single_embedding.shape == (3072,)
    
    # Test multiple texts
    multiple_embeddings = embedding_model(sample_texts)
    assert isinstance(multiple_embeddings, list)
    assert len(multiple_embeddings) == len(sample_texts)


def test_embed_mock() -> None:
    """Test embed method with mocked API response."""
    mock_embedding_values = [0.1] * 3072  # Mock 3072-dimensional embedding (for exp model)
    
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the API response
        mock_result = MagicMock()
        mock_result.embeddings = [MagicMock()]
        mock_result.embeddings[0].values = mock_embedding_values
        mock_client.models.embed_content.return_value = mock_result
        
        embeddings = GeminiEmbeddings(api_key="test-key")
        result = embeddings.embed("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3072,)
        assert np.allclose(result, mock_embedding_values)
        
        # Verify the API was called correctly
        mock_client.models.embed_content.assert_called_once()
        call_args = mock_client.models.embed_content.call_args
        assert call_args[1]["model"] == "gemini-embedding-exp-03-07"
        assert call_args[1]["contents"] == "test text"


def test_embed_batch_mock() -> None:
    """Test embed_batch method with mocked API response."""
    mock_embedding_values = [0.1] * 3072  # Updated for new default model
    
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the API response
        mock_result = MagicMock()
        mock_result.embeddings = [MagicMock()]
        mock_result.embeddings[0].values = mock_embedding_values
        mock_client.models.embed_content.return_value = mock_result
        
        embeddings = GeminiEmbeddings(api_key="test-key")
        texts = ["text1", "text2"]
        results = embeddings.embed_batch(texts)
        
        assert len(results) == 2
        assert all(isinstance(result, np.ndarray) for result in results)
        assert all(result.shape == (3072,) for result in results)
        
        # Verify the API was called for each text
        assert mock_client.models.embed_content.call_count == 2


def test_embed_retry_logic() -> None:
    """Test retry logic when API calls fail."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # First two calls fail, third succeeds
        mock_client.models.embed_content.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            MagicMock(embeddings=[MagicMock(values=[0.1] * 768)])
        ]
        
        embeddings = GeminiEmbeddings(api_key="test-key", max_retries=3)
        result = embeddings.embed("test text")
        
        assert isinstance(result, np.ndarray)
        assert mock_client.models.embed_content.call_count == 3


def test_embed_failure_after_retries() -> None:
    """Test that embed raises exception after all retries fail."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # All calls fail
        mock_client.models.embed_content.side_effect = Exception("API Error")
        
        embeddings = GeminiEmbeddings(api_key="test-key", max_retries=2)
        
        with pytest.raises(RuntimeError, match="Failed to get embeddings after 2 attempts"):
            embeddings.embed("test text")


def test_is_available() -> None:
    """Test the _is_available method."""
    with patch("importlib.util.find_spec") as mock_find_spec:
        # Test when dependencies are available
        mock_find_spec.return_value = MagicMock()  # Mock spec found
        
        embeddings = GeminiEmbeddings.__new__(GeminiEmbeddings)  # Create without calling __init__
        assert embeddings._is_available() is True
        
        # Test when dependencies are not available
        mock_find_spec.return_value = None
        assert embeddings._is_available() is False


def test_import_dependencies_failure() -> None:
    """Test that _import_dependencies raises ImportError when dependencies not available."""
    with patch.object(GeminiEmbeddings, "_is_available", return_value=False):
        with pytest.raises(ImportError, match="google-genai, numpy"):
            GeminiEmbeddings(api_key="test-key")


def test_unknown_model_warning() -> None:
    """Test that unknown model warning uses dynamic default model values."""
    import warnings
    
    with patch("google.genai.Client"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create embeddings with unknown model
            embeddings = GeminiEmbeddings(model="unknown-future-model", api_key="test-key")
            
            # Check that warning was issued
            assert len(w) == 1
            warning_message = str(w[0].message)
            
            # Verify warning contains the unknown model name
            assert "unknown-future-model" in warning_message
            
            # Verify warning contains the default model name and its specs
            default_dimension, default_max_tokens = GeminiEmbeddings.AVAILABLE_MODELS[GeminiEmbeddings.DEFAULT_MODEL]
            assert GeminiEmbeddings.DEFAULT_MODEL in warning_message
            assert str(default_dimension) in warning_message
            assert str(default_max_tokens) in warning_message
            
            # Verify the embeddings instance uses the default model's specs
            assert embeddings.dimension == default_dimension
            assert embeddings._max_tokens == default_max_tokens