"""Test Cohere embeddings."""

import os
from importlib.util import find_spec
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.cohere import CohereEmbeddings


@pytest.fixture(autouse=True)
def mock_cohere_dependencies():
    """Mock Cohere dependencies to avoid real API calls and downloads."""
    # Mock tokenizer requests and creation
    mock_tokenizer_response = MagicMock()
    mock_tokenizer_response.text = '{"vocab": {}, "model": {"type": "BPE"}}'
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer_instance.encode_batch.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Mock Cohere ClientV2
    mock_client_instance = MagicMock()
    
    def mock_embed_response(model=None, input_type=None, embedding_types=None, texts=None, **kwargs):
        mock_response = MagicMock()
        # Return different embeddings for each text in the batch to avoid similarity issues
        num_texts = len(texts) if texts else 1
        embeddings = []
        for i in range(num_texts):
            # Create slightly different embeddings for each text
            base_embedding = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01] * 96
            embeddings.append(base_embedding)
        mock_response.embeddings.float_ = embeddings
        return mock_response
    
    mock_client_instance.embed.side_effect = mock_embed_response
    
    with patch('requests.get', return_value=mock_tokenizer_response), \
         patch('tokenizers.Tokenizer.from_str', return_value=mock_tokenizer_instance), \
         patch('cohere.ClientV2', return_value=mock_client_instance):
        yield mock_client_instance


@pytest.fixture
def embedding_model() -> CohereEmbeddings:
    """Fixture to create a CohereEmbeddings instance."""
    api_key = os.environ.get("COHERE_API_KEY", "test_key")  # Use test key if no real key
    return CohereEmbeddings(model="embed-english-light-v3.0", api_key=api_key)


@pytest.fixture
def sample_text() -> str:
    """Fixture to create a sample text."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture to create a list of sample texts."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization_with_model_name() -> None:
    """Test initialization with model name."""
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0", api_key="test_key")
    assert embeddings.model == "embed-english-light-v3.0"
    assert embeddings.client is not None


def test_embed_single_text(embedding_model: CohereEmbeddings, sample_text: str) -> None:
    """Test embedding a single text."""
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


def test_embed_batch_texts(
    embedding_model: CohereEmbeddings, sample_texts: List[str]
) -> None:
    """Test embedding a batch of texts."""
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (embedding_model.dimension,) for embedding in embeddings
    )


def test_count_tokens_single_text(
    embedding_model: CohereEmbeddings, sample_text: str
) -> None:
    """Test counting tokens for a single text."""
    token_count = embedding_model.count_tokens(sample_text)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_count_tokens_batch_texts(
    embedding_model: CohereEmbeddings, sample_texts: List[str]
) -> None:
    """Test counting tokens for a batch of texts."""
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


def test_similarity(embedding_model: CohereEmbeddings, sample_texts: List[str]) -> None:
    """Test similarity between two embeddings."""
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32)
    assert -1.0 <= similarity_score <= 1.0


def test_dimension_property(embedding_model: CohereEmbeddings) -> None:
    """Test dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_is_available() -> None:
    """Test _is_available method."""
    if find_spec("cohere") is not None:
        assert CohereEmbeddings._is_available() is True
    else:
        assert CohereEmbeddings._is_available() is False


def test_repr(embedding_model: CohereEmbeddings) -> None:
    """Test repr method."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("CohereEmbeddings")


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping real API test because COHERE_API_KEY is not defined",
)
@pytest.mark.skipif(True, reason="Disabled for CI - mocked tests cover functionality")
def test_real_api_integration() -> None:
    """Test with real Cohere API if COHERE_API_KEY is available."""
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    text = "This is a test sentence for the real API."
    embedding = embeddings.embed(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embeddings.dimension,)
    assert not np.allclose(embedding, 0)  # Should not be all zeros


if __name__ == "__main__":
    pytest.main()
