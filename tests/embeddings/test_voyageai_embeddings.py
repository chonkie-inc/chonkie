"""Test suite for VoyageAIEmbeddings."""

import os
from typing import List

import numpy as np
import pytest

from chonkie import VoyageAIEmbeddings


@pytest.fixture
def dummy_embedding_model() -> VoyageAIEmbeddings:
    """Fixture for a VoyageAIEmbeddings instance with a dummy API key for non-API tests."""
    return VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key")

@pytest.fixture
def api_embedding_model() -> VoyageAIEmbeddings:
    """Fixture for a VoyageAIEmbeddings instance with the actual API key for API tests."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        pytest.skip("VOYAGE_API_KEY not set")
    return VoyageAIEmbeddings(model="voyage-3", api_key=api_key)

@pytest.fixture
def sample_short_text() -> str:
    """Fixture for a short sample text."""
    return "This is a short text."

@pytest.fixture
def sample_long_text() -> str:
    """Fixture for a long sample text likely exceeding token limits."""
    return "a " * 100000  # Approximately 100,000 tokens

@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture for a list of sample texts for batch testing."""
    return [
        "First sample text.",
        "Another example sentence.",
        "Multiple sentences for testing.",
    ]


def test_initialization_valid_model(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings initializes with a valid model name."""
    assert dummy_embedding_model.model == "voyage-3"
    assert dummy_embedding_model._dimension == 1024
    assert dummy_embedding_model._token_limit == 32000
    assert dummy_embedding_model.batch_size == 128
    assert dummy_embedding_model.truncation is True

def test_initialization_invalid_model(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings does not initializes and raises ValueError."""
    with pytest.raises(ValueError, match="Model 'invalid_model' not available"):
        VoyageAIEmbeddings(model="invalid_model", api_key="dummy_key")

# def test_initialization_no_api_key(dummy_embedding_model: VoyageAIEmbeddings) -> None:
#     """Test that VoyageAIEmbeddings raises ValueError if no API key is provided."""

#     with pytest.raises(voyageai.error.AuthenticationError) as excinfo:
#         VoyageAIEmbeddings(model="voyage-3")
#     msg = str(excinfo.value)
#     assert "No API key provided" in msg

    
def test_initialization_invalid_output_dimension(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings raises ValueError for invalid output_dimension."""
    with pytest.raises(ValueError, match="Invalid output_dimension"):
        VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key", output_dimension=300)

def test_initialization_batch_size_cap(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that batch_size is capped at 128."""
    embedding = VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key", batch_size=200)
    assert embedding.batch_size == 128


def test_count_tokens(dummy_embedding_model: VoyageAIEmbeddings, sample_short_text: str) -> None:
    """Test token counting for a short text."""
    tokens = dummy_embedding_model.count_tokens(sample_short_text)
    assert isinstance(tokens, int)
    assert tokens > 0

def test_count_tokens_batch(dummy_embedding_model: VoyageAIEmbeddings, sample_texts: List[str]) -> None:
    """Test token counting for a batch of texts."""
    token_counts = dummy_embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    for count in token_counts:
        assert isinstance(count, int)
        assert count > 0

# Synchronous Embedding Tests
@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
def test_embed_short_text(api_embedding_model: VoyageAIEmbeddings, sample_short_text: str) -> None:
    """Test synchronous embedding of a short text."""
    embedding = api_embedding_model.embed(sample_short_text)
    assert isinstance(embedding, np.ndarray)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    assert embedding.shape == (expected_dim,)
    assert embedding.dtype == np.float32

@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
def test_embed_batch(api_embedding_model: VoyageAIEmbeddings, sample_texts: List[str]) -> None:
    """Test synchronous batch embedding."""
    embeddings = api_embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (expected_dim,)
        assert emb.dtype == np.float32

@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
def test_embed_with_output_dimension(api_embedding_model: VoyageAIEmbeddings) -> None:
    """Test embedding with a custom output_dimension."""
    embedding_model = VoyageAIEmbeddings(model="voyage-3", output_dimension=1024)
    embedding = embedding_model.embed("text")
    assert embedding.shape == (1024,)

# Asynchronous Embedding Tests
@pytest.mark.asyncio
@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
async def test_aembed_short_text(api_embedding_model: VoyageAIEmbeddings, sample_short_text: str) -> None:
    """Test asynchronous embedding of a short text."""
    embedding = await api_embedding_model.aembed(sample_short_text)
    assert isinstance(embedding, np.ndarray)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    assert embedding.shape == (expected_dim,)
    assert embedding.dtype == np.float32

@pytest.mark.asyncio
@pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGE_API_KEY is not defined",
)
async def test_aembed_batch(api_embedding_model: VoyageAIEmbeddings, sample_texts: List[str]) -> None:
    """Test asynchronous batch embedding."""
    embeddings = await api_embedding_model.aembed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (expected_dim,)
        assert emb.dtype == np.float32

# Utility Tests
def test_dimension_property(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings correctly calculates the dimension property."""
    assert dummy_embedding_model.dimension == 1024

def test_is_available(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings correctly checks if it is available."""
    assert dummy_embedding_model.is_available() is True

def test_repr(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test that VoyageAIEmbeddings correctly returns a string representation."""
    assert repr(dummy_embedding_model) == "VoyageAIEmbeddings(model=voyage-3, dimension=1024)"

def test_get_tokenizer_or_token_counter(dummy_embedding_model: VoyageAIEmbeddings) -> None:
    """Test retrieval of the tokenizer."""
    tokenizer = dummy_embedding_model.get_tokenizer_or_token_counter()
    assert hasattr(tokenizer, "encode")  # Basic check for tokenizer functionality

if __name__ == "__main__":
    pytest.main()