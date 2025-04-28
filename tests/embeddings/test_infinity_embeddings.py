"""Test suite for InfinityEmbeddings."""

import os
from typing import List
import aiohttp
import pytest
import numpy as np
from chonkie.embeddings.infinity import InfinityEmbeddings

@pytest.fixture
def embedding_model() -> InfinityEmbeddings:
    """Fixture to create an InfinityEmbeddings instance."""
    return InfinityEmbeddings(model="michaelfeil/bge-small-en-v1.5")

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

# Initialization Tests
def test_initialization_with_model_name(embedding_model: InfinityEmbeddings) -> None:
    """Test that InfinityEmbeddings initializes with a model name."""
    assert embedding_model.model == "michaelfeil/bge-small-en-v1.5"
    assert embedding_model.infinity_api_url == "https://infinity.modal.michaelfeil.eu"
    assert embedding_model.dimension == 384
    assert embedding_model._batch_size == 32
    assert embedding_model.timeout == 60.0

def test_initialization_with_invalid_model() -> None:
    """Test that InfinityEmbeddings does not initializes with a invalid model name."""
    with pytest.raises(ValueError, match="Model 'invalid_model' not supported"):
        InfinityEmbeddings(model="invalid_model")

# Synchronous Embedding Tests
@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_embed_single_text(embedding_model: InfinityEmbeddings, sample_text: str) -> None:
    """Test that InfinityEmbeddings correctly embeds a single text."""
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)
    assert embedding.dtype == np.float32

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_embed_batch_texts(embedding_model: InfinityEmbeddings, sample_texts: list[str]) -> None:
    """Test that InfinityEmbeddings correctly embeds a batch of texts."""
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (embedding_model.dimension,) for emb in embeddings)

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_embed_empty_list(embedding_model: InfinityEmbeddings) -> None:
    """"Test that InfinityEmbeddings correctly handles an empty list."""
    embeddings = embedding_model.embed_batch([])
    assert embeddings == []

# Asynchronous Embedding Tests
@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
@pytest.mark.anyio
async def test_aembed_single_text(embedding_model: InfinityEmbeddings, sample_text: str) -> None:
    """"Test that InfinityEmbeddings correctly embeds a single text asynchronously."""
    embedding = await embedding_model.aembed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)
    assert embedding.dtype == np.float32

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
@pytest.mark.anyio
async def test_aembed_batch_texts(embedding_model: InfinityEmbeddings, sample_texts: list[str]) -> None:
    """Test that InfinityEmbeddings correctly embeds a batch of texts asynchronously."""
    embeddings = await embedding_model.aembed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (embedding_model.dimension,) for emb in embeddings)

# Internal Method Tests
@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_prepare_batches(embedding_model: InfinityEmbeddings, sample_texts: list[str]) -> None:
    """Test that InfinityEmbeddings correctly prepares batches."""
    batches, _ = embedding_model._prepare_batches(sample_texts)
    assert len(batches) == 1
    assert len(batches[0]) == len(sample_texts)

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_prepare_batches_large(embedding_model: InfinityEmbeddings) -> None:
    """Test that InfinityEmbeddings correctly prepares batches for large input."""
    large_texts = ["text"] * 40
    batches, _ = embedding_model._prepare_batches(large_texts)
    assert len(batches) == 2
    assert len(batches[0]) == 32
    assert len(batches[1]) == 8

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_sync_request(embedding_model: InfinityEmbeddings, sample_texts: list[str]) -> None:
    """Test that InfinityEmbeddings correctly handles synchronous requests."""
    response = embedding_model._sync_request(sample_texts)
    assert isinstance(response, list)
    assert len(response) == len(sample_texts)
    assert all(isinstance(emb, np.ndarray) for emb in response)
    assert all(emb.shape == (embedding_model.dimension,) for emb in response)

@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
@pytest.mark.anyio
async def test_async_request(embedding_model: InfinityEmbeddings, sample_texts: list[str]) -> None:
    """"Test that InfinityEmbeddings correctly handles asynchronous requests."""
    async with aiohttp.ClientSession() as session:
        response = await embedding_model._async_request(session, sample_texts)
        assert isinstance(response, list)
        assert len(response) == len(sample_texts)
        assert all(isinstance(emb, np.ndarray) for emb in response)
        assert all(emb.shape == (embedding_model.dimension,) for emb in response)

# Utility Tests
@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_dimension_property(embedding_model: InfinityEmbeddings) -> None:
    """Test that InfinityEmbeddings correctly returns the dimension property."""
    assert embedding_model.dimension > 0
    assert isinstance(embedding_model.dimension, int)


@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_is_available(embedding_model: InfinityEmbeddings) -> None:
    """Test that InfinityEmbeddings correctly checks availability."""
    assert embedding_model.is_available() is True


@pytest.mark.skipif("INFINITY_API_URL" not in os.environ,
     reason="Skipping test because INFINITY_API_URL is not defined",
)
def test_repr(embedding_model: InfinityEmbeddings) -> None:
    """Test that InfinityEmbeddings correctly returns a string representation."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("InfinityEmbeddings")

if __name__ == "__main__":
    pytest.main()