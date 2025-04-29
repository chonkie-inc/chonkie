"""Test Ollama embeddings."""

import os

import numpy as np
import pytest

from chonkie.embeddings.ollama import OllamaEmbeddings


@pytest.fixture
def embedding_model():
    """Fixture to create a OllamaEmbeddings instance."""
    return OllamaEmbeddings(model="all-minilm", endpoint=os.getenv("OLLAMA_ENDPOINT"))


@pytest.fixture
def sample_text():
    """Fixture to create a sample text."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts():
    """Fixture to create a list of sample texts."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_initialization_with_model_name():
    """Test initialization with model name."""
    embeddings = OllamaEmbeddings("all-minilm")
    assert embeddings.model == "all-minilm"
    assert embeddings.model is not None


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_embed_single_text(embedding_model, sample_text):
    """Test embedding a single text."""
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_embed_batch_texts(embedding_model, sample_texts):
    """Test embedding a batch of texts."""
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (embedding_model.dimension,) for embedding in embeddings
    )


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_count_tokens_single_text(embedding_model, sample_text):
    """Test counting tokens for a single text."""
    token_count = embedding_model.count_tokens(sample_text)
    assert isinstance(token_count, float)
    assert token_count > 0


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_count_tokens_batch_texts(embedding_model, sample_texts):
    """Test counting tokens for a batch of texts."""
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, float) for count in token_counts)
    assert all(count > 0 for count in token_counts)


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_similarity(embedding_model, sample_texts):
    """Test similarity between two embeddings."""
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, float)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_dimension_property(embedding_model):
    """Test dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_is_available():
    """Test is_available method."""
    assert OllamaEmbeddings.is_available() is True


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_repr(embedding_model):
    """Test repr method."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("OllamaEmbeddings")


if __name__ == "__main__":
    pytest.main()
