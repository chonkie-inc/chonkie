"""Test the SentenceTransformerEmbeddings class."""

from typing import List

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings


@pytest.fixture
def embedding_model() -> SentenceTransformerEmbeddings:
    """Return a SentenceTransformerEmbeddings instance."""
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def sample_text() -> str:
    """Return a sample text for testing."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> List[str]:
    """Return a list of sample texts for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization_with_model_name(
    embedding_model: SentenceTransformerEmbeddings,
) -> None:
    """Test the initialization with a model name."""
    assert embedding_model.model_name_or_path == "all-MiniLM-L6-v2"
    assert embedding_model.model is not None


def test_initialization_with_model_instance(
    embedding_model: SentenceTransformerEmbeddings,
) -> None:
    """Test the initialization with a model instance."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model)
    assert embeddings.model_name_or_path == model.model_card_data.base_model
    assert embeddings.model is model


def test_embed_single_text(
    embedding_model: SentenceTransformerEmbeddings, sample_text: str
) -> None:
    """Test the embed method with a single text."""
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


def test_embed_batch_texts(
    embedding_model: SentenceTransformerEmbeddings, sample_texts: List[str]
) -> None:
    """Test the embed_batch method with a list of texts."""
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (embedding_model.dimension,) for embedding in embeddings
    )


def test_count_tokens_single_text(
    embedding_model: SentenceTransformerEmbeddings, sample_text: str
) -> None:
    """Test the count_tokens method with a single text."""
    token_count = embedding_model.count_tokens(sample_text)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_count_tokens_batch_texts(
    embedding_model: SentenceTransformerEmbeddings, sample_texts: List[str]
) -> None:
    """Test the count_tokens_batch method with a list of texts."""
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


def test_similarity(
    embedding_model: SentenceTransformerEmbeddings, sample_texts: List[str]
) -> None:
    """Test the similarity method."""
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, float)
    assert 0.0 <= similarity_score <= 1.0


def test_dimension_property(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test the dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_is_available() -> None:
    """Test the _is_available method."""
    assert SentenceTransformerEmbeddings._is_available() is True


def test_repr(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test the repr method."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("SentenceTransformerEmbeddings")


if __name__ == "__main__":
    pytest.main()
