"""Test suite for AzureOpenAIEmbeddings."""

import os
from typing import List

import numpy as np
import pytest

from chonkie.embeddings.azure_openai import AzureOpenAIEmbeddings


@pytest.fixture
def azure_embedding_model() -> AzureOpenAIEmbeddings:
    """Fixture to create an AzureOpenAIEmbeddings instance."""
    return AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        deployment="text-embedding-3-small",
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )


@pytest.fixture
def sample_text() -> str:
    """Fixture to create a sample text for testing."""
    return "This is a sample text for Azure testing."


@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture to create a list of sample texts for testing."""
    return [
        "This is the first Azure sample.",
        "Second test input.",
        "Yet another sentence to embed.",
    ]


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_initialization_with_model_name(
    azure_embedding_model: AzureOpenAIEmbeddings,
) -> None:
    """Test that AzureOpenAIEmbeddings initializes with a model name."""
    assert azure_embedding_model.model == "text-embedding-3-small"
    assert azure_embedding_model.client is not None


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_embed_single_text(
    azure_embedding_model: AzureOpenAIEmbeddings, sample_text: str
) -> None:
    """Test that AzureOpenAIEmbeddings correctly embeds a single text."""
    embedding = azure_embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (azure_embedding_model.dimension,)


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_embed_batch_texts(
    azure_embedding_model: AzureOpenAIEmbeddings, sample_texts: List[str]
) -> None:
    """Test that AzureOpenAIEmbeddings correctly embeds a batch of texts."""
    embeddings = azure_embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(
        embedding.shape == (azure_embedding_model.dimension,)
        for embedding in embeddings
    )


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_similarity(
    azure_embedding_model: AzureOpenAIEmbeddings, sample_texts: List[str]
) -> None:
    """Test that AzureOpenAIEmbeddings calculates similarity between embeddings."""
    embeddings = azure_embedding_model.embed_batch(sample_texts)
    similarity_score = azure_embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_dimension_property(azure_embedding_model: AzureOpenAIEmbeddings) -> None:
    """Test that AzureOpenAIEmbeddings has a valid dimension property."""
    assert isinstance(azure_embedding_model.dimension, int)
    assert azure_embedding_model.dimension > 0


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_is_available() -> None:
    """Test that AzureOpenAIEmbeddings is available with the correct environment variables."""
    assert (
        AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            deployment="text-embedding-3-small",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )._is_available()
        is True
    )


@pytest.mark.skipif(
    "AZURE_OPENAI_ENDPOINT" not in os.environ,
    reason="Skipping test because AZURE_OPENAI_ENDPOINT is not defined",
)
def test_repr(azure_embedding_model: AzureOpenAIEmbeddings) -> None:
    """Test the string representation of AzureOpenAIEmbeddings."""
    repr_str = repr(azure_embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("AzureOpenAIEmbeddings")
