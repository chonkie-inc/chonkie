"""Tests for the AutoEmbeddings class."""

import os

import pytest

from chonkie import AutoEmbeddings
from chonkie.embeddings.cohere import CohereEmbeddings
from chonkie.embeddings.model2vec import Model2VecEmbeddings
from chonkie.embeddings.ollama import OllamaEmbeddings
from chonkie.embeddings.openai import OpenAIEmbeddings
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings


@pytest.fixture
def model2vec_identifier() -> str:
    """Fixture providing a model2vec identifier."""
    return "minishlab/potion-base-8M"


@pytest.fixture
def sentence_transformer_identifier() -> str:
    """Fixture providing a sentence transformer identifier."""
    return "all-MiniLM-L6-v2"


@pytest.fixture
def sentence_transformer_identifier_small() -> str:
    """Fixture providing a small sentence transformer identifier."""
    return "all-minilm-l6-v2"


@pytest.fixture
def openai_identifier() -> str:
    """Fixture providing an OpenAI identifier."""
    return "text-embedding-3-small"


@pytest.fixture
def cohere_identifier() -> str:
    """Fixture providing an Cohere identifier."""
    return "embed-english-light-v3.0"


@pytest.fixture
def ollama_identifier():
    """Fixture providing an Ollama identifier."""
    return "all-minilm"


@pytest.fixture
def invalid_identifier() -> str:
    """Fixture providing an invalid identifier."""
    return "invalid-identifier-123"


def test_auto_embeddings_model2vec(model2vec_identifier: str) -> None:
    """Test that the AutoEmbeddings class can get model2vec embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(model2vec_identifier)
    assert isinstance(embeddings, Model2VecEmbeddings)
    assert embeddings.model_name_or_path == model2vec_identifier


def test_auto_embeddings_sentence_transformer(
    sentence_transformer_identifier: str,
) -> None:
    """Test that the AutoEmbeddings class can get sentence transformer embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier


def test_auto_embeddings_sentence_transformer_alt(
    sentence_transformer_identifier_small: str,
) -> None:
    """Test that the AutoEmbeddings class can get a small sentence transformer embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier_small)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier_small


def test_auto_embeddings_openai(openai_identifier: str) -> None:
    """Test that the AutoEmbeddings class can get OpenAI embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(
        openai_identifier, api_key="your_openai_api_key"
    )
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == openai_identifier


def test_auto_embeddings_cohere(cohere_identifier: str) -> None:
    """Test that the AutoEmbeddings class can get Cohere embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(
        cohere_identifier, api_key="your_cohere_api_key"
    )
    assert isinstance(embeddings, CohereEmbeddings)
    assert embeddings.model == cohere_identifier


@pytest.mark.skipif(
    "OLLAMA_ENDPOINT" not in os.environ,
    reason="Skipping test because OLLAMA_ENDPOINT is not defined",
)
def test_auto_embeddings_ollama(ollama_identifier):
    """Test that the AutoEmbeddings class can get Ollama embeddings."""
    embeddings = AutoEmbeddings.get_embeddings(
        ollama_identifier
    )
    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.model == ollama_identifier 


def test_auto_embeddings_invalid_identifier(invalid_identifier: str) -> None:
    """Test that the AutoEmbeddings class raises an error for an invalid identifier."""
    with pytest.raises(ValueError):
        AutoEmbeddings.get_embeddings(invalid_identifier)


if __name__ == "__main__":
    pytest.main()
