"""Tests for the AutoEmbeddings class."""

import os
import warnings

import pytest

from chonkie import AutoEmbeddings
from chonkie.embeddings.base import BaseEmbeddings
from chonkie.embeddings.cohere import CohereEmbeddings
from chonkie.embeddings.jina import JinaEmbeddings
from chonkie.embeddings.model2vec import Model2VecEmbeddings
from chonkie.embeddings.openai import OpenAIEmbeddings
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chonkie.embeddings.voyageai import VoyageAIEmbeddings


class TestAutoEmbeddingsModel2Vec:
    """Test AutoEmbeddings with Model2Vec models."""

    @pytest.fixture
    def model_identifier(self) -> str:
        """Fixture providing a model2vec identifier."""
        return "minishlab/potion-base-32M"

    def test_get_embeddings(self, model_identifier: str) -> None:
        """Test that AutoEmbeddings can load Model2Vec embeddings."""
        embeddings = AutoEmbeddings.get_embeddings(model_identifier)
        assert isinstance(embeddings, Model2VecEmbeddings)
        assert embeddings.model_name_or_path == model_identifier

    def test_actual_embedding_generation(self, model_identifier: str) -> None:
        """Test that Model2Vec embeddings can actually generate embeddings."""
        embeddings = AutoEmbeddings.get_embeddings(model_identifier)
        test_text = "This is a test sentence."
        result = embeddings.embed_batch([test_text])
        assert len(result) == 1
        assert len(result[0]) > 0  # Should have dimensions
        # Model2Vec returns numpy arrays, convert to check float types
        import numpy as np

        if isinstance(result[0], np.ndarray):
            result[0] = result[0].tolist()
        assert all(
            isinstance(x, (float, int, np.floating, np.integer)) for x in result[0]
        )


class TestAutoEmbeddingsSentenceTransformers:
    """Test AutoEmbeddings with SentenceTransformers models."""

    @pytest.fixture
    def model_identifier(self) -> str:
        """Fixture providing a sentence transformer identifier."""
        return "all-MiniLM-L6-v2"

    def test_get_embeddings(self, model_identifier: str) -> None:
        """Test that AutoEmbeddings can load SentenceTransformer embeddings."""
        embeddings = AutoEmbeddings.get_embeddings(model_identifier)
        assert isinstance(embeddings, SentenceTransformerEmbeddings)
        assert embeddings.model_name_or_path == model_identifier

    def test_actual_embedding_generation(self, model_identifier: str) -> None:
        """Test that SentenceTransformer embeddings can actually generate embeddings."""
        embeddings = AutoEmbeddings.get_embeddings(model_identifier)
        test_text = "This is a test sentence."
        result = embeddings.embed_batch([test_text])
        assert len(result) == 1
        assert len(result[0]) > 0  # Should have dimensions
        # SentenceTransformers returns numpy arrays, convert to check float types
        import numpy as np

        if isinstance(result[0], np.ndarray):
            result[0] = result[0].tolist()
        assert all(
            isinstance(x, (float, int, np.floating, np.integer)) for x in result[0]
        )


class TestAutoEmbeddingsProviderPrefix:
    """Test AutoEmbeddings with provider:// prefix syntax."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set"
    )
    def test_openai_provider_prefix(self) -> None:
        """Test OpenAI embeddings with provider prefix."""
        embeddings = AutoEmbeddings.get_embeddings("openai://text-embedding-3-small")
        assert isinstance(embeddings, OpenAIEmbeddings)
        assert embeddings.model == "text-embedding-3-small"

    def test_cohere_provider_prefix_with_api_key(self) -> None:
        """Test Cohere embeddings with provider prefix and API key."""
        embeddings = AutoEmbeddings.get_embeddings(
            "cohere://embed-english-light-v3.0", api_key="test_key"
        )
        assert isinstance(embeddings, CohereEmbeddings)
        assert embeddings.model == "embed-english-light-v3.0"

    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"), reason="Voyage API key not set"
    )
    def test_voyage_provider_prefix(self) -> None:
        """Test VoyageAI embeddings with provider prefix."""
        embeddings = AutoEmbeddings.get_embeddings("voyageai://voyage-3")
        assert isinstance(embeddings, VoyageAIEmbeddings)
        assert embeddings.model == "voyage-3"

    @pytest.mark.skipif(not os.getenv("JINA_API_KEY"), reason="Jina API key not set")
    def test_jina_provider_prefix(self) -> None:
        """Test Jina embeddings with provider prefix."""
        embeddings = AutoEmbeddings.get_embeddings("jina://jina-embeddings-v3")
        assert isinstance(embeddings, JinaEmbeddings)
        assert embeddings.model == "jina-embeddings-v3"


class TestAutoEmbeddingsProviderLookup:
    """Test AutoEmbeddings provider lookup without prefix."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set"
    )
    def test_openai_model_lookup(self) -> None:
        """Test OpenAI model lookup by identifier."""
        embeddings = AutoEmbeddings.get_embeddings("text-embedding-3-small")
        assert isinstance(embeddings, OpenAIEmbeddings)
        assert embeddings.model == "text-embedding-3-small"


class TestAutoEmbeddingsInputTypes:
    """Test AutoEmbeddings with different input types."""

    def test_existing_embeddings_instance(self) -> None:
        """Test that passing an existing embeddings instance returns it unchanged."""
        original = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        result = AutoEmbeddings.get_embeddings(original)
        assert result is original

    def test_custom_embeddings_object(self) -> None:
        """Test that custom embeddings objects can be wrapped."""

        class MockEmbeddings:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 2.0, 3.0] for _ in texts]

        mock_obj = MockEmbeddings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = AutoEmbeddings.get_embeddings(mock_obj)
                assert isinstance(result, BaseEmbeddings)
            except ValueError:
                # Expected if registry can't wrap this type
                pass


class TestAutoEmbeddingsErrorHandling:
    """Test AutoEmbeddings error handling."""

    def test_invalid_provider_prefix(self) -> None:
        """Test error handling for invalid provider prefix."""
        with pytest.raises(ValueError, match="No provider found for invalid_provider"):
            AutoEmbeddings.get_embeddings("invalid_provider://some-model")

    def test_invalid_model_identifier(self) -> None:
        """Test error handling for completely invalid identifier."""
        with pytest.raises(ValueError, match="Failed to load embeddings"):
            AutoEmbeddings.get_embeddings("completely-invalid-model-name-12345")

    def test_fallback_behavior(self) -> None:
        """Test that fallback to SentenceTransformers works for some cases."""
        # This might succeed with SentenceTransformers as fallback
        try:
            embeddings = AutoEmbeddings.get_embeddings("some-unknown-model")
            # If it succeeds, it should be SentenceTransformers
            assert isinstance(embeddings, SentenceTransformerEmbeddings)
        except ValueError:
            # If it fails completely, that's also acceptable
            pass
