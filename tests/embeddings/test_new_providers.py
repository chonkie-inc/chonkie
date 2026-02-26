"""Test suite for new provider wrapper classes (Mistral, Together, Mixedbread, Nomic, DeepInfra, Cloudflare)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.embeddings.cloudflare import CloudflareEmbeddings
from chonkie.embeddings.deepinfra import DeepInfraEmbeddings
from chonkie.embeddings.mistral import MistralEmbeddings
from chonkie.embeddings.mixedbread import MixedbreadEmbeddings
from chonkie.embeddings.nomic import NomicEmbeddings
from chonkie.embeddings.together import TogetherEmbeddings


def make_mock_catsu_client(dimension: int = 1024, model_name: str = "default-model"):
    """Create a mock Catsu client for testing."""
    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, dimension).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = model_name
    mock_model_info.dimensions = dimension
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client


PROVIDER_TEST_PARAMS = [
    pytest.param(
        MistralEmbeddings,
        "mistral-embed",
        1024,
        "MISTRAL_API_KEY",
        "mistral",
        id="mistral",
    ),
    pytest.param(
        TogetherEmbeddings,
        "togethercomputer/m2-bert-80M-8k-retrieval",
        768,
        "TOGETHER_API_KEY",
        "together",
        id="together",
    ),
    pytest.param(
        MixedbreadEmbeddings,
        "mxbai-embed-large-v1",
        1024,
        "MIXEDBREAD_API_KEY",
        "mixedbread",
        id="mixedbread",
    ),
    pytest.param(
        NomicEmbeddings,
        "nomic-embed-text-v1.5",
        768,
        "NOMIC_API_KEY",
        "nomic",
        id="nomic",
    ),
    pytest.param(
        DeepInfraEmbeddings,
        "BAAI/bge-large-en-v1.5",
        1024,
        "DEEPINFRA_API_KEY",
        "deepinfra",
        id="deepinfra",
    ),
    pytest.param(
        CloudflareEmbeddings,
        "@cf/baai/bge-base-en-v1.5",
        768,
        "CLOUDFLARE_API_KEY",
        "cloudflare",
        id="cloudflare",
    ),
]


@pytest.mark.parametrize(
    "provider_class,default_model,dimension,api_key_env,provider_name",
    PROVIDER_TEST_PARAMS,
)
class TestNewProviderEmbeddings:
    """Parametrized tests for all new provider wrapper classes."""

    @pytest.fixture
    def mock_client(self, default_model, dimension):
        return make_mock_catsu_client(dimension=dimension, model_name=default_model)

    @pytest.fixture
    def embeddings(self, provider_class, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return provider_class(api_key="test-key")

    def test_initialization(self, embeddings, default_model):
        assert embeddings.model == default_model
        assert embeddings._catsu is not None

    def test_default_model(self, provider_class, default_model):
        assert provider_class.DEFAULT_MODEL == default_model

    def test_initialization_with_env_var(
        self, provider_class, mock_client, api_key_env, default_model
    ):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {api_key_env: "env-key"}):
                e = provider_class()
                assert e.model == default_model

    def test_embed(self, embeddings, mock_client, dimension):
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(1, dimension).astype(np.float32)
        mock_client.embed.return_value = mock_response
        result = embeddings.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_batch(self, embeddings, mock_client, dimension):
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(3, dimension).astype(np.float32)
        mock_client.embed.return_value = mock_response
        results = embeddings.embed_batch(["a", "b", "c"])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_dimension(self, embeddings, dimension):
        assert isinstance(embeddings.dimension, int)
        assert embeddings.dimension == dimension

    def test_get_tokenizer(self, embeddings):
        assert embeddings.get_tokenizer() is not None

    def test_repr(self, embeddings, provider_class, default_model):
        class_name = provider_class.__name__
        repr_str = repr(embeddings)
        assert class_name in repr_str
        assert default_model in repr_str

    def test_is_available(self, provider_class):
        assert isinstance(provider_class._is_available(), bool)

    def test_missing_catsu(self, provider_class):
        with patch.object(provider_class, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                provider_class(api_key="test-key")

    def test_correct_provider(self, provider_class, mock_client, provider_name):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            provider_class(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {provider_name: "my-key"}


if __name__ == "__main__":
    pytest.main()
