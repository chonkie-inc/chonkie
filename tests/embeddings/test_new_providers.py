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


# ============================================================================
# MistralEmbeddings Tests
# ============================================================================

class TestMistralEmbeddings:
    """Test MistralEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=1024, model_name="mistral-embed")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return MistralEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "mistral-embed"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert MistralEmbeddings.DEFAULT_MODEL == "mistral-embed"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-key"}):
                e = MistralEmbeddings()
                assert e.model == "mistral-embed"

    def test_embed(self, embeddings, mock_client):
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(1, 1024).astype(np.float32)
        mock_client.embed.return_value = mock_response
        result = embeddings.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_batch(self, embeddings, mock_client):
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(3, 1024).astype(np.float32)
        mock_client.embed.return_value = mock_response
        results = embeddings.embed_batch(["a", "b", "c"])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_dimension(self, embeddings):
        assert isinstance(embeddings.dimension, int)
        assert embeddings.dimension == 1024

    def test_get_tokenizer(self, embeddings):
        assert embeddings.get_tokenizer() is not None

    def test_repr(self, embeddings):
        assert "MistralEmbeddings" in repr(embeddings)
        assert "mistral-embed" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(MistralEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(MistralEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                MistralEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            MistralEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"mistral": "my-key"}


# ============================================================================
# TogetherEmbeddings Tests
# ============================================================================

class TestTogetherEmbeddings:
    """Test TogetherEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=768, model_name="togethercomputer/m2-bert-80M-8k-retrieval")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return TogetherEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "togethercomputer/m2-bert-80M-8k-retrieval"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert TogetherEmbeddings.DEFAULT_MODEL == "togethercomputer/m2-bert-80M-8k-retrieval"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"TOGETHER_API_KEY": "env-key"}):
                e = TogetherEmbeddings()
                assert e.model == "togethercomputer/m2-bert-80M-8k-retrieval"

    def test_embed(self, embeddings, mock_client):
        mock_response = MagicMock()
        mock_response.to_numpy.return_value = np.random.rand(1, 768).astype(np.float32)
        mock_client.embed.return_value = mock_response
        result = embeddings.embed("test text")
        assert isinstance(result, np.ndarray)

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_repr(self, embeddings):
        assert "TogetherEmbeddings" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(TogetherEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(TogetherEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                TogetherEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            TogetherEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"together": "my-key"}


# ============================================================================
# MixedbreadEmbeddings Tests
# ============================================================================

class TestMixedbreadEmbeddings:
    """Test MixedbreadEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=1024, model_name="mxbai-embed-large-v1")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return MixedbreadEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "mxbai-embed-large-v1"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert MixedbreadEmbeddings.DEFAULT_MODEL == "mxbai-embed-large-v1"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"MIXEDBREAD_API_KEY": "env-key"}):
                e = MixedbreadEmbeddings()
                assert e.model == "mxbai-embed-large-v1"

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_repr(self, embeddings):
        assert "MixedbreadEmbeddings" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(MixedbreadEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(MixedbreadEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                MixedbreadEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            MixedbreadEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"mixedbread": "my-key"}


# ============================================================================
# NomicEmbeddings Tests
# ============================================================================

class TestNomicEmbeddings:
    """Test NomicEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=768, model_name="nomic-embed-text-v1.5")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return NomicEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "nomic-embed-text-v1.5"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert NomicEmbeddings.DEFAULT_MODEL == "nomic-embed-text-v1.5"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"NOMIC_API_KEY": "env-key"}):
                e = NomicEmbeddings()
                assert e.model == "nomic-embed-text-v1.5"

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_repr(self, embeddings):
        assert "NomicEmbeddings" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(NomicEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(NomicEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                NomicEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            NomicEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"nomic": "my-key"}


# ============================================================================
# DeepInfraEmbeddings Tests
# ============================================================================

class TestDeepInfraEmbeddings:
    """Test DeepInfraEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=1024, model_name="BAAI/bge-large-en-v1.5")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return DeepInfraEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "BAAI/bge-large-en-v1.5"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert DeepInfraEmbeddings.DEFAULT_MODEL == "BAAI/bge-large-en-v1.5"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "env-key"}):
                e = DeepInfraEmbeddings()
                assert e.model == "BAAI/bge-large-en-v1.5"

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_repr(self, embeddings):
        assert "DeepInfraEmbeddings" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(DeepInfraEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(DeepInfraEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                DeepInfraEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            DeepInfraEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"deepinfra": "my-key"}


# ============================================================================
# CloudflareEmbeddings Tests
# ============================================================================

class TestCloudflareEmbeddings:
    """Test CloudflareEmbeddings wrapper."""

    @pytest.fixture
    def mock_client(self):
        return make_mock_catsu_client(dimension=768, model_name="@cf/baai/bge-base-en-v1.5")

    @pytest.fixture
    def embeddings(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            return CloudflareEmbeddings(api_key="test-key")

    def test_initialization(self, embeddings):
        assert embeddings.model == "@cf/baai/bge-base-en-v1.5"
        assert embeddings._catsu is not None

    def test_default_model(self):
        assert CloudflareEmbeddings.DEFAULT_MODEL == "@cf/baai/bge-base-en-v1.5"

    def test_initialization_with_env_var(self, mock_client):
        with patch("catsu.Client", return_value=mock_client):
            with patch.dict(os.environ, {"CLOUDFLARE_API_KEY": "env-key"}):
                e = CloudflareEmbeddings()
                assert e.model == "@cf/baai/bge-base-en-v1.5"

    def test_embed_batch_empty(self, embeddings):
        assert embeddings.embed_batch([]) == []

    def test_repr(self, embeddings):
        assert "CloudflareEmbeddings" in repr(embeddings)

    def test_is_available(self):
        assert isinstance(CloudflareEmbeddings._is_available(), bool)

    def test_missing_catsu(self):
        with patch.object(CloudflareEmbeddings, "_is_available", return_value=False):
            with pytest.raises(ImportError, match=r"One \(or more\) of the following packages"):
                CloudflareEmbeddings(api_key="test-key")

    def test_correct_provider(self, mock_client):
        with patch("catsu.Client", return_value=mock_client) as mock_cls:
            CloudflareEmbeddings(api_key="my-key")
            assert mock_cls.call_args[1].get("api_keys") == {"cloudflare": "my-key"}


if __name__ == "__main__":
    pytest.main()
