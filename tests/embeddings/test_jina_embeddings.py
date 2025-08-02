"""Test suite for JinaEmbeddings."""

import json
import os
from typing import Any, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests

from chonkie.embeddings.jina import JinaEmbeddings


@pytest.fixture(autouse=True)
def mock_tokenizer() -> Generator[Any, None, None]:
    """Mock tokenizer initialization to avoid internet dependency in CI."""
    with patch('tokenizers.Tokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = Mock(ids=[1, 2, 3, 4, 5])
        mock_tokenizer_instance.encode_batch.return_value = [Mock(ids=[1, 2, 3]), Mock(ids=[4, 5, 6])]
        mock_tokenizer.return_value = mock_tokenizer_instance
        yield mock_tokenizer


class TestJinaEmbeddingsInitialization:
    """Test JinaEmbeddings initialization and configuration."""

    def test_initialization_with_api_key(self) -> None:
        """Test JinaEmbeddings initialization with explicit API key."""
        embeddings = JinaEmbeddings(api_key="test_key")
        assert embeddings.model == "jina-embeddings-v4"
        assert embeddings.task == "text-matching"
        assert embeddings.late_chunking is False
        assert embeddings.embedding_type == "float"
        assert embeddings.api_key == "test_key"
        assert embeddings.headers["Authorization"] == "Bearer test_key"
        assert embeddings.url == "https://api.jina.ai/v1/embeddings"
        assert embeddings.dimension == 2048

    def test_initialization_with_env_key(self) -> None:
        """Test JinaEmbeddings initialization using environment API key."""
        with patch.dict(os.environ, {"JINA_API_KEY": "env_test_key"}):
            embeddings = JinaEmbeddings()
            assert embeddings.api_key == "env_test_key"
            assert embeddings.headers["Authorization"] == "Bearer env_test_key"

    def test_initialization_without_api_key(self) -> None:
        """Test JinaEmbeddings initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key is required"):
                JinaEmbeddings()

    def test_initialization_with_custom_model(self) -> None:
        """Test JinaEmbeddings initialization with custom model."""
        embeddings = JinaEmbeddings(
            model="jina-embeddings-v2-base-en",
            api_key="test_key"
        )
        assert embeddings.model == "jina-embeddings-v2-base-en"
        assert embeddings.dimension == 768

    def test_initialization_with_invalid_model(self) -> None:
        """Test JinaEmbeddings initialization fails with invalid model."""
        with pytest.raises(ValueError, match="Model invalid-model not available"):
            JinaEmbeddings(model="invalid-model", api_key="test_key")

    def test_initialization_with_custom_parameters(self) -> None:
        """Test JinaEmbeddings initialization with custom parameters."""
        embeddings = JinaEmbeddings(
            model="jina-embeddings-v2-base-code",
            task="retrieval.passage",
            batch_size=16,
            max_retries=5,
            api_key="test_key"
        )
        assert embeddings.model == "jina-embeddings-v2-base-code"
        assert embeddings.task == "retrieval.passage"
        assert embeddings._batch_size == 16
        assert embeddings._max_retries == 5
        assert embeddings.dimension == 768

    def test_tokenizer_initialization(self) -> None:
        """Test that tokenizer is properly initialized."""
        with patch('tokenizers.Tokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.return_value = "mock_tokenizer"
            embeddings = JinaEmbeddings(api_key="test_key")
            assert embeddings._tokenizer == "mock_tokenizer"
            mock_tokenizer.assert_called_once_with("jinaai/jina-embeddings-v4")

    def test_tokenizer_initialization_failure(self) -> None:
        """Test tokenizer initialization failure handling."""
        with patch('tokenizers.Tokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Tokenizer error")
            with pytest.raises(ValueError, match="Failed to initialize tokenizer"):
                JinaEmbeddings(api_key="test_key")

    def test_repr(self) -> None:
        """Test the __repr__ method."""
        embeddings = JinaEmbeddings(api_key="test_key")
        repr_str = repr(embeddings)
        assert repr_str == "JinaEmbeddings(model=jina-embeddings-v4, dimensions=2048)"


class TestJinaEmbeddingsProperties:
    """Test JinaEmbeddings properties and methods."""

    @pytest.fixture
    def embeddings(self) -> JinaEmbeddings:
        """Create JinaEmbeddings instance for testing."""
        return JinaEmbeddings(api_key="test_key")

    def test_dimension_property(self, embeddings: JinaEmbeddings) -> None:
        """Test the dimension property returns correct value."""
        assert embeddings.dimension == 2048
        assert isinstance(embeddings.dimension, int)

    def test_get_tokenizer_or_token_counter(self, embeddings: JinaEmbeddings) -> None:
        """Test get_tokenizer_or_token_counter returns tokenizer instance."""
        tokenizer = embeddings.get_tokenizer_or_token_counter()
        assert tokenizer is embeddings._tokenizer
        # Since we're using a mock, check that it has the expected methods
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "encode_batch")

    def test_available_models_contains_expected_models(self) -> None:
        """Test that AVAILABLE_MODELS contains expected models."""
        expected_models = [
            "jina-embeddings-v3",
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-base-es",
            "jina-embeddings-v2-base-de",
            "jina-embeddings-v2-base-zh",
            "jina-embeddings-v2-base-code",
            "jina-embeddings-b-en-v1",
            "jina-embeddings-v4"
        ]
        for model in expected_models:
            assert model in JinaEmbeddings.AVAILABLE_MODELS
            assert isinstance(JinaEmbeddings.AVAILABLE_MODELS[model], int)

    def test_is_available(self, embeddings: JinaEmbeddings) -> None:
        """Test _is_available method."""
        # Should return True since dependencies are available in test environment
        assert embeddings._is_available() is True

    def test_import_dependencies_success(self, embeddings: JinaEmbeddings) -> None:
        """Test _import_dependencies when packages are available."""
        # Should not raise an exception
        embeddings._import_dependencies()

    def test_import_dependencies_failure(self) -> None:
        """Test _import_dependencies when packages are not available."""
        with patch('chonkie.embeddings.jina.JinaEmbeddings._is_available', return_value=False):
            with pytest.raises(ImportError, match="One \\(or more\\) of the following packages is not available"):
                JinaEmbeddings(api_key="test_key")


class TestJinaEmbeddingsAPIMocking:
    """Test JinaEmbeddings with mocked API responses."""

    @pytest.fixture
    def embeddings(self) -> JinaEmbeddings:
        """Create JinaEmbeddings instance for testing."""
        return JinaEmbeddings(api_key="test_key")

    @pytest.fixture
    def mock_single_embedding_response(self) -> dict[str, Any]:
        """Mock response for single embedding."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 2048  # Mock 2048-dimensional embedding
                }
            ],
            "model": "jina-embeddings-v4",
            "usage": {
                "total_tokens": 10,
                "prompt_tokens": 10
            }
        }

    @pytest.fixture
    def mock_batch_embedding_response(self) -> dict[str, Any]:
        """Mock response for batch embedding."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 2048
                },
                {
                    "object": "embedding", 
                    "index": 1,
                    "embedding": [0.2] * 2048
                },
                {
                    "object": "embedding",
                    "index": 2,
                    "embedding": [0.3] * 2048
                }
            ],
            "model": "jina-embeddings-v4",
            "usage": {
                "total_tokens": 30,
                "prompt_tokens": 30
            }
        }

    def test_embed_single_text_success(
        self, 
        embeddings: JinaEmbeddings, 
        mock_single_embedding_response: dict[str, Any]
    ) -> None:
        """Test successful single text embedding."""
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps(mock_single_embedding_response).encode()
            mock_post.return_value = mock_response

            result = embeddings.embed("Test text")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (2048,)
            assert result.dtype == np.float32
            np.testing.assert_array_equal(result, np.array([0.1] * 2048, dtype=np.float32))

            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['input'] == ["Test text"]
            assert call_args[1]['json']['model'] == "jina-embeddings-v4"

    def test_embed_empty_text_raises_error(self, embeddings: JinaEmbeddings) -> None:
        """Test that embedding empty text raises ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            embeddings.embed("")

    def test_embed_single_text_api_error(self, embeddings: JinaEmbeddings) -> None:
        """Test single text embedding with API error."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("API Error")
            
            with pytest.raises(ValueError, match="Failed to embed text.*after 3 attempts"):
                embeddings.embed("Test text")
            
            assert mock_post.call_count == 3  # Should retry 3 times

    def test_embed_single_text_invalid_response(self, embeddings: JinaEmbeddings) -> None:
        """Test single text embedding with invalid API response."""
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps({"invalid": "response"}).encode()
            mock_post.return_value = mock_response

            with pytest.raises(ValueError, match="Unexpected API response format"):
                embeddings.embed("Test text")

    def test_embed_batch_success(
        self, 
        embeddings: JinaEmbeddings, 
        mock_batch_embedding_response: dict[str, Any]
    ) -> None:
        """Test successful batch text embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps(mock_batch_embedding_response).encode()
            mock_post.return_value = mock_response

            results = embeddings.embed_batch(texts)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, np.ndarray)
                assert result.shape == (2048,)
                assert result.dtype == np.float32
                expected_value = [0.1, 0.2, 0.3][i]
                np.testing.assert_array_equal(result, np.array([expected_value] * 2048, dtype=np.float32))

            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['input'] == texts

    def test_embed_batch_empty_list(self, embeddings: JinaEmbeddings) -> None:
        """Test batch embedding with empty list."""
        result = embeddings.embed_batch([])
        assert result == []

    def test_embed_batch_with_fallback(
        self, 
        embeddings: JinaEmbeddings,
        mock_single_embedding_response: dict[str, Any]
    ) -> None:
        """Test batch embedding with fallback to single embeddings."""
        texts = ["Text 1", "Text 2"]
        
        with patch('requests.post') as mock_post:
            # First call (batch) fails, subsequent calls (single) succeed
            batch_response = requests.Response()
            batch_response.status_code = 400
            
            single_response = requests.Response()
            single_response.status_code = 200
            single_response._content = json.dumps(mock_single_embedding_response).encode()
            
            mock_post.side_effect = [
                requests.exceptions.HTTPError("Batch failed"),
                single_response,
                single_response
            ]

            with patch('warnings.warn') as mock_warn:
                results = embeddings.embed_batch(texts)
                
                assert len(results) == 2
                for result in results:
                    assert isinstance(result, np.ndarray)
                    assert result.shape == (2048,)
                
                # Should have warned about batch failure
                mock_warn.assert_called_once()
                assert "Failed to embed batch" in str(mock_warn.call_args[0][0])

    def test_embed_batch_large_batch_chunking(
        self, 
        embeddings: JinaEmbeddings
    ) -> None:
        """Test batch embedding with large batch that gets chunked."""
        # Create embeddings with small batch size
        embeddings._batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        def mock_response_side_effect(*args: Any, **kwargs: Any) -> requests.Response:
            """Create response based on batch size."""
            mock_response = requests.Response()
            mock_response.status_code = 200
            
            # Get the batch size from the request
            batch_input = kwargs.get('json', {}).get('input', [])
            batch_size = len(batch_input)
            
            response_data = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1 + i * 0.1] * 2048}
                    for i in range(batch_size)
                ],
                "model": "jina-embeddings-v4",
                "usage": {"total_tokens": batch_size * 10, "prompt_tokens": batch_size * 10}
            }
            mock_response._content = json.dumps(response_data).encode()
            return mock_response
        
        with patch('requests.post', side_effect=mock_response_side_effect) as mock_post:
            results = embeddings.embed_batch(texts)
            
            assert len(results) == 5
            # Should make 3 API calls (2+2+1 texts)
            assert mock_post.call_count == 3

    def test_similarity_calculation(self, embeddings: JinaEmbeddings) -> None:
        """Test similarity calculation between embeddings."""
        # Create test vectors
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        w = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Same as u
        
        # Test orthogonal vectors (similarity should be 0)
        sim_orthogonal = embeddings.similarity(u, v)
        assert isinstance(sim_orthogonal, np.float32)
        assert abs(sim_orthogonal) < 1e-6  # Should be very close to 0
        
        # Test identical vectors (similarity should be 1)
        sim_identical = embeddings.similarity(u, w)
        assert isinstance(sim_identical, np.float32)
        assert abs(float(sim_identical) - 1.0) < 1e-6  # Should be very close to 1


class TestJinaEmbeddingsErrorHandling:
    """Test JinaEmbeddings error handling scenarios."""

    @pytest.fixture
    def embeddings(self) -> JinaEmbeddings:
        """Create JinaEmbeddings instance for testing."""
        return JinaEmbeddings(api_key="test_key")

    def test_embed_with_http_error_retries(self, embeddings: JinaEmbeddings) -> None:
        """Test that embed retries on HTTP errors."""
        with patch('requests.post') as mock_post:
            # Fail first two attempts, succeed on third
            error_response = requests.Response()
            error_response.status_code = 500
            
            success_response = requests.Response()
            success_response.status_code = 200
            success_response._content = json.dumps({
                "data": [{"embedding": [0.1] * 2048}]
            }).encode()
            
            mock_post.side_effect = [
                requests.exceptions.HTTPError("Server error"),
                requests.exceptions.HTTPError("Server error"), 
                success_response
            ]

            with patch('warnings.warn') as mock_warn:
                result = embeddings.embed("Test text")
                assert isinstance(result, np.ndarray)
                
                # Should have warned about retries
                assert mock_warn.call_count == 2

    def test_embed_batch_single_text_failure(self, embeddings: JinaEmbeddings) -> None:
        """Test batch embedding when single text batch fails."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.HTTPError("API Error")
            
            with pytest.raises(ValueError, match="Failed to embed text.*due to"):
                embeddings.embed_batch(["Single text"])

    def test_embed_batch_invalid_text_type(self, embeddings: JinaEmbeddings) -> None:
        """Test batch embedding with invalid text type in fallback."""
        with patch('requests.post') as mock_post:
            # First call fails (triggering fallback)
            mock_post.side_effect = requests.exceptions.HTTPError("Batch failed")
            
            with pytest.raises(ValueError, match="Invalid text type found in batch"):
                embeddings.embed_batch([123, "valid text"])  # type: ignore

    def test_embed_with_response_status_error(self, embeddings: JinaEmbeddings) -> None:
        """Test embed handling response status errors."""
        with patch('requests.post') as mock_post:
            error_response = requests.Response()
            error_response.status_code = 401
            mock_post.return_value = error_response
            
            # Mock the raise_for_status to raise HTTPError
            with patch.object(error_response, 'raise_for_status', side_effect=requests.exceptions.HTTPError("Unauthorized")):
                with pytest.raises(ValueError, match="Failed to embed text.*after 3 attempts"):
                    embeddings.embed("Test text")


class TestJinaEmbeddingsIntegration:
    """Integration tests for JinaEmbeddings (these may need real API key)."""

    @pytest.fixture
    def skip_if_no_api_key(self) -> None:
        """Skip test if no API key is available."""
        if not os.environ.get("JINA_API_KEY"):
            pytest.skip("Skipping integration test because JINA_API_KEY is not set")

    @pytest.fixture
    def real_embeddings(self, skip_if_no_api_key: None) -> JinaEmbeddings:
        """Create real JinaEmbeddings instance for integration testing."""
        return JinaEmbeddings()

    def test_real_api_single_embedding(self, real_embeddings: JinaEmbeddings) -> None:
        """Test single embedding with real API (requires JINA_API_KEY)."""
        text = "This is a test sentence for Jina embeddings."
        result = real_embeddings.embed(text)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real_embeddings.dimension,)
        assert result.dtype == np.float32
        # Embeddings should not be all zeros
        assert not np.allclose(result, 0)

    def test_real_api_batch_embedding(self, real_embeddings: JinaEmbeddings) -> None:
        """Test batch embedding with real API (requires JINA_API_KEY)."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        results = real_embeddings.embed_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (real_embeddings.dimension,)
            assert result.dtype == np.float32
            assert not np.allclose(result, 0)

    def test_real_api_similarity(self, real_embeddings: JinaEmbeddings) -> None:
        """Test similarity calculation with real API (requires JINA_API_KEY)."""
        text1 = "The cat sits on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "The dog runs in the park."
        
        embedding1 = real_embeddings.embed(text1)
        embedding2 = real_embeddings.embed(text2)
        embedding3 = real_embeddings.embed(text3)
        
        # Similar sentences should have higher similarity
        sim_similar = real_embeddings.similarity(embedding1, embedding2)
        sim_different = real_embeddings.similarity(embedding1, embedding3)
        
        assert isinstance(sim_similar, np.float32)
        assert isinstance(sim_different, np.float32)
        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
        
        # Similar sentences should be more similar than different ones
        assert float(sim_similar) > float(sim_different)


class TestJinaEmbeddingsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def embeddings(self) -> JinaEmbeddings:
        """Create JinaEmbeddings instance for testing."""
        return JinaEmbeddings(api_key="test_key")

    def test_embed_very_long_text(self, embeddings: JinaEmbeddings) -> None:
        """Test embedding very long text."""
        long_text = "word " * 10000  # Very long text
        
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps({
                "data": [{"embedding": [0.1] * 2048}]
            }).encode()
            mock_post.return_value = mock_response

            result = embeddings.embed(long_text)
            assert isinstance(result, np.ndarray)
            assert result.shape == (2048,)

    def test_embed_unicode_text(self, embeddings: JinaEmbeddings) -> None:
        """Test embedding text with unicode characters."""
        unicode_text = "Hello 你好 🌍 café naïve résumé"
        
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps({
                "data": [{"embedding": [0.1] * 2048}]
            }).encode()
            mock_post.return_value = mock_response

            result = embeddings.embed(unicode_text)
            assert isinstance(result, np.ndarray)
            
            # Verify unicode text was sent correctly
            call_args = mock_post.call_args
            assert call_args[1]['json']['input'] == [unicode_text]

    def test_batch_embedding_with_mixed_lengths(self, embeddings: JinaEmbeddings) -> None:
        """Test batch embedding with texts of very different lengths."""
        texts = [
            "Short",
            "This is a medium length sentence for testing purposes.",
            "This is a very long sentence that contains many words and should test how the embedding model handles texts of varying lengths within the same batch call to ensure robust processing." * 5
        ]
        
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            response_data = {
                "data": [
                    {"embedding": [0.1] * 2048},
                    {"embedding": [0.2] * 2048},
                    {"embedding": [0.3] * 2048}
                ]
            }
            mock_response._content = json.dumps(response_data).encode()
            mock_post.return_value = mock_response

            results = embeddings.embed_batch(texts)
            assert len(results) == 3
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (2048,)

    def test_custom_batch_size_configuration(self) -> None:
        """Test that custom batch size is respected."""
        embeddings = JinaEmbeddings(api_key="test_key", batch_size=5)
        assert embeddings._batch_size == 5
        
        texts = ["text"] * 12  # 12 texts with batch size 5 should create 3 batches
        
        with patch('requests.post') as mock_post:
            mock_response = requests.Response()
            mock_response.status_code = 200
            response_data = {
                "data": [{"embedding": [0.1] * 2048}] * 5  # Max 5 per batch
            }
            mock_response._content = json.dumps(response_data).encode()
            mock_post.return_value = mock_response

            embeddings.embed_batch(texts)
            
            # Should make 3 API calls: 5+5+2
            assert mock_post.call_count == 3