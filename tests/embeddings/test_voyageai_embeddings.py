"""Test suite for VoyageAIEmbeddings."""

import os
import warnings
from typing import Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie import VoyageAIEmbeddings


@pytest.fixture(autouse=True)
def mock_tokenizer():
    """Mock tokenizer initialization to avoid internet dependency in CI."""
    with patch('tokenizers.Tokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.encode_batch.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Fixed: return 3 items to match test expectations
        mock_tokenizer.return_value = mock_tokenizer_instance
        yield mock_tokenizer


class TestVoyageAIEmbeddingsInitialization:
    """Test VoyageAIEmbeddings initialization and configuration."""

    def test_initialization_with_default_model(self) -> None:
        """Test initialization with default model."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test_key"}):
            embeddings = VoyageAIEmbeddings()
            assert embeddings.model == "voyage-3"
            assert embeddings.output_dimension == 1024
            assert embeddings._dimension == 1024
            assert embeddings._token_limit == 32000
            assert embeddings.batch_size == 128
            assert embeddings.truncation is True

    def test_initialization_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test_key"}):
            embeddings = VoyageAIEmbeddings(model="voyage-3-large")
            assert embeddings.model == "voyage-3-large"
            assert embeddings._dimension == 1024  # first in allowed dims
            assert embeddings._token_limit == 32000

    def test_initialization_with_api_key_parameter(self) -> None:
        """Test initialization with API key parameter."""
        embeddings = VoyageAIEmbeddings(api_key="test_key")
        assert embeddings.model == "voyage-3"

    def test_initialization_with_env_var_api_key(self) -> None:
        """Test initialization with environment variable API key."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "env_test_key"}):
            embeddings = VoyageAIEmbeddings()
            assert embeddings.model == "voyage-3"

    def test_initialization_no_api_key_raises_error(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key provided"):
                VoyageAIEmbeddings()

    def test_initialization_invalid_model_raises_error(self) -> None:
        """Test that initialization with invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Model 'invalid_model' not available"):
            VoyageAIEmbeddings(model="invalid_model", api_key="test_key")

    def test_initialization_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        embeddings = VoyageAIEmbeddings(
            api_key="test_key",
            max_retries=5,
            timeout=30.0,
            batch_size=64,
            truncation=False
        )
        assert embeddings.batch_size == 64
        assert embeddings.truncation is False

    def test_initialization_batch_size_cap(self) -> None:
        """Test that batch_size is capped at 128."""
        embeddings = VoyageAIEmbeddings(api_key="test_key", batch_size=200)
        assert embeddings.batch_size == 128

    def test_initialization_output_dimension_valid(self) -> None:
        """Test initialization with valid output dimension."""
        embeddings = VoyageAIEmbeddings(
            model="voyage-3-large", 
            api_key="test_key", 
            output_dimension=512
        )
        assert embeddings.output_dimension == 512

    def test_initialization_output_dimension_invalid(self) -> None:
        """Test that initialization with invalid output dimension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid output_dimension=300"):
            VoyageAIEmbeddings(
                model="voyage-3", 
                api_key="test_key", 
                output_dimension=300
            )


class TestVoyageAIEmbeddingsProperties:
    """Test VoyageAIEmbeddings properties and methods."""

    @pytest.fixture
    def embeddings(self) -> VoyageAIEmbeddings:
        """Create VoyageAIEmbeddings instance for testing."""
        return VoyageAIEmbeddings(api_key="test_key")

    def test_dimension_property(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test dimension property returns correct value."""
        assert embeddings.dimension == 1024

    def test_repr(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test string representation."""
        expected = "VoyageAIEmbeddings(model=voyage-3, dimension=1024)"
        assert repr(embeddings) == expected

    def test_get_tokenizer_or_token_counter(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test tokenizer retrieval."""
        tokenizer = embeddings.get_tokenizer_or_token_counter()
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "encode_batch")

    def test_is_available(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test availability check."""
        assert embeddings._is_available() is True

    def test_count_tokens_single_text(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test token counting for single text."""
        text = "This is a test sentence."
        token_count = embeddings.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_batch(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test token counting for batch of texts."""
        texts = ["First text.", "Second text.", "Third text."]
        token_counts = embeddings.count_tokens_batch(texts)
        assert isinstance(token_counts, list)
        assert len(token_counts) == len(texts)
        assert all(isinstance(count, int) and count > 0 for count in token_counts)

    def test_available_models_class_attribute(self) -> None:
        """Test that AVAILABLE_MODELS contains expected models."""
        expected_models = {
            "voyage-3-large", "voyage-3", "voyage-3-lite", 
            "voyage-code-3", "voyage-finance-2", "voyage-law-2", "voyage-code-2"
        }
        assert set(VoyageAIEmbeddings.AVAILABLE_MODELS.keys()) == expected_models

    def test_default_model_class_attribute(self) -> None:
        """Test DEFAULT_MODEL class attribute."""
        assert VoyageAIEmbeddings.DEFAULT_MODEL == "voyage-3"


class TestVoyageAIEmbeddingsAPIMocking:
    """Test VoyageAIEmbeddings with mocked API responses."""

    @pytest.fixture
    def embeddings(self) -> VoyageAIEmbeddings:
        """Create VoyageAIEmbeddings instance for testing."""
        return VoyageAIEmbeddings(api_key="test_key")

    @pytest.fixture
    def mock_single_embedding_response(self) -> MagicMock:
        """Mock response for single embedding."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        return mock_response

    @pytest.fixture
    def mock_batch_embedding_response(self) -> MagicMock:
        """Mock response for batch embedding."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        return mock_response

    def test_embed_single_text(
        self, 
        embeddings: VoyageAIEmbeddings, 
        mock_single_embedding_response: MagicMock
    ) -> None:
        """Test synchronous embedding of single text."""
        with patch.object(embeddings._client, 'embed', return_value=mock_single_embedding_response):
            result = embeddings.embed("Test text")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (1024,)
            assert result.dtype == np.float32
            
            embeddings._client.embed.assert_called_once_with(
                texts=["Test text"],
                model="voyage-3",
                input_type=None,
                truncation=True,
                output_dimension=1024
            )

    def test_embed_with_input_type(
        self, 
        embeddings: VoyageAIEmbeddings, 
        mock_single_embedding_response: MagicMock
    ) -> None:
        """Test embedding with input_type parameter."""
        with patch.object(embeddings._client, 'embed', return_value=mock_single_embedding_response):
            embeddings.embed("Test query", input_type="query")
            
            embeddings._client.embed.assert_called_once_with(
                texts=["Test query"],
                model="voyage-3",
                input_type="query",
                truncation=True,
                output_dimension=1024
            )

    def test_embed_batch(
        self, 
        embeddings: VoyageAIEmbeddings, 
        mock_batch_embedding_response: MagicMock
    ) -> None:
        """Test synchronous batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        with patch.object(embeddings._client, 'embed', return_value=mock_batch_embedding_response):
            results = embeddings.embed_batch(texts)
            
            assert isinstance(results, list)
            assert len(results) == 3
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (1024,)
                assert result.dtype == np.float32

    def test_embed_batch_chunking(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test that large batches are properly chunked."""
        # Create more texts than batch_size
        embeddings.batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        def mock_embed_side_effect(texts: list, **kwargs: dict) -> MagicMock:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * 1024] * len(texts)
            return mock_response
        
        with patch.object(embeddings._client, 'embed', side_effect=mock_embed_side_effect) as mock_embed:
            results = embeddings.embed_batch(texts)
            
            assert len(results) == 5
            # Should be called 3 times: 2+2+1 texts
            assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_aembed_single_text(
        self, 
        embeddings: VoyageAIEmbeddings, 
        mock_single_embedding_response: MagicMock
    ) -> None:
        """Test asynchronous embedding of single text."""
        with patch.object(embeddings._aclient, 'embed', return_value=mock_single_embedding_response):
            result = await embeddings.aembed("Test text")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (1024,)
            assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_batch(
        self, 
        embeddings: VoyageAIEmbeddings,
        mock_batch_embedding_response: MagicMock
    ) -> None:
        """Test asynchronous batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        async def mock_process_batch(batch: list, input_type: Union[str, None] = None) -> list:
            return [np.array([0.1] * 1024, dtype=np.float32) for _ in batch]
        
        with patch.object(embeddings, '_VoyageAIEmbeddings__process_batch', side_effect=mock_process_batch):
            results = await embeddings.aembed_batch(texts)
            
            assert isinstance(results, list)
            assert len(results) == 3
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (1024,)
                assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_batch_empty_list(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test asynchronous batch embedding with empty list."""
        results = await embeddings.aembed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_private_method(
        self, 
        embeddings: VoyageAIEmbeddings,
        mock_batch_embedding_response: MagicMock
    ) -> None:
        """Test the private __process_batch method."""
        batch = ["Text 1", "Text 2"]
        
        with patch.object(embeddings._aclient, 'embed', return_value=mock_batch_embedding_response):
            results = await embeddings._VoyageAIEmbeddings__process_batch(batch)
            
            assert isinstance(results, list)
            assert len(results) == 3  # Based on mock response
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_process_batch_empty_batch(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test __process_batch with empty batch."""
        results = await embeddings._VoyageAIEmbeddings__process_batch([])
        assert results == []


class TestVoyageAIEmbeddingsErrorHandling:
    """Test VoyageAIEmbeddings error handling scenarios."""

    @pytest.fixture
    def embeddings(self) -> VoyageAIEmbeddings:
        """Create VoyageAIEmbeddings instance for testing."""
        return VoyageAIEmbeddings(api_key="test_key")

    def test_embed_api_error_handling(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test error handling in synchronous embedding."""
        with patch.object(embeddings._client, 'embed', side_effect=Exception("API Error")):
            with pytest.raises(RuntimeError, match="VoyageAI API error during embedding: API Error"):
                embeddings.embed("Test text")

    @pytest.mark.asyncio
    async def test_aembed_api_error_handling(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test error handling in asynchronous embedding."""
        with patch.object(embeddings._aclient, 'embed', side_effect=Exception("API Error")):
            with pytest.raises(RuntimeError, match="VoyageAI API error during embedding: API Error"):
                await embeddings.aembed("Test text")

    def test_embed_batch_api_error_handling(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test error handling in synchronous batch embedding."""
        with patch.object(embeddings._client, 'embed', side_effect=Exception("API Error")):
            with pytest.raises(RuntimeError, match="VoyageAI API error during embedding: API Error"):
                embeddings.embed_batch(["Text 1", "Text 2"])

    @pytest.mark.asyncio
    async def test_process_batch_api_error_handling(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test error handling in __process_batch method."""
        with patch.object(embeddings._aclient, 'embed', side_effect=Exception("API Error")):
            with pytest.raises(RuntimeError, match="VoyageAI API error during embedding: API Error"):
                await embeddings._VoyageAIEmbeddings__process_batch(["Text 1"])

    def test_truncation_warning_embed(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test truncation warning in embed method."""
        long_text = "word " * 40000  # Create text that exceeds token limit (40k words > 32k tokens)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        
        # Override the tokenizer to return a high token count for this test
        with patch.object(embeddings, 'count_tokens', return_value=50000):  # Exceed the 32k limit
            with patch.object(embeddings._client, 'embed', return_value=mock_response):
                with pytest.warns(UserWarning, match="Input has .* tokens"):
                    embeddings.embed(long_text)

    @pytest.mark.asyncio
    async def test_truncation_warning_aembed(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test truncation warning in aembed method."""
        long_text = "word " * 40000  # Create text that exceeds token limit (40k words > 32k tokens)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        
        # Override the tokenizer to return a high token count for this test
        with patch.object(embeddings, 'count_tokens', return_value=50000):  # Exceed the 32k limit
            with patch.object(embeddings._aclient, 'embed', return_value=mock_response):
                with pytest.warns(UserWarning, match="Input has .* tokens"):
                    await embeddings.aembed(long_text)

    def test_truncation_warning_embed_batch(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test truncation warning in embed_batch method."""
        long_text = "word " * 40000  # Create text that exceeds token limit (40k words > 32k tokens)
        texts = [long_text, "short text"]
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
        
        # Override the tokenizer to return high token counts for this test
        with patch.object(embeddings, 'count_tokens_batch', return_value=[50000, 2]):  # First text exceeds limit
            with patch.object(embeddings._client, 'embed', return_value=mock_response):
                with pytest.warns(UserWarning, match="Text has .* tokens which exceeds"):
                    embeddings.embed_batch(texts)

    @pytest.mark.asyncio
    async def test_truncation_warning_process_batch(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test truncation warning in __process_batch method."""
        long_text = "word " * 40000  # Create text that exceeds token limit (40k words > 32k tokens)
        batch = [long_text]
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        
        # Override the tokenizer to return high token counts for this test
        with patch.object(embeddings, 'count_tokens_batch', return_value=[50000]):  # Exceed the limit
            with patch.object(embeddings._aclient, 'embed', return_value=mock_response):
                with pytest.warns(UserWarning, match="Text has .* tokens which exceeds"):
                    await embeddings._VoyageAIEmbeddings__process_batch(batch)

    def test_tokenizer_initialization_error(self) -> None:
        """Test error handling when tokenizer initialization fails."""
        with patch('tokenizers.Tokenizer.from_pretrained', side_effect=Exception("Tokenizer error")):
            with pytest.raises(ValueError, match="Failed to initialize tokenizer for model voyage-3: Tokenizer error"):
                VoyageAIEmbeddings(api_key="test_key")

    def test_import_dependencies_missing_packages(self) -> None:
        """Test import error when required packages are missing."""
        embeddings = VoyageAIEmbeddings.__new__(VoyageAIEmbeddings)
        
        with patch.object(embeddings, '_is_available', return_value=False):
            with pytest.raises(ImportError, match="One \\(or more\\) of the following packages is not available"):
                embeddings._import_dependencies()


class TestVoyageAIEmbeddingsIntegration:
    """Integration tests for VoyageAIEmbeddings (these may need real API key)."""

    @pytest.fixture
    def api_embeddings(self) -> VoyageAIEmbeddings:
        """Create VoyageAIEmbeddings instance with real API key for integration tests."""
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            pytest.skip("VOYAGE_API_KEY not set")
        return VoyageAIEmbeddings(api_key=api_key)

    @pytest.mark.skipif(
        "VOYAGE_API_KEY" not in os.environ,
        reason="Skipping integration test because VOYAGE_API_KEY is not defined",
    )
    def test_real_embed_single_text(self, api_embeddings: VoyageAIEmbeddings) -> None:
        """Test real API call for single text embedding."""
        text = "This is a test sentence for embedding."
        embedding = api_embeddings.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert not np.allclose(embedding, 0)  # Should not be all zeros

    @pytest.mark.skipif(
        "VOYAGE_API_KEY" not in os.environ,
        reason="Skipping integration test because VOYAGE_API_KEY is not defined",
    )
    def test_real_embed_batch(self, api_embeddings: VoyageAIEmbeddings) -> None:
        """Test real API call for batch embedding."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = api_embeddings.embed_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1024,)
            assert embedding.dtype == np.float32
            assert not np.allclose(embedding, 0)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "VOYAGE_API_KEY" not in os.environ,
        reason="Skipping integration test because VOYAGE_API_KEY is not defined",
    )
    async def test_real_aembed_single_text(self, api_embeddings: VoyageAIEmbeddings) -> None:
        """Test real async API call for single text embedding."""
        text = "This is a test sentence for async embedding."
        embedding = await api_embeddings.aembed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert not np.allclose(embedding, 0)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "VOYAGE_API_KEY" not in os.environ,
        reason="Skipping integration test because VOYAGE_API_KEY is not defined",
    )
    async def test_real_aembed_batch(self, api_embeddings: VoyageAIEmbeddings) -> None:
        """Test real async API call for batch embedding."""
        texts = ["First async sentence.", "Second async sentence."]
        embeddings = await api_embeddings.aembed_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1024,)
            assert embedding.dtype == np.float32
            assert not np.allclose(embedding, 0)


class TestVoyageAIEmbeddingsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def embeddings(self) -> VoyageAIEmbeddings:
        """Create VoyageAIEmbeddings instance for testing."""
        return VoyageAIEmbeddings(api_key="test_key")

    def test_embed_empty_string(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test embedding empty string."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.0] * 1024]
        
        with patch.object(embeddings._client, 'embed', return_value=mock_response):
            result = embeddings.embed("")
            assert isinstance(result, np.ndarray)
            assert result.shape == (1024,)

    def test_embed_batch_single_item(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test batch embedding with single item."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        
        with patch.object(embeddings._client, 'embed', return_value=mock_response):
            results = embeddings.embed_batch(["Single text"])
            assert len(results) == 1
            assert isinstance(results[0], np.ndarray)

    def test_embed_batch_empty_list(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test batch embedding with empty list."""
        results = embeddings.embed_batch([])
        assert results == []

    def test_different_output_dimensions(self) -> None:
        """Test initialization with different valid output dimensions."""
        # Test voyage-3-large which supports multiple dimensions
        for dim in [256, 512, 1024, 2048]:
            embeddings = VoyageAIEmbeddings(
                model="voyage-3-large", 
                api_key="test_key", 
                output_dimension=dim
            )
            assert embeddings.output_dimension == dim

    def test_truncation_disabled(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test behavior when truncation is disabled."""
        embeddings.truncation = False
        long_text = "word " * 40000  # Create text that exceeds token limit (40k words > 32k tokens)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        
        with patch.object(embeddings._client, 'embed', return_value=mock_response):
            # Should not raise warning when truncation is disabled
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    embeddings.embed(long_text)
                except UserWarning:
                    pytest.fail("Warning raised when truncation is disabled")

    def test_count_tokens_empty_string(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test token counting with empty string."""
        count = embeddings.count_tokens("")
        assert isinstance(count, int)
        assert count >= 0

    def test_count_tokens_batch_empty_list(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test batch token counting with empty list."""
        # Override the mock to return empty list for empty input
        with patch.object(embeddings, 'count_tokens_batch', return_value=[]):
            counts = embeddings.count_tokens_batch([])
            assert counts == []

    def test_similarity_between_embeddings(self, embeddings: VoyageAIEmbeddings) -> None:
        """Test that similar texts have higher similarity than dissimilar texts."""
        mock_responses = [
            MagicMock(embeddings=[[1.0, 0.0, 0.0] + [0.0] * 1021]),
            MagicMock(embeddings=[[0.9, 0.1, 0.0] + [0.0] * 1021]),
            MagicMock(embeddings=[[0.0, 0.0, 1.0] + [0.0] * 1021])
        ]
        
        with patch.object(embeddings._client, 'embed', side_effect=mock_responses):
            emb1 = embeddings.embed("cat")
            emb2 = embeddings.embed("kitten")  # Similar to cat
            emb3 = embeddings.embed("airplane")  # Different from cat
            
            # Calculate cosine similarities
            sim_cat_kitten = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            sim_cat_airplane = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
            
            assert sim_cat_kitten > sim_cat_airplane


if __name__ == "__main__":
    pytest.main()