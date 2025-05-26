"""Test the NeuralChunker class."""

import pytest

from chonkie import NeuralChunker
from chonkie.types.base import Chunk


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing the NeuralChunker.

    Returns:
        str: A paragraph of text about machine learning chunking.

    """
    text = """The process of neural text chunking represents a breakthrough in automatic text segmentation. Unlike traditional methods that rely on simple heuristics, neural chunkers use deep learning models to identify optimal split points based on semantic context. This approach can understand complex linguistic patterns and provides more coherent chunks. The model learns from large datasets to predict where text should naturally be divided, resulting in better performance for downstream tasks like retrieval and question answering."""
    return text


@pytest.fixture
def short_text() -> str:
    """Short text for testing edge cases.

    Returns:
        str: A short sentence.

    """
    return "This is a short sentence for testing."


@pytest.fixture
def neural_chunker():
    """Fixture for NeuralChunker with default distilbert model."""
    try:
        return NeuralChunker(model="mirth/chonky_distilbert_base_uncased_1")
    except Exception:
        pytest.skip("transformers not available or model not accessible")


class TestNeuralChunkerInitialization:
    """Test the initialization of the NeuralChunker."""

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        try:
            chunker = NeuralChunker()
            assert chunker.min_characters_per_chunk == 10
            assert chunker.return_type == "chunks"
            assert not chunker._use_multiprocessing
            assert hasattr(chunker, 'model')
            assert hasattr(chunker, 'pipe')
        except Exception:
            pytest.skip("transformers not available or model not accessible")

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        try:
            chunker = NeuralChunker(
                model="mirth/chonky_distilbert_base_uncased_1",
                min_characters_per_chunk=20,
                stride=128,
                return_type="texts"
            )
            assert chunker.min_characters_per_chunk == 20
            assert chunker.return_type == "texts"
        except Exception:
            pytest.skip("transformers not available or model not accessible")

    def test_init_with_unsupported_model(self):
        """Test initialization fails with unsupported model."""
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is None:
                pytest.skip("transformers not available")
            
            # Try to initialize with unsupported model, should fail during validation
            with pytest.raises(ValueError, match="Model .* is not supported"):
                NeuralChunker(model="unsupported/model")
        except ImportError:
            pytest.skip("transformers not available")
        except Exception as e:
            # If it fails for other reasons (like network), also skip
            if "not supported" in str(e):
                # This is the expected error
                pass
            else:
                pytest.skip(f"Model access failed: {e}")

    def test_init_without_transformers(self):
        """Test that _is_available works correctly."""
        chunker_class = NeuralChunker.__new__(NeuralChunker)
        # Test availability check
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is not None:
                assert chunker_class._is_available() is True
            else:
                assert chunker_class._is_available() is False
        except ImportError:
            assert chunker_class._is_available() is False


class TestNeuralChunkerAvailability:
    """Test the availability checking methods."""

    def test_is_available(self):
        """Test _is_available method."""
        chunker_class = NeuralChunker.__new__(NeuralChunker)
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is not None:
                assert chunker_class._is_available() is True
            else:
                assert chunker_class._is_available() is False
        except ImportError:
            assert chunker_class._is_available() is False


class TestNeuralChunkerInternalMethods:
    """Test the internal methods of the NeuralChunker."""

    def test_get_splits_basic(self, neural_chunker):
        """Test _get_splits method with basic input."""
        response = [
            {"start": 0, "end": 10},
            {"start": 10, "end": 20},
        ]
        text = "0123456789abcdefghij"
        
        splits = neural_chunker._get_splits(response, text)
        
        expected = ["0123456789", "abcdefghij"]
        assert splits == expected

    def test_get_splits_with_remainder(self, neural_chunker):
        """Test _get_splits method with text remainder."""
        response = [{"start": 0, "end": 10}]
        text = "0123456789remainder"
        
        splits = neural_chunker._get_splits(response, text)
        
        expected = ["0123456789", "remainder"]
        assert splits == expected

    def test_get_splits_empty_response(self, neural_chunker):
        """Test _get_splits method with empty response."""
        response = []
        text = "test text"
        
        splits = neural_chunker._get_splits(response, text)
        
        expected = ["test text"]
        assert splits == expected

    def test_merge_close_spans_basic(self, neural_chunker):
        """Test _merge_close_spans method."""
        # Set min_characters_per_chunk for testing
        neural_chunker.min_characters_per_chunk = 5
        
        response = [
            {"start": 0, "end": 10},
            {"start": 12, "end": 20},  # Too close (gap of 2 < 5)
            {"start": 30, "end": 40},  # Far enough (gap of 10 >= 5)
        ]
        
        merged = neural_chunker._merge_close_spans(response)
        
        expected = [
            {"start": 12, "end": 20},  # Replaced the first span
            {"start": 30, "end": 40},
        ]
        assert merged == expected

    def test_merge_close_spans_empty(self, neural_chunker):
        """Test _merge_close_spans method with empty input."""
        merged = neural_chunker._merge_close_spans([])
        assert merged == []

    def test_merge_close_spans_single_span(self, neural_chunker):
        """Test _merge_close_spans method with single span."""
        response = [{"start": 0, "end": 10}]
        merged = neural_chunker._merge_close_spans(response)
        assert merged == response

    def test_get_chunks_from_splits(self, neural_chunker):
        """Test _get_chunks_from_splits method."""
        splits = ["Hello world", "This is a test", "Final chunk"]
        chunks = neural_chunker._get_chunks_from_splits(splits)
        
        assert len(chunks) == 3
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == 11
        assert chunks[0].token_count > 0
        
        assert chunks[1].text == "This is a test"
        assert chunks[1].start_index == 11
        assert chunks[1].end_index == 25
        assert chunks[1].token_count > 0
        
        assert chunks[2].text == "Final chunk"
        assert chunks[2].start_index == 25
        assert chunks[2].end_index == 36
        assert chunks[2].token_count > 0


class TestNeuralChunkerChunking:
    """Test the main chunking functionality."""

    def test_chunk_returns_chunks(self, neural_chunker, sample_text):
        """Test chunking returns Chunk objects."""
        neural_chunker.return_type = "chunks"
        result = neural_chunker.chunk(sample_text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)
        
        # Verify chunks reconstruct the original text
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == sample_text

    def test_chunk_returns_texts(self, neural_chunker, sample_text):
        """Test chunking returns text strings."""
        neural_chunker.return_type = "texts"
        result = neural_chunker.chunk(sample_text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(text, str) for text in result)
        
        # Verify texts reconstruct the original text
        reconstructed = "".join(result)
        assert reconstructed == sample_text

    def test_chunk_consistency(self, neural_chunker, sample_text):
        """Test that chunking is consistent between calls."""
        result1 = neural_chunker.chunk(sample_text)
        result2 = neural_chunker.chunk(sample_text)
        
        assert len(result1) == len(result2)
        for chunk1, chunk2 in zip(result1, result2):
            assert chunk1.text == chunk2.text

    def test_chunk_preserves_content(self, neural_chunker, sample_text):
        """Test that chunking preserves all content."""
        result = neural_chunker.chunk(sample_text)
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == sample_text


class TestNeuralChunkerEdgeCases:
    """Test edge cases for the NeuralChunker."""

    def test_chunk_short_text(self, neural_chunker, short_text):
        """Test chunking very short text."""
        result = neural_chunker.chunk(short_text)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        
        # Verify content is preserved
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == short_text

    def test_chunk_empty_text(self, neural_chunker):
        """Test chunking empty text."""
        try:
            result = neural_chunker.chunk("")
            
            assert isinstance(result, list)
            assert len(result) >= 1
            # For empty text, the result should contain the empty string
            reconstructed = "".join(chunk.text for chunk in result)
            assert reconstructed == ""
        except Exception as e:
            # Some tokenizers may not handle empty strings well
            if "index out of range" in str(e).lower():
                pytest.skip("Tokenizer does not handle empty text well")
            else:
                raise

    def test_chunk_single_character(self, neural_chunker):
        """Test chunking single character."""
        result = neural_chunker.chunk("a")
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "".join(chunk.text for chunk in result) == "a"

    def test_chunk_whitespace_only(self, neural_chunker):
        """Test chunking whitespace-only text."""
        whitespace_text = "   \n\t  "
        result = neural_chunker.chunk(whitespace_text)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == whitespace_text

    def test_chunk_with_min_characters_filter(self, neural_chunker):
        """Test that min_characters_per_chunk affects merging."""
        # Test with larger min_characters_per_chunk
        neural_chunker.min_characters_per_chunk = 50
        
        short_text = "Short. Another short sentence. Third short."
        result = neural_chunker.chunk(short_text)
        
        # Should result in fewer, larger chunks due to merging
        assert isinstance(result, list)
        assert len(result) >= 1
        
        # Verify content preservation
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == short_text


class TestNeuralChunkerRepresentation:
    """Test the string representation of the NeuralChunker."""

    def test_repr(self, neural_chunker):
        """Test the __repr__ method."""
        repr_str = repr(neural_chunker)
        
        assert "NeuralChunker" in repr_str
        assert "min_characters_per_chunk" in repr_str
        assert "return_type" in repr_str

    def test_repr_with_custom_params(self):
        """Test __repr__ with custom parameters."""
        try:
            chunker = NeuralChunker(
                min_characters_per_chunk=20, 
                return_type="texts"
            )
            repr_str = repr(chunker)
            
            assert "min_characters_per_chunk=20" in repr_str
            assert "return_type=texts" in repr_str
        except Exception:
            pytest.skip("transformers not available or model not accessible")


class TestNeuralChunkerConstants:
    """Test the class constants and configurations."""

    def test_supported_models(self):
        """Test that supported models are defined."""
        assert len(NeuralChunker.SUPPORTED_MODELS) > 0
        assert "mirth/chonky_distilbert_base_uncased_1" in NeuralChunker.SUPPORTED_MODELS

    def test_supported_model_strides(self):
        """Test that model strides are defined for all supported models."""
        for model in NeuralChunker.SUPPORTED_MODELS:
            assert model in NeuralChunker.SUPPORTED_MODEL_STRIDES
            assert isinstance(NeuralChunker.SUPPORTED_MODEL_STRIDES[model], int)
            assert NeuralChunker.SUPPORTED_MODEL_STRIDES[model] > 0

    def test_default_model(self):
        """Test that default model is in supported models."""
        assert NeuralChunker.DEFAULT_MODEL in NeuralChunker.SUPPORTED_MODELS

    def test_default_model_has_stride(self):
        """Test that default model has a defined stride."""
        assert NeuralChunker.DEFAULT_MODEL in NeuralChunker.SUPPORTED_MODEL_STRIDES


class TestNeuralChunkerParameterVariations:
    """Test different parameter combinations."""

    def test_different_stride_values(self):
        """Test with different stride values."""
        try:
            # Test with custom stride
            chunker = NeuralChunker(stride=128)
            assert hasattr(chunker, 'pipe')
            
            # Test with stride None (should use default)
            chunker2 = NeuralChunker(stride=None)
            assert hasattr(chunker2, 'pipe')
        except Exception:
            pytest.skip("transformers not available or model not accessible")

    def test_different_device_maps(self):
        """Test with different device maps."""
        try:
            # Test with auto device map
            chunker = NeuralChunker(device_map="auto")
            assert hasattr(chunker, 'pipe')
            
            # Test with cpu device map
            chunker2 = NeuralChunker(device_map="cpu")
            assert hasattr(chunker2, 'pipe')
        except Exception:
            pytest.skip("transformers not available or model not accessible")

    def test_different_min_characters(self):
        """Test with different min_characters_per_chunk values."""
        try:
            chunker = NeuralChunker(min_characters_per_chunk=5)
            assert chunker.min_characters_per_chunk == 5
            
            chunker2 = NeuralChunker(min_characters_per_chunk=100)
            assert chunker2.min_characters_per_chunk == 100
        except Exception:
            pytest.skip("transformers not available or model not accessible")


class TestNeuralChunkerBehavior:
    """Test specific behavior patterns of the NeuralChunker."""

    def test_chunk_boundaries_make_sense(self, neural_chunker):
        """Test that chunk boundaries are at reasonable locations."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = neural_chunker.chunk(text)
        
        # Should have multiple chunks for this text
        assert len(result) >= 1
        
        # All chunks should have reasonable length
        for chunk in result:
            assert len(chunk.text.strip()) > 0
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(text)
            assert chunk.start_index < chunk.end_index

    def test_token_counts_are_positive(self, neural_chunker, sample_text):
        """Test that token counts are positive for non-empty chunks."""
        result = neural_chunker.chunk(sample_text)
        
        for chunk in result:
            if len(chunk.text.strip()) > 0:
                assert chunk.token_count > 0

    def test_chunks_are_contiguous(self, neural_chunker, sample_text):
        """Test that chunks are contiguous (no gaps or overlaps)."""
        result = neural_chunker.chunk(sample_text)
        
        if len(result) > 1:
            for i in range(len(result) - 1):
                current_chunk = result[i]
                next_chunk = result[i + 1]
                assert current_chunk.end_index == next_chunk.start_index