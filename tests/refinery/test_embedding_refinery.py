"""Test the EmbeddingsRefinery module."""

import pytest
from typing import TYPE_CHECKING, Callable
from unittest.mock import Mock, patch

from chonkie.embeddings import BaseEmbeddings
from chonkie.refinery import EmbeddingsRefinery
from chonkie.types import Chunk

if TYPE_CHECKING:
    import numpy as np


class MockEmbeddings(BaseEmbeddings):
    """Mock embeddings class for testing."""
    
    def __init__(self, dimension: int = 128, available: bool = True):
        """Initialize MockEmbeddings."""
        self._dimension = dimension
        self._available = available
        # Skip the parent __init__ to avoid dependency checks
    
    def _is_available(self) -> bool:
        """Check if mock embeddings are available."""
        return self._available
    
    def embed(self, text: str) -> "np.ndarray":
        """Mock embed method."""
        import numpy as np
        # Simple mock: return array of floats based on text length
        return np.array([float(i % 10) for i in range(self._dimension)], dtype=np.float32)
    
    def embed_batch(self, texts: list[str]) -> list["np.ndarray"]:
        """Mock batch embed method."""
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Mock dimension property."""
        return self._dimension
    
    def count_tokens(self, text: str) -> int:
        """Mock token counting method."""
        # Simple mock: split by whitespace
        return len(text.split())
    
    def get_tokenizer_or_token_counter(self) -> Callable[[str], int]:
        """Return the token counter function."""
        return self.count_tokens


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Fixture to create sample chunks."""
    return [
        Chunk(text="This is the first chunk.", start_index=0, end_index=24, token_count=5),
        Chunk(text="This is the second chunk.", start_index=25, end_index=49, token_count=5),
        Chunk(text="This is the third chunk.", start_index=50, end_index=74, token_count=5),
    ]


@pytest.fixture
def mock_embeddings() -> MockEmbeddings:
    """Fixture to create mock embeddings."""
    return MockEmbeddings(dimension=128, available=True)


def test_embeddings_refinery_initialization_with_string() -> None:
    """Test EmbeddingsRefinery initialization with string model name."""
    with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get:
        mock_get.return_value = MockEmbeddings()
        
        refinery = EmbeddingsRefinery("test-model")
        
        assert refinery is not None
        assert isinstance(refinery, EmbeddingsRefinery)
        mock_get.assert_called_once_with("test-model")


def test_embeddings_refinery_initialization_with_string_and_kwargs() -> None:
    """Test EmbeddingsRefinery initialization with string model and kwargs."""
    with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get:
        mock_get.return_value = MockEmbeddings()
        
        kwargs = {"batch_size": 32, "device": "cpu"}
        refinery = EmbeddingsRefinery("test-model", **kwargs)
        
        assert refinery is not None
        mock_get.assert_called_once_with("test-model", **kwargs)


def test_embeddings_refinery_initialization_with_embeddings_instance(mock_embeddings: MockEmbeddings) -> None:
    """Test EmbeddingsRefinery initialization with BaseEmbeddings instance."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    assert refinery is not None
    assert isinstance(refinery, EmbeddingsRefinery)
    assert refinery.embedding_model is mock_embeddings


def test_embeddings_refinery_initialization_with_invalid_type() -> None:
    """Test EmbeddingsRefinery initialization with invalid model type."""
    with pytest.raises(ValueError, match="Model must be a string or a BaseEmbeddings instance"):
        EmbeddingsRefinery(123)  # Invalid type
    
    with pytest.raises(ValueError, match="Model must be a string or a BaseEmbeddings instance"):
        EmbeddingsRefinery(["not", "valid"])  # Invalid type


def test_embeddings_refinery_is_available(mock_embeddings: MockEmbeddings) -> None:
    """Test EmbeddingsRefinery._is_available method."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    assert refinery._is_available() is True
    
    # Test with unavailable embeddings
    unavailable_embeddings = MockEmbeddings(available=False)
    refinery_unavailable = EmbeddingsRefinery(unavailable_embeddings)
    assert refinery_unavailable._is_available() is False


def test_embeddings_refinery_refine_basic(mock_embeddings: MockEmbeddings, sample_chunks: list[Chunk]) -> None:
    """Test basic refine functionality."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    # Ensure chunks don't have embeddings initially
    for chunk in sample_chunks:
        assert not hasattr(chunk, 'embedding')
    
    refined_chunks = refinery.refine(sample_chunks)
    
    # Check that embeddings were added
    assert len(refined_chunks) == 3
    for chunk in refined_chunks:
        assert hasattr(chunk, 'embedding')
        import numpy as np
        assert isinstance(chunk.embedding, np.ndarray)
        assert chunk.embedding.shape == (128,)  # Mock dimension
        assert chunk.embedding.dtype == np.float32


def test_embeddings_refinery_refine_empty_list(mock_embeddings: MockEmbeddings) -> None:
    """Test refine with empty chunk list."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    empty_chunks = []
    refined_chunks = refinery.refine(empty_chunks)
    
    assert refined_chunks == []
    assert len(refined_chunks) == 0


def test_embeddings_refinery_refine_single_chunk(mock_embeddings: MockEmbeddings) -> None:
    """Test refine with single chunk."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    single_chunk = [Chunk(text="Single chunk", start_index=0, end_index=11, token_count=2)]
    refined_chunks = refinery.refine(single_chunk)
    
    assert len(refined_chunks) == 1
    assert hasattr(refined_chunks[0], 'embedding')
    import numpy as np
    assert isinstance(refined_chunks[0].embedding, np.ndarray)
    assert refined_chunks[0].embedding.shape == (128,)


def test_embeddings_refinery_refine_preserves_chunk_properties(mock_embeddings: MockEmbeddings) -> None:
    """Test that refine preserves all original chunk properties."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    chunk = Chunk(text="Test chunk", start_index=10, end_index=19, token_count=2)
    original_text = chunk.text
    original_start = chunk.start_index
    original_end = chunk.end_index
    original_token_count = chunk.token_count
    
    refined_chunks = refinery.refine([chunk])
    refined_chunk = refined_chunks[0]
    
    # Check that original properties are preserved
    assert refined_chunk.text == original_text
    assert refined_chunk.start_index == original_start
    assert refined_chunk.end_index == original_end
    assert refined_chunk.token_count == original_token_count
    
    # Check that embedding was added
    assert hasattr(refined_chunk, 'embedding')


def test_embeddings_refinery_dimension_property(mock_embeddings: MockEmbeddings) -> None:
    """Test dimension property."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    assert refinery.dimension == 128
    
    # Test with different dimension
    different_embeddings = MockEmbeddings(dimension=256)
    refinery_256 = EmbeddingsRefinery(different_embeddings)
    assert refinery_256.dimension == 256


def test_embeddings_refinery_repr(mock_embeddings: MockEmbeddings) -> None:
    """Test __repr__ method."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    repr_str = repr(refinery)
    
    assert "EmbeddingsRefinery" in repr_str
    assert "embedding_model=" in repr_str
    assert isinstance(repr_str, str)


def test_embeddings_refinery_embed_batch_called_correctly(mock_embeddings: MockEmbeddings, sample_chunks: list[Chunk]) -> None:
    """Test that embed_batch is called with correct texts."""
    # Spy on the embed_batch method
    original_embed_batch = mock_embeddings.embed_batch
    mock_embeddings.embed_batch = Mock(side_effect=original_embed_batch)
    
    refinery = EmbeddingsRefinery(mock_embeddings)
    refinery.refine(sample_chunks)
    
    # Check that embed_batch was called once with the correct texts
    mock_embeddings.embed_batch.assert_called_once()
    called_texts = mock_embeddings.embed_batch.call_args[0][0]
    expected_texts = [chunk.text for chunk in sample_chunks]
    assert called_texts == expected_texts


def test_embeddings_refinery_with_different_chunk_types() -> None:
    """Test refinery with different chunk types."""
    from chonkie.types import RecursiveChunk, SentenceChunk
    
    mock_embeddings = MockEmbeddings()
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    # Mix of different chunk types
    chunks = [
        Chunk(text="Regular chunk", start_index=0, end_index=12, token_count=2),
        RecursiveChunk(text="Recursive chunk", start_index=13, end_index=27, token_count=2, level=1),
        SentenceChunk(text="Sentence chunk", start_index=28, end_index=41, token_count=2),
    ]
    
    refined_chunks = refinery.refine(chunks)
    
    # All chunks should get embeddings regardless of type
    assert len(refined_chunks) == 3
    for chunk in refined_chunks:
        assert hasattr(chunk, 'embedding')
        import numpy as np
        assert isinstance(chunk.embedding, np.ndarray)


def test_embeddings_refinery_large_batch(mock_embeddings: MockEmbeddings) -> None:
    """Test refinery with large batch of chunks."""
    # Create 100 chunks
    large_chunks = [
        Chunk(text=f"Chunk number {i}", start_index=i*20, end_index=(i+1)*20-1, token_count=3)
        for i in range(100)
    ]
    
    refinery = EmbeddingsRefinery(mock_embeddings)
    refined_chunks = refinery.refine(large_chunks)
    
    assert len(refined_chunks) == 100
    for chunk in refined_chunks:
        assert hasattr(chunk, 'embedding')
        import numpy as np
        assert isinstance(chunk.embedding, np.ndarray)
        assert chunk.embedding.shape == (128,)


def test_embeddings_refinery_with_empty_text_chunks(mock_embeddings: MockEmbeddings) -> None:
    """Test refinery with chunks containing empty or whitespace text."""
    chunks = [
        Chunk(text="", start_index=0, end_index=0, token_count=0),
        Chunk(text="   ", start_index=1, end_index=3, token_count=0),
        Chunk(text="Normal text", start_index=4, end_index=14, token_count=2),
    ]
    
    refinery = EmbeddingsRefinery(mock_embeddings)
    refined_chunks = refinery.refine(chunks)
    
    assert len(refined_chunks) == 3
    for chunk in refined_chunks:
        assert hasattr(chunk, 'embedding')


def test_embeddings_refinery_modifies_chunks_inplace(mock_embeddings: MockEmbeddings) -> None:
    """Test that refinery modifies chunks in place."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    original_chunks = [
        Chunk(text="Test chunk 1", start_index=0, end_index=11, token_count=2),
        Chunk(text="Test chunk 2", start_index=12, end_index=23, token_count=2),
    ]
    
    # Keep references to original chunks
    chunk1_ref = original_chunks[0]
    chunk2_ref = original_chunks[1]
    
    refined_chunks = refinery.refine(original_chunks)
    
    # Should return the same objects (modified in place)
    assert refined_chunks is original_chunks
    assert refined_chunks[0] is chunk1_ref
    assert refined_chunks[1] is chunk2_ref
    
    # Original chunk objects should now have embeddings
    assert hasattr(chunk1_ref, 'embedding')
    assert hasattr(chunk2_ref, 'embedding')


def test_embeddings_refinery_embedding_consistency(mock_embeddings: MockEmbeddings) -> None:
    """Test that same text produces same embeddings."""
    refinery = EmbeddingsRefinery(mock_embeddings)
    
    # Create chunks with same text
    chunks1 = [Chunk(text="Same text", start_index=0, end_index=8, token_count=2)]
    chunks2 = [Chunk(text="Same text", start_index=10, end_index=18, token_count=2)]
    
    refined1 = refinery.refine(chunks1)
    refined2 = refinery.refine(chunks2)
    
    # Same text should produce same embeddings
    import numpy as np
    assert np.array_equal(refined1[0].embedding, refined2[0].embedding)


@pytest.mark.skipif(
    True,  # Skip by default since it requires actual model
    reason="Requires actual embedding model - enable for integration testing"
)
def test_embeddings_refinery_with_real_model() -> None:
    """Integration test with real embedding model."""
    # This test can be enabled for integration testing with real models
    refinery = EmbeddingsRefinery("sentence-transformers/all-MiniLM-L6-v2")
    
    chunks = [
        Chunk(text="Hello world", start_index=0, end_index=10, token_count=2),
        Chunk(text="How are you", start_index=11, end_index=21, token_count=3),
    ]
    
    if refinery._is_available():
        refined_chunks = refinery.refine(chunks)
        
        for chunk in refined_chunks:
            assert hasattr(chunk, 'embedding')
            import numpy as np
            assert isinstance(chunk.embedding, np.ndarray)
            assert chunk.embedding.shape[0] > 0
            assert chunk.embedding.dtype in [np.float32, np.float64]