"""Tests for the semantic chunker with peak detection."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import types

from chonkie.chunker.semantic import SemanticChunker
from chonkie.embeddings.base import BaseEmbeddings

@pytest.fixture
def embedding_model() -> BaseEmbeddings:
    """Create a mock embedding model for testing."""
    model = MagicMock(spec=BaseEmbeddings)
    
    # Create a sequence of embeddings that will produce clear peaks
    # Each embedding is a 384-dimensional vector with varying patterns
    def mock_embed_batch(texts: list[str]) -> list[np.ndarray]:
        embeddings = []
        for i, text in enumerate(texts):
            base = np.zeros(384)
            peak_value = 0.8 if i % 3 == 0 else 0.2
            base[0:10] = peak_value
            noise = np.random.normal(0, 0.1, 384)
            embeddings.append(base + noise)
        return embeddings
    model.embed_batch = mock_embed_batch
    model.count_tokens = lambda text: 10
    model.similarity = lambda u, v: float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return model

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with count_tokens_batch method."""
    tokenizer = MagicMock()
    tokenizer.count_tokens_batch = lambda texts: [10 for _ in texts]
    return tokenizer

@pytest.fixture
def sample_text() -> str:
    """Create a sample text for testing."""
    return "This is the first sentence. This is the second sentence. This is the third sentence. " \
           "This is the fourth sentence. This is the fifth sentence. This is the sixth sentence. " \
           "This is the seventh sentence. This is the eighth sentence. This is the ninth sentence."

def test_semantic_chunker_peak_detection_initialization(
    embedding_model: BaseEmbeddings,
    mock_tokenizer
) -> None:
    """Test that the semantic chunker initializes correctly with peak detection."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=True,
        peak_window_length=5,
        peak_polyorder=2
    )
    chunker.tokenizer = mock_tokenizer
    assert chunker.use_peak_detection
    assert chunker.peak_detector.window_length == 5
    assert chunker.peak_detector.polyorder == 2
    assert chunker.peak_detector is not None

def test_semantic_chunker_peak_detection_chunking(
    embedding_model: BaseEmbeddings,
    sample_text: str,
    mock_tokenizer
) -> None:
    """Test that the semantic chunker chunks text correctly using peak detection."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=True,
        peak_window_length=5,
        peak_polyorder=2
    )
    chunker.tokenizer = mock_tokenizer
    chunks = chunker.chunk(sample_text)
    
    # Verify that we got some chunks
    assert len(chunks) > 0
    
    # Verify that each chunk has the required attributes
    for chunk in chunks:
        assert hasattr(chunk, 'text')
        assert hasattr(chunk, 'start_index')
        assert hasattr(chunk, 'end_index')
        assert hasattr(chunk, 'token_count')
        assert hasattr(chunk, 'sentences')
        
        # Verify that the chunk's text is a substring of the original text
        assert chunk.text in sample_text
        
        # Verify that the chunk's sentences are properly ordered
        for i in range(len(chunk.sentences) - 1):
            assert chunk.sentences[i].end_index <= chunk.sentences[i + 1].start_index

def test_semantic_chunker_peak_detection_vs_threshold(
    embedding_model: BaseEmbeddings,
    sample_text: str,
    mock_tokenizer
) -> None:
    """Test that peak detection produces different results than threshold-based chunking."""
    # Create two chunkers with different methods
    peak_chunker = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=True,
        peak_window_length=5,
        peak_polyorder=2,
        chunk_size=512
    )
    peak_chunker.tokenizer = mock_tokenizer
    threshold_chunker = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=False,
        threshold=0.5,
        chunk_size=512
    )
    threshold_chunker.tokenizer = mock_tokenizer
    
    # Chunk the text with both methods
    peak_chunks = peak_chunker.chunk(sample_text)
    threshold_chunks = threshold_chunker.chunk(sample_text)
    
    # The results should be different
    assert len(peak_chunks) != len(threshold_chunks)
    
    # Verify that the chunks are different
    peak_texts = [chunk.text for chunk in peak_chunks]
    threshold_texts = [chunk.text for chunk in threshold_chunks]
    assert peak_texts != threshold_texts

def test_semantic_chunker_peak_detection_parameters(
    embedding_model: BaseEmbeddings,
    sample_text: str,
    mock_tokenizer
) -> None:
    """Test that different peak detection parameters produce different results."""
    # Create chunkers with different parameters
    chunker1 = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=True,
        peak_window_length=5,
        peak_polyorder=2,
        chunk_size=512
    )
    chunker1.tokenizer = mock_tokenizer
    chunker2 = SemanticChunker(
        embedding_model=embedding_model,
        use_peak_detection=True,
        peak_window_length=7,
        peak_polyorder=3,
        chunk_size=512
    )
    chunker2.tokenizer = mock_tokenizer
    
    # Chunk the text with both parameter sets
    chunks1 = chunker1.chunk(sample_text)
    chunks2 = chunker2.chunk(sample_text)
    
    # The results should be different
    assert len(chunks1) != len(chunks2)
    
    # Verify that the chunks are different
    texts1 = [chunk.text for chunk in chunks1]
    texts2 = [chunk.text for chunk in chunks2]
    assert texts1 != texts2 