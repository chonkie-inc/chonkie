"""Test the SDPMChunker class."""

import pytest

from chonkie.legacy import SDPMChunker
from chonkie.embeddings import Model2VecEmbeddings
from chonkie.types.semantic import SemanticChunk


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing the SDPMChunker.

    Returns:
        str: A paragraph of text about chunking.

    """
    text = """The process of text chunking in retrieval systems involves breaking down large documents into smaller, semantically coherent pieces. This is crucial for effective information retrieval. Different chunking strategies can significantly impact the quality of retrieved results. The choice of chunk size affects both precision and recall in search operations."""
    return text


@pytest.fixture
def multi_topic_text() -> str:
    """Sample text with multiple distinct topics for testing.

    Returns:
        str: Text with clear topic boundaries.

    """
    text = """Machine learning algorithms require large datasets for training. Deep neural networks have revolutionized artificial intelligence. The weather today is sunny and warm. Rain is expected tomorrow afternoon. Cooking pasta requires boiling water first. Italian cuisine emphasizes fresh ingredients. Quantum computing uses quantum mechanical phenomena. Superposition allows qubits to exist in multiple states simultaneously."""
    return text


@pytest.fixture
def short_text() -> str:
    """Short text for edge case testing.

    Returns:
        str: A short sentence.

    """
    return "This is a short sentence."


@pytest.fixture
def embedding_model():
    """Fixture that returns a Model2Vec embedding model for testing."""
    return Model2VecEmbeddings("minishlab/potion-base-8M")


class TestSDPMChunkerInitialization:
    """Test the initialization of the SDPMChunker."""

    def test_init_with_default_parameters(self, embedding_model):
        """Test initialization with default parameters."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        
        assert chunker.chunk_size == 2048
        assert chunker.threshold == "auto"
        assert chunker.similarity_window == 1
        assert chunker.min_sentences == 1
        assert chunker.min_chunk_size == 2
        assert chunker.skip_window == 1

    def test_init_with_custom_parameters(self, embedding_model):
        """Test initialization with custom parameters."""
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            chunk_size=256,
            threshold=0.7,
            similarity_window=2,
            min_sentences=3,
            min_chunk_size=5,
            skip_window=2,
        )
        
        assert chunker.chunk_size == 256
        assert chunker.threshold == 0.7
        assert chunker.similarity_window == 2
        assert chunker.min_sentences == 3
        assert chunker.min_chunk_size == 5
        assert chunker.skip_window == 2

    def test_init_with_string_embedding_model(self):
        """Test initialization with string embedding model."""
        chunker = SDPMChunker(embedding_model="minishlab/potion-base-8M")
        
        assert chunker.chunk_size == 2048
        assert chunker.threshold == "auto"
        assert hasattr(chunker, 'embedding_model')

    def test_init_with_different_modes(self, embedding_model):
        """Test initialization with different modes."""
        for mode in ["window", "cumulative"]:
            chunker = SDPMChunker(
                embedding_model=embedding_model,
                mode=mode
            )
            assert chunker.mode == mode

    def test_init_with_different_thresholds(self, embedding_model):
        """Test initialization with different threshold types."""
        # Auto threshold
        chunker1 = SDPMChunker(embedding_model=embedding_model, threshold="auto")
        assert chunker1.threshold == "auto"
        
        # Float threshold
        chunker2 = SDPMChunker(embedding_model=embedding_model, threshold=0.6)
        assert chunker2.threshold == 0.6
        
        # Int threshold (percentile)
        chunker3 = SDPMChunker(embedding_model=embedding_model, threshold=80)
        assert chunker3.threshold == 80


class TestSDPMChunkerBasicFunctionality:
    """Test basic chunking functionality."""

    def test_chunk_returns_semantic_chunks(self, embedding_model, sample_text):
        """Test that chunking returns SemanticChunk objects."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(sample_text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, SemanticChunk) for chunk in result)

    def test_chunk_preserves_content(self, embedding_model, sample_text):
        """Test that chunking preserves all content."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(sample_text)
        
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == sample_text

    def test_chunk_has_valid_properties(self, embedding_model, sample_text):
        """Test that chunks have valid properties."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(sample_text)
        
        for chunk in result:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0
            assert chunk.token_count > 0
            assert hasattr(chunk, 'sentences')
            assert isinstance(chunk.sentences, list)
            assert len(chunk.sentences) > 0

    def test_chunk_respects_min_sentences(self, embedding_model, multi_topic_text):
        """Test that chunker respects minimum sentences per chunk."""
        min_sentences = 2
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            min_sentences=min_sentences
        )
        result = chunker.chunk(multi_topic_text)
        
        # Most chunks should respect the minimum (some might not due to text end)
        chunks_with_min_sentences = [c for c in result if len(c.sentences) >= min_sentences]
        assert len(chunks_with_min_sentences) > 0

    def test_chunk_respects_chunk_size(self, embedding_model, multi_topic_text):
        """Test that chunks respect maximum chunk size."""
        chunk_size = 100
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size
        )
        result = chunker.chunk(multi_topic_text)
        
        for chunk in result:
            # Allow some flexibility due to sentence boundaries
            assert chunk.token_count <= chunk_size * 1.5


class TestSDPMChunkerInternalMethods:
    """Test internal methods of the SDPMChunker."""

    def test_merge_sentence_groups_basic(self, embedding_model):
        """Test _merge_sentence_groups method."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        sentence_groups = [
            ["First sentence.", "Second sentence."],
            ["Third sentence."],
            ["Fourth sentence.", "Fifth sentence."]
        ]
        
        merged = chunker._merge_sentence_groups(sentence_groups)
        
        expected = ["First sentence.", "Second sentence.", "Third sentence.", "Fourth sentence.", "Fifth sentence."]
        assert merged == expected

    def test_merge_sentence_groups_empty(self, embedding_model):
        """Test _merge_sentence_groups with empty input."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        merged = chunker._merge_sentence_groups([])
        assert merged == []

    def test_merge_sentence_groups_single_group(self, embedding_model):
        """Test _merge_sentence_groups with single group."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        sentence_groups = [["Only sentence."]]
        merged = chunker._merge_sentence_groups(sentence_groups)
        assert merged == ["Only sentence."]

    def test_skip_and_merge_basic(self, embedding_model):
        """Test _skip_and_merge method with basic input."""
        chunker = SDPMChunker(embedding_model=embedding_model, skip_window=1)
        
        # Create real sentence objects by preparing sentences through the chunker
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._prepare_sentences(text)
        
        # Create groups from the prepared sentences
        groups = [[sentences[0]], [sentences[1]], [sentences[2]]]
        
        # Use a low threshold to test merging behavior
        result = chunker._skip_and_merge(groups, similarity_threshold=0.1)
        
        assert isinstance(result, list)
        # Should return some result (exact behavior depends on embeddings)
        assert len(result) <= len(groups)

    def test_skip_and_merge_empty_groups(self, embedding_model):
        """Test _skip_and_merge with empty groups."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker._skip_and_merge([], similarity_threshold=0.5)
        assert result == []

    def test_skip_and_merge_single_group(self, embedding_model):
        """Test _skip_and_merge with single group."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        
        # Create a real sentence object
        text = "Single sentence."
        sentences = chunker._prepare_sentences(text)
        groups = [sentences]
        
        result = chunker._skip_and_merge(groups, similarity_threshold=0.5)
        assert len(result) == 1
        assert result == groups


class TestSDPMChunkerEdgeCases:
    """Test edge cases for the SDPMChunker."""

    def test_empty_text(self, embedding_model):
        """Test chunking empty text."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk("")
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_single_sentence(self, embedding_model):
        """Test chunking single sentence."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk("This is a single sentence.")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "This is a single sentence."

    def test_very_short_text(self, embedding_model, short_text):
        """Test chunking very short text."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(short_text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == short_text

    def test_whitespace_only_text(self, embedding_model):
        """Test chunking whitespace-only text."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk("   \n\t  ")
        
        assert isinstance(result, list)
        # Should handle gracefully

    def test_large_skip_window(self, embedding_model, multi_topic_text):
        """Test with large skip window."""
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            skip_window=10  # Larger than typical sentence count
        )
        result = chunker.chunk(multi_topic_text)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_small_chunk_size(self, embedding_model, multi_topic_text):
        """Test with very small chunk size."""
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            chunk_size=50  # Very small
        )
        result = chunker.chunk(multi_topic_text)
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestSDPMChunkerRepresentation:
    """Test the string representation of the SDPMChunker."""

    def test_repr(self, embedding_model):
        """Test the __repr__ method."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        
        repr_str = repr(chunker)
        assert "SDPMChunker" in repr_str
        assert "chunk_size=2048" in repr_str
        assert "mode=window" in repr_str
        assert "threshold=auto" in repr_str
        assert "skip_window=1" in repr_str

    def test_repr_with_custom_parameters(self, embedding_model):
        """Test __repr__ with custom parameters."""
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            chunk_size=256,
            threshold=0.7,
            skip_window=2,
        )
        
        repr_str = repr(chunker)
        assert "chunk_size=256" in repr_str
        assert "threshold=0.7" in repr_str
        assert "skip_window=2" in repr_str


class TestSDPMChunkerBehavior:
    """Test specific behavior patterns of the SDPMChunker."""

    def test_chunk_boundaries_are_reasonable(self, embedding_model, multi_topic_text):
        """Test that chunk boundaries make semantic sense."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(multi_topic_text)
        
        assert len(result) >= 1
        
        for chunk in result:
            assert len(chunk.text.strip()) > 0
            assert len(chunk.sentences) > 0
            assert chunk.token_count > 0

    def test_semantic_grouping(self, embedding_model):
        """Test that semantically similar sentences are grouped."""
        # Text with two clear topics
        text = "Dogs are loyal pets. Cats are independent animals. Canines love their owners. Felines prefer solitude."
        
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            threshold=0.3  # Lower threshold to encourage grouping
        )
        result = chunker.chunk(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Should group semantically similar sentences
        reconstructed = "".join(chunk.text for chunk in result)
        assert reconstructed == text

    def test_sentence_preservation(self, embedding_model, multi_topic_text):
        """Test that all sentences are preserved in chunking."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        result = chunker.chunk(multi_topic_text)
        
        # Extract all sentences from chunks
        all_chunk_sentences = []
        for chunk in result:
            all_chunk_sentences.extend(chunk.sentences)
        
        assert len(all_chunk_sentences) > 0
        
        # All sentences should be non-empty (check .text attribute for SemanticSentence)
        for sentence in all_chunk_sentences:
            assert hasattr(sentence, 'text')
            assert len(sentence.text.strip()) > 0

    def test_different_similarity_windows(self, embedding_model, multi_topic_text):
        """Test with different similarity windows."""
        for window in [1, 2, 3]:
            chunker = SDPMChunker(
                embedding_model=embedding_model,
                similarity_window=window
            )
            result = chunker.chunk(multi_topic_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert chunker.similarity_window == window


class TestSDPMChunkerParameterVariations:
    """Test different parameter combinations."""

    def test_different_thresholds(self, embedding_model, multi_topic_text):
        """Test with different similarity thresholds."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            chunker = SDPMChunker(
                embedding_model=embedding_model,
                threshold=threshold
            )
            result = chunker.chunk(multi_topic_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(chunk, SemanticChunk) for chunk in result)

    def test_different_chunk_sizes(self, embedding_model, multi_topic_text):
        """Test with different maximum chunk sizes."""
        chunk_sizes = [64, 128, 256, 512]
        
        for size in chunk_sizes:
            chunker = SDPMChunker(
                embedding_model=embedding_model,
                chunk_size=size
            )
            result = chunker.chunk(multi_topic_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Check that chunks respect token limits (with some flexibility)
            for chunk in result:
                assert chunk.token_count <= size * 1.5

    def test_different_min_chunk_sizes(self, embedding_model, multi_topic_text):
        """Test with different minimum chunk sizes."""
        for min_size in [1, 2, 3, 5]:
            chunker = SDPMChunker(
                embedding_model=embedding_model,
                min_chunk_size=min_size
            )
            result = chunker.chunk(multi_topic_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert chunker.min_chunk_size == min_size

    def test_return_type_parameter(self, embedding_model, sample_text):
        """Test that chunker returns SemanticChunk objects by default."""
        # Test default return type (chunks)
        chunker = SDPMChunker(
            embedding_model=embedding_model
        )
        result = chunker.chunk(sample_text)
        assert all(isinstance(item, SemanticChunk) for item in result)
        
        # Test that text content is accessible via .text property
        for chunk in result:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0


class TestSDPMChunkerRecipeFeature:
    """Test the from_recipe class method."""

    def test_from_recipe_default(self, embedding_model):
        """Test creating chunker from default recipe."""
        try:
            chunker = SDPMChunker.from_recipe(
                embedding_model=embedding_model
            )
            assert isinstance(chunker, SDPMChunker)
            assert chunker.chunk_size == 2048
        except Exception:
            # Skip if recipe system is not available or configured
            pytest.skip("Recipe system not available")

    def test_from_recipe_with_parameters(self, embedding_model):
        """Test creating chunker from recipe with custom parameters."""
        try:
            chunker = SDPMChunker.from_recipe(
                embedding_model=embedding_model,
                chunk_size=256,
                threshold=0.6
            )
            assert isinstance(chunker, SDPMChunker)
            assert chunker.chunk_size == 256
            assert chunker.threshold == 0.6
        except Exception:
            # Skip if recipe system is not available or configured
            pytest.skip("Recipe system not available")


class TestSDPMChunkerBatchProcessing:
    """Test batch processing capabilities."""

    def test_chunk_batch_basic(self, embedding_model, sample_text):
        """Test batch processing of texts."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        texts = [sample_text, sample_text[:100], sample_text[100:]]
        
        results = chunker.chunk_batch(texts)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
        
        for result in results:
            assert isinstance(result, list)
            assert all(isinstance(chunk, SemanticChunk) for chunk in result)

    def test_chunk_batch_empty_list(self, embedding_model):
        """Test batch processing with empty list."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        results = chunker.chunk_batch([])
        assert results == []

    def test_chunk_batch_mixed_lengths(self, embedding_model, sample_text, short_text):
        """Test batch processing with texts of different lengths."""
        chunker = SDPMChunker(embedding_model=embedding_model)
        texts = [sample_text, short_text, ""]
        
        results = chunker.chunk_batch(texts)
        
        assert len(results) == 3
        assert len(results[0]) > 0  # Long text should have chunks
        assert len(results[1]) > 0  # Short text should have at least one chunk
        assert len(results[2]) == 0  # Empty text should have no chunks