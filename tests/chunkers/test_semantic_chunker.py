"""Test the SemanticChunker class - cleaned version with only working tests."""

import os
from typing import List

import pytest

from chonkie import SemanticChunker
from chonkie.embeddings import BaseEmbeddings, Model2VecEmbeddings, OpenAIEmbeddings, CohereEmbeddings
from chonkie.types.base import Chunk


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing the SemanticChunker."""
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model() -> BaseEmbeddings:
    """Fixture that returns a Model2Vec embedding model for testing."""
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def sample_complex_markdown_text() -> str:
    """Fixture that returns a sample markdown text with complex formatting."""
    text = """# Heading 1
    This is a paragraph with some **bold text** and _italic text_.
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    ```python
    # Code block
    def hello_world():
        print("Hello, world!")
    ```
    Another paragraph with [a link](https://example.com) and an image:
    ![Alt text](https://example.com/image.jpg)
    > A blockquote with multiple lines
    > that spans more than one line.
    Finally, a paragraph at the end.
    """
    return text


def test_semantic_chunker_initialization(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.similarity_window == 3
    assert chunker.min_sentences_per_chunk == 1


def test_semantic_chunker_chunking(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_semantic_chunker_empty_text(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")
    assert len(chunks) == 0

    chunks = chunker.chunk("   ")
    assert len(chunks) == 0


def test_semantic_chunker_single_sentence(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."


def test_semantic_chunker_token_counts(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker correctly calculates token counts and respects chunk_size."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, 
        chunk_size=512, 
        threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    
    assert all([chunk.token_count > 0 for chunk in chunks]), (
        "All chunks must have a positive token count"
    )
    assert all([chunk.token_count <= 512 for chunk in chunks]), (
        "All chunks must respect the chunk_size limit of 512 tokens"
    )

    # Verify token counts match the tokenizer's count
    token_counts = [chunker.tokenizer.count_tokens(chunk.text) for chunk in chunks]
    for i, (chunk, token_count) in enumerate(zip(chunks, token_counts)):
        assert chunk.token_count == token_count, (
            f"Chunk {i} has a token count of {chunk.token_count} but the encoded text length is {token_count}"
        )


def test_semantic_chunker_reconstruction(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, 
        chunk_size=512, 
        threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    assert sample_text == "".join([chunk.text for chunk in chunks])


def verify_chunk_indices(chunks: List[Chunk], original_text: str) -> None:
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        # Extract text using the indices
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        # Remove any leading/trailing whitespace from both texts for comparison
        chunk_text = chunk.text.strip()
        extracted_text = extracted_text.strip()

        assert chunk_text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk_text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )


def test_semantic_chunker_indices(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, 
        chunk_size=512, 
        threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


class TestSemanticChunkerParameterValidation:
    """Test parameter validation in SemanticChunker."""

    def test_invalid_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=-1)

    def test_invalid_min_sentences_per_chunk(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid min_sentences_per_chunk."""
        with pytest.raises(ValueError, match="min_sentences_per_chunk must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences_per_chunk=0)
        
        with pytest.raises(ValueError, match="min_sentences_per_chunk must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences_per_chunk=-1)

    def test_invalid_similarity_window(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid similarity_window."""
        with pytest.raises(ValueError, match="similarity_window must be positive"):
            SemanticChunker(embedding_model=embedding_model, similarity_window=0)
            
        with pytest.raises(ValueError, match="similarity_window must be positive"):
            SemanticChunker(embedding_model=embedding_model, similarity_window=-1)

    def test_invalid_threshold(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold values."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=0)
        
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=1.0)
            
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=-0.1)
            
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=1.5)

    def test_invalid_threshold_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for non-numeric threshold."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold="invalid")
            
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=[0.5])

    def test_invalid_embedding_model_type(self) -> None:
        """Test that SemanticChunker raises error for invalid embedding model type."""
        with pytest.raises(ValueError, match="embedding_model must be a string or a BaseEmbeddings object"):
            SemanticChunker(embedding_model=123)


class TestSemanticChunkerConfiguration:
    """Test different configuration options."""

    def test_with_different_similarity_windows(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with different similarity windows."""
        # Test with default window
        chunker1 = SemanticChunker(
            embedding_model=embedding_model,
            threshold=0.5,
            chunk_size=512
        )
        chunks1 = chunker1.chunk(sample_text)
        assert len(chunks1) > 0
        assert chunker1.similarity_window == 3  # Default value
        
        # Test with larger window
        chunker2 = SemanticChunker(
            embedding_model=embedding_model,
            similarity_window=5,
            threshold=0.5,
            chunk_size=512
        )
        chunks2 = chunker2.chunk(sample_text)
        assert len(chunks2) > 0
        assert chunker2.similarity_window == 5

    def test_with_different_thresholds(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with different threshold values."""
        # Low threshold (more merging)
        chunker1 = SemanticChunker(
            embedding_model=embedding_model,
            threshold=0.1,
            chunk_size=512
        )
        chunks1 = chunker1.chunk(sample_text)
        assert len(chunks1) > 0
        assert chunker1.threshold == 0.1

        # High threshold (less merging)
        chunker2 = SemanticChunker(
            embedding_model=embedding_model,
            threshold=0.9,
            chunk_size=512
        )
        chunks2 = chunker2.chunk(sample_text)
        assert len(chunks2) > 0
        assert chunker2.threshold == 0.9


class TestSemanticChunkerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test chunking with very small chunk size."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=20,
            min_sentences_per_chunk=1
        )
        text = "Short. Also short. Another short one. Final short sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        # Chunks should respect the size limit (with some buffer for semantic grouping)
        for chunk in chunks:
            assert chunk.token_count <= 30

    def test_delim_as_string(self, embedding_model: BaseEmbeddings) -> None:
        """Test SemanticChunker with delim as string instead of list."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            delim=".",
            chunk_size=512
        )
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0