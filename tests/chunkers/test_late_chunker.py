"""Test cases for the LateChunker class."""

import numpy as np
import pytest

from chonkie import LateChunker
from chonkie.embeddings import SentenceTransformerEmbeddings
from chonkie.types import LateChunk, RecursiveLevel, RecursiveRules


@pytest.fixture
def embedding_model() -> SentenceTransformerEmbeddings:
    """Return an object of SentenceTransformerEmbeddings type."""
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def sample_text() -> str:
    """Return a sample text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text


def test_late_chunker_init_with_instance(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test the initialization of the LateChunker with an embedding model instance."""
    chunker = LateChunker(
        embedding_model=embedding_model,
        chunk_size=256,
        min_characters_per_chunk=10,
    )
    assert chunker is not None
    assert chunker.embedding_model == embedding_model
    assert chunker.chunk_size == 256
    assert chunker.min_characters_per_chunk == 10
    assert chunker._use_multiprocessing is False  # Should be disabled


def test_late_chunker_init_with_string() -> None:
    """Test the initialization of the LateChunker with a model name string."""
    model_name = "all-MiniLM-L6-v2"
    chunker = LateChunker(
        embedding_model=model_name,
        chunk_size=512,
    )
    assert chunker is not None
    assert isinstance(chunker.embedding_model, SentenceTransformerEmbeddings)
    assert chunker.chunk_size == 512


def test_late_chunker_init_invalid_model() -> None:
    """Test initialization failure with an invalid embedding model type."""
    with pytest.raises(ValueError, match="is not a valid embedding model"):
        LateChunker(embedding_model=123)  # type: ignore


def test_late_chunker_chunk_basic(embedding_model: SentenceTransformerEmbeddings, sample_text: str) -> None:
    """Test basic chunking functionality."""
    chunk_size = 512
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=chunk_size)
    chunks = chunker.chunk(sample_text)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, LateChunk) for chunk in chunks)

    # Check attributes of each chunk
    for chunk in chunks:
        assert isinstance(chunk.text, str)
        assert len(chunk.text) > 0
        assert isinstance(chunk.start_index, int)
        assert chunk.start_index >= 0
        assert isinstance(chunk.end_index, int)
        assert chunk.end_index > chunk.start_index
        assert isinstance(chunk.token_count, int)
        assert chunk.token_count > 0
        # Although recursive chunker aims for chunk_size, late chunker's token_count
        # can be slightly different due to adjustment. Check positivity.
        assert isinstance(chunk.embedding, np.ndarray)
        assert chunk.embedding.shape == (embedding_model.dimension,)

    # Rough check: total length should approximate original text length
    assert abs(sum(len(c.text) for c in chunks) - len(sample_text)) < len(chunks) * 2


def test_late_chunker_chunk_empty_text(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test chunking empty text."""
    chunker = LateChunker(embedding_model=embedding_model)
    chunks = chunker.chunk("")
    assert chunks == []


def test_late_chunker_chunk_short_text(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test chunking text shorter than chunk size."""
    text = "This is a short text, definitely shorter than the chunk size."
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=512)
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, LateChunk)
    assert chunk.text == text
    assert chunk.start_index == 0
    assert chunk.end_index == len(text)
    assert chunk.token_count > 0
    assert isinstance(chunk.embedding, np.ndarray)
    assert chunk.embedding.shape == (embedding_model.dimension,)


def verify_chunk_indices(chunks: list[LateChunk], original_text: str) -> None:
    """Verify that chunk indices correctly map to the original text."""
    reconstructed_text = ""
    for i, chunk in enumerate(chunks):
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        assert chunk.text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk.text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )
        reconstructed_text += chunk.text

    # Allow minor discrepancies at the very end if needed, but usually should match
    assert reconstructed_text == original_text, (
        "Reconstructed text does not match original"
    )


def test_late_chunker_indices(embedding_model: SentenceTransformerEmbeddings, sample_text: str) -> None:
    """Test that LateChunker correctly maps chunk indices to the original text."""
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=256)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_late_chunk_repr(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test the string representation of LateChunk."""
    text = "This is a short text, definitely shorter than the chunk size."
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=50)
    chunks = chunker.chunk(text)
    if not chunks:
        pytest.skip("No chunks generated for repr test")

    chunk = chunks[0]
    representation = repr(chunk)
    assert f"text={chunk.text}" in representation
    assert f"start_index={chunk.start_index}" in representation
    assert f"end_index={chunk.end_index}" in representation
    assert f"token_count={chunk.token_count}" in representation
    assert "embedding=" in representation  # Check for embedding presence
    assert isinstance(chunk.embedding, np.ndarray)


def test_late_chunker_custom_rules(embedding_model: SentenceTransformerEmbeddings, sample_text: str) -> None:
    """Test that LateChunker works even with custom rules."""
    custom_rules = RecursiveRules([RecursiveLevel([".", "!", "?", "\n"])])
    chunker = LateChunker(
        embedding_model=embedding_model, chunk_size=256, rules=custom_rules
    )
    chunks = chunker.chunk(sample_text)

    # Check if the chunks are generated correctly
    assert len(chunks) > 0, "No chunks generated"
    assert chunks[0].text[-1] in custom_rules.levels[0].delimiters
    assert all([chunk.token_count for chunk in chunks]) < 256

def test_late_chunker_from_recipe_default() -> None:
    """Test that LateChunker.from_recipe works with default parameters."""
    chunker = LateChunker.from_recipe()

    assert chunker is not None
    assert isinstance(chunker.rules, RecursiveRules)

def test_late_chunker_from_recipe_custom_params(embedding_model: SentenceTransformerEmbeddings) -> None:
    """Test that LateChunker.from_recipe works with custom parameters."""
    chunker = LateChunker.from_recipe(
        name="default",
        lang="en",
        chunk_size=256,
        min_characters_per_chunk=10,
        embedding_model=embedding_model,
    )

    assert chunker is not None
    assert isinstance(chunker.rules, RecursiveRules)
    assert chunker.chunk_size == 256
    assert chunker.min_characters_per_chunk == 10

def test_late_chunker_from_recipe_nonexistent() -> None:
    """Test that LateChunker.from_recipe raises an error if the recipe does not exist."""
    with pytest.raises(ValueError):
        LateChunker.from_recipe(name="invalid")

    with pytest.raises(ValueError):
        LateChunker.from_recipe(name="default", lang="invalid")

if __name__ == "__main__":
    pytest.main()