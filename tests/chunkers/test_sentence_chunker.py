"""Test cases for the SentenceChunker."""

from __future__ import annotations

from typing import List

import pytest
from tokenizers import Tokenizer

from chonkie import Chunk, SentenceChunker


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Return a tokenizer instance."""
    return Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def sample_text() -> str:
    """Return a sample text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text


@pytest.fixture
def sample_complex_markdown_text() -> str:
    """Return a sample complex markdown text."""
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


def test_sentence_chunker_initialization(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker can be initialized with a tokenizer."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
        approximate=False,
        delim=[".", "!", "?", "\n"],
        include_delim="prev",
    )

    assert chunker is not None
    assert chunker.tokenizer.tokenizer == tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128
    assert chunker.min_sentences_per_chunk == 1
    assert chunker.approximate == False
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.return_type == "chunks"


def test_sentence_chunker_chunking(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that the SentenceChunker can chunk a sample text into sentences."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
        approximate=False,
        delim=[".", "!", "?", "\n"],
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_sentence_chunker_empty_text(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker can handle empty text input."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
    )
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_sentence_chunker_single_sentence(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker can handle text with a single sentence."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."


def test_sentence_chunker_single_chunk_text(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker can handle text that fits within a single chunk."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
    )
    chunks = chunker.chunk("Hello, how are you? I am doing well.")

    assert len(chunks) == 1
    assert chunks[0].text == "Hello, how are you? I am doing well."


def test_sentence_chunker_repr(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker has a string representation."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
    )

    assert repr(chunker) == (
        f"SentenceChunker(tokenizer={chunker.tokenizer}, "
        f"chunk_size={chunker.chunk_size}, "
        f"chunk_overlap={chunker.chunk_overlap}, "
        f"min_sentences_per_chunk={chunker.min_sentences_per_chunk}, "
        f"min_characters_per_sentence={chunker.min_characters_per_sentence}, "
        f"approximate={chunker.approximate}, "
        f"delim={chunker.delim}, "
        f"include_delim={chunker.include_delim}, "
        f"return_type={chunker.return_type})"
    )


def test_sentence_chunker_overlap(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that the SentenceChunker creates overlapping chunks correctly."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, 
        chunk_size=512, 
        chunk_overlap=128,
    )
    chunks = chunker.chunk(sample_text)

    for i in range(1, len(chunks)):
        assert chunks[i].start_index < chunks[i - 1].end_index


def test_sentence_chunker_min_sentences(tokenizer: Tokenizer) -> None:
    """Test that the SentenceChunker respects minimum sentences per chunk."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=512,
        chunk_overlap=128,
        min_sentences_per_chunk=2,
    )
    chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")

    assert len(chunks) > 0
    for chunk in chunks:
        # Count sentences by splitting on periods
        sentence_count = len([s for s in chunk.text.split(".") if s.strip()])
        assert (
            sentence_count >= 2 or sentence_count == 1
        )  # Last chunk might have fewer sentences


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


def test_sentence_chunker_indices(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, chunk_size=512, chunk_overlap=128
    )
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_sentence_chunker_indices_complex_md(tokenizer: Tokenizer, sample_complex_markdown_text: str) -> None:
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, chunk_size=512, chunk_overlap=128
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


def test_sentence_chunker_token_counts(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that the SentenceChunker correctly calculates token counts."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer, chunk_size=512, chunk_overlap=128
    )
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), (
        "All chunks must have a positive token count"
    )
    assert all([chunk.token_count <= 512 for chunk in chunks]), (
        "All chunks must have a token count less than or equal to 512"
    )

    token_counts = [len(tokenizer.encode(chunk.text)) for chunk in chunks]
    assert all([
        chunk.token_count == token_count
        for chunk, token_count in zip(chunks, token_counts)
    ]), "All chunks must have a token count equal to the length of the encoded text"


def test_sentence_chunker_return_type(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that SentenceChunker's return type is correctly set."""
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=512,
        chunk_overlap=128,
        return_type="texts",
    )
    chunks = chunker.chunk(sample_text)
    assert all([type(chunk) is str for chunk in chunks])
    assert all([len(tokenizer.encode(chunk)) <= 512 for chunk in chunks])


def test_sentence_chunker_min_sentences_per_chunk(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that SentenceChunker respects minimum sentences per chunk."""
    # Test that the minimum sentences per chunk is respected, giving a warning otherwise!
    sample_text = "This is a test."
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=512,
        chunk_overlap=128,
        min_sentences_per_chunk=2,
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1
    assert chunks[0].text == "This is a test."
    assert chunks[0].token_count == len(tokenizer.encode(sample_text))


def test_sentence_chunker_min_characters_per_sentence(tokenizer: Tokenizer, sample_text: str) -> None:
    """Test that SentenceChunker respects minimum characters per sentence and when less than min_characters_per_sentence, it is merged with the next sentence."""
    sample_text = "Hello!"
    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=512,
        min_characters_per_sentence=20,
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello!"

def test_sentence_chunker_from_recipe_default() -> None:
    """Test that SentenceChunker.from_recipe works with default parameters."""
    chunker = SentenceChunker.from_recipe()

    assert chunker is not None
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"

def test_sentence_chunker_from_recipe_custom_params() -> None:
    """Test that SentenceChunker.from_recipe works with custom parameters."""
    chunker = SentenceChunker.from_recipe(
        name="default",
        lang="en",
        chunk_size=256,
        min_characters_per_sentence=32,
        return_type="texts"
    )

    assert chunker is not None
    assert chunker.chunk_size == 256
    assert chunker.min_characters_per_sentence == 32
    assert chunker.return_type == "texts"
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"

def test_sentence_chunker_from_recipe_custom_lang() -> None:
    """Test that SentenceChunker.from_recipe works with custom language."""
    chunker = SentenceChunker.from_recipe(
        name="default",
        lang="hi",
        chunk_size=512,
        min_characters_per_sentence=12,
        return_type="chunks",
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_sentence == 12
    assert chunker.return_type == "chunks"
    assert chunker.delim is not None
    assert chunker.include_delim is not None

def test_sentence_chunker_from_recipe_nonexistent() -> None:
    """Test that SentenceChunker.from_recipe raises an error if the recipe does not exist."""
    with pytest.raises(ValueError):
        SentenceChunker.from_recipe(name="invalid")
    
    with pytest.raises(ValueError):
        SentenceChunker.from_recipe(name="default", lang="invalid")

if __name__ == "__main__":
    pytest.main()
