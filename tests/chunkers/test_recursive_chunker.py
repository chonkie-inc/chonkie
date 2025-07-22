"""Tests for the RecursiveChunker class.

This module contains test cases for the RecursiveChunker class, which implements
a recursive chunking strategy based on configurable rules. The tests verify:

- Initialization with different rule configurations
- Basic chunking functionality with simple and complex texts
- Proper handling of recursive levels
- Token count accuracy
- Index mapping and text reconstruction
- Edge cases and error conditions
"""

import pytest

from chonkie import (
    Chunk,
    RecursiveChunker,
    RecursiveLevel,
    RecursiveRules,
)


@pytest.fixture
def sample_text() -> str:
    """Return a sample text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text


@pytest.fixture
def default_rules() -> RecursiveRules:
    """Return a default set of rules."""
    return RecursiveRules()


@pytest.fixture
def paragraph_rules() -> RecursiveRules:
    """Return a paragraph set of rules."""
    paragraph_level = RecursiveLevel(
        delimiters=["\n\n", "\r\n", "\n", "\r", "\t"], whitespace=False
    )
    return RecursiveRules(levels=[paragraph_level])


@pytest.fixture
def sentence_rules() -> RecursiveRules:
    """Return a sentence set of rules."""
    sentence_level = RecursiveLevel(delimiters=[".", "?", "!"], whitespace=False)
    return RecursiveRules(levels=[sentence_level])


@pytest.fixture
def subsentence_rules() -> RecursiveRules:
    """Return a subsentence set of rules."""
    subsentence_level = RecursiveLevel(
        delimiters=[
            ",",
            ";",
            ":",
            "-",
            "/",
            "|",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "<",
            ">",
        ],
        whitespace=False,
    )
    return RecursiveRules(levels=[subsentence_level])


@pytest.fixture
def word_rules() -> RecursiveRules:
    """Return a word set of rules."""
    word_level = RecursiveLevel(delimiters=None, whitespace=True)
    return RecursiveRules(levels=[word_level])


@pytest.fixture
def token_rules() -> RecursiveRules:
    """Return a token set of rules."""
    token_level = RecursiveLevel(delimiters=None, whitespace=False)
    return RecursiveRules(levels=[token_level])


def test_recursive_chunker_initialization(
    sample_text: str, default_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can be initialized with a sample text."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=12
    )
    assert chunker is not None
    assert chunker.rules == default_rules
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_chunk == 12


def test_recursive_chunker_chunking(
    sample_text: str, default_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.text is not None for chunk in chunks)
    assert all(chunk.start_index is not None for chunk in chunks)
    assert all(chunk.end_index is not None for chunk in chunks)
    assert all(chunk.level is not None for chunk in chunks)


def test_recursive_chunker_token_count_default_rules(
    sample_text: str, default_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with default rules."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.text) >= 12 for chunk in chunks)


def test_recursive_chunker_reconstruction_default_rules(
    sample_text: str, default_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with default rules."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_indices_default_rules(
    sample_text: str, default_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with default rules."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_token_count_paragraph_rules(
    sample_text: str, paragraph_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with paragraph rules."""
    chunker = RecursiveChunker(
        rules=paragraph_rules, chunk_size=2048, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 2048 for chunk in chunks)
    assert all(len(chunk.text) >= 12 for chunk in chunks)


def test_recursive_chunker_reconstruction_paragraph_rules(
    sample_text: str, paragraph_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with paragraph rules."""
    chunker = RecursiveChunker(
        rules=paragraph_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_indices_paragraph_rules(
    sample_text: str, paragraph_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with paragraph rules."""
    chunker = RecursiveChunker(
        rules=paragraph_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_token_count_sentence_rules(
    sample_text: str, sentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with sentence rules."""
    chunker = RecursiveChunker(
        rules=sentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.text) >= 12 for chunk in chunks)


def test_recursive_chunker_reconstruction_sentence_rules(
    sample_text: str, sentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with sentence rules."""
    chunker = RecursiveChunker(
        rules=sentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_indices_sentence_rules(
    sample_text: str, sentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with sentence rules."""
    chunker = RecursiveChunker(
        rules=sentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_token_count_subsentence_rules(
    sample_text: str, subsentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with subsentence rules."""
    chunker = RecursiveChunker(
        rules=subsentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.text) >= 12 for chunk in chunks)


def test_recursive_chunker_reconstruction_subsentence_rules(
    sample_text: str, subsentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with subsentence rules."""
    chunker = RecursiveChunker(
        rules=subsentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_indices_subsentence_rules(
    sample_text: str, subsentence_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with subsentence rules."""
    chunker = RecursiveChunker(
        rules=subsentence_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_token_count_word_rules(
    sample_text: str, word_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with word rules."""
    chunker = RecursiveChunker(
        rules=word_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.text) >= 12 for chunk in chunks)


def test_recursive_chunker_reconstruction_word_rules(
    sample_text: str, word_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with word rules."""
    chunker = RecursiveChunker(
        rules=word_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_indices_word_rules(
    sample_text: str, word_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with word rules."""
    chunker = RecursiveChunker(
        rules=word_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_indices_token_rules(
    sample_text: str, token_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with token rules."""
    chunker = RecursiveChunker(
        rules=token_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index <= len(sample_text) for chunk in chunks)
    assert all(
        chunk.text == sample_text[chunk.start_index : chunk.end_index]
        for chunk in chunks
    )


def test_recursive_chunker_token_count_token_rules(
    sample_text: str, token_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can chunk a sample text with token rules."""
    chunker = RecursiveChunker(
        rules=token_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    for chunk in chunks:
        assert chunk.token_count <= 512, (
            f"Chunk {chunk} has token count {chunk.token_count}"
        )
    for chunk in chunks:
        assert len(chunk.text) >= 12, (
            f"Chunk {chunk} has character count {chunk.token_count}"
        )


def test_recursive_chunker_reconstruction_token_rules(
    sample_text: str, token_rules: RecursiveRules
) -> None:
    """Test that the RecursiveChunker can reconstruct a sample text with token rules."""
    chunker = RecursiveChunker(
        rules=token_rules, chunk_size=512, min_characters_per_chunk=12
    )
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    assert sample_text == "".join(chunk.text for chunk in chunks)


def test_recursive_chunker_empty_text(default_rules: RecursiveRules) -> None:
    """Test that the RecursiveChunker handles empty text correctly."""
    chunker = RecursiveChunker(rules=default_rules, chunk_size=512)
    chunks = chunker.chunk("")
    assert len(chunks) == 0


def test_recursive_chunker_single_character(default_rules: RecursiveRules) -> None:
    """Test that the RecursiveChunker handles single character text correctly."""
    chunker = RecursiveChunker(
        rules=default_rules, chunk_size=512, min_characters_per_chunk=1
    )
    chunks = chunker.chunk("a")
    assert len(chunks) == 1
    assert chunks[0].text == "a"
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == 1
    assert chunks[0].level == 0


def test_recursive_chunker_min_characters_per_chunk(sample_text: str) -> None:
    """Test that the RecursiveChunker handles min_characters_per_chunk correctly."""
    sample_text = "Hello!"
    chunker = RecursiveChunker(chunk_size=512, min_characters_per_chunk=20)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello!"

def test_recursive_chunker_from_recipe_default() -> None:
    """Test that RecursiveChunker.from_recipe works with default parameters."""
    chunker = RecursiveChunker.from_recipe()

    assert chunker is not None
    assert chunker.rules is not None and isinstance(chunker.rules, RecursiveRules)
    assert chunker.chunk_size == 2048
    assert chunker.min_characters_per_chunk == 24

def test_recursive_chunker_from_recipe_custom_params() -> None:
    """Test that RecursiveChunker.from_recipe works with custom parameters."""
    chunker = RecursiveChunker.from_recipe(
        name="default",
        lang="en",
        chunk_size=256,
        min_characters_per_chunk=32
    )

    assert chunker is not None
    assert chunker.rules is not None and isinstance(chunker.rules, RecursiveRules)
    assert chunker.chunk_size == 256
    assert chunker.min_characters_per_chunk == 32

def test_recursive_chunker_from_recipe_custom_lang() -> None:
    """Test that RecursiveChunker.from_recipe works with custom language."""
    chunker = RecursiveChunker.from_recipe(
        name="default",
        lang="en",
        tokenizer_or_token_counter="character",
        chunk_size=256,
        min_characters_per_chunk=32
    )
    
    assert chunker is not None
    assert chunker.rules is not None and isinstance(chunker.rules, RecursiveRules)
    assert chunker.chunk_size == 256
    assert chunker.min_characters_per_chunk == 32

def test_recursive_chunker_from_recipe_nonexistent() -> None:
    """Test that RecursiveChunker.from_recipe raises an error for nonexistent recipes."""
    with pytest.raises(ValueError):
        RecursiveChunker.from_recipe(name="invalid")

    with pytest.raises(ValueError):
        RecursiveChunker.from_recipe(name="default", lang="invalid")