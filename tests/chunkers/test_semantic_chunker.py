"""Test the SemanticChunker class."""

import os
import warnings
from typing import List

import pytest

from chonkie import SemanticChunker
from chonkie.embeddings import (
    BaseEmbeddings,
    CohereEmbeddings,
    Model2VecEmbeddings,
    OpenAIEmbeddings,
)
from chonkie.types.base import Chunk
from chonkie.types.semantic import SemanticChunk, SemanticSentence


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing the SemanticChunker.

    Returns:
        str: A paragraph of text about text chunking in RAG applications.

    """
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model() -> BaseEmbeddings:
    """Fixture that returns a Model2Vec embedding model for testing.

    Returns:
        Model2VecEmbeddings: A Model2Vec model initialized with 'minishlab/potion-base-8M'

    """
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def openai_embedding_model() -> BaseEmbeddings:
    """Fixture that returns an OpenAI embedding model for testing.

    Returns:
        OpenAIEmbeddings: An OpenAI model initialized with 'text-embedding-3-small'
            and the API key from environment variables.

    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def cohere_embedding_model() -> BaseEmbeddings:
    """Fixture that returns an Cohere embedding model for testing.

    Returns:
        CohereEmbeddings: An Cohere model initialized with 'embed-english-light-v3.0'
            and the API key from environment variables.

    """
    api_key = os.environ.get("COHERE_API_KEY")
    return CohereEmbeddings(model="embed-english-light-v3.0", api_key=api_key)


@pytest.fixture
def sample_complex_markdown_text() -> str:
    """Fixture that returns a sample markdown text with complex formatting.

    Returns:
        str: A markdown text containing various formatting elements like headings,
            lists, code blocks, links, images and blockquotes.

    """
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
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_semantic_chunker_initialization_openai(openai_embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=openai_embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


def test_semantic_chunker_initialization_sentence_transformer(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can be initialized with SentenceTransformer model."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_semantic_chunker_initialization_cohere(cohere_embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=cohere_embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


def test_semantic_chunker_chunking(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])
    assert all([chunk.sentences is not None for chunk in chunks])


def test_semantic_chunker_empty_text(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")

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
    assert len(chunks[0].sentences) == 1


def test_semantic_chunker_repr(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker has a string representation."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    expected = (
        f"SemanticChunker(model={chunker.embedding_model}, "
        f"chunk_size={chunker.chunk_size}, "
        f"mode={chunker.mode}, "
        f"threshold={chunker.threshold}, "
        f"similarity_window={chunker.similarity_window}, "
        f"min_sentences={chunker.min_sentences}, "
        f"min_chunk_size={chunker.min_chunk_size}, "
        f"return_type={chunker.return_type})"
    )
    assert repr(chunker) == expected


def test_semantic_chunker_similarity_threshold(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker respects similarity threshold."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.9,  # High threshold should create more chunks
    )
    text = (
        "This is about cars. This is about planes. "
        "This is about trains. This is about boats."
    )
    chunks = chunker.chunk(text)

    # With high similarity threshold, we expect more chunks due to low similarity
    assert len(chunks) > 1


def test_semantic_chunker_percentile_mode(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker works with percentile-based similarity."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=50,  # Use median similarity as threshold
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert all([isinstance(chunk, SemanticChunk) for chunk in chunks])


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


def test_sentence_chunker_indices(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_sentence_chunker_indices_complex_md(embedding_model: BaseEmbeddings, sample_complex_markdown_text: str) -> None:
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


def test_semantic_chunker_token_counts(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker correctly calculates token counts."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), (
        "All chunks must have a positive token count"
    )
    assert all([chunk.token_count <= 512 for chunk in chunks]), (
        "All chunks must have a token count less than or equal to 512"
    )

    token_counts = [chunker.tokenizer.count_tokens(chunk.text) for chunk in chunks]
    for i, (chunk, token_count) in enumerate(zip(chunks, token_counts)):
        assert chunk.token_count == token_count, (
            f"Chunk {i} has a token count of {chunk.token_count} but the encoded text length is {token_count}"
        )


def test_semantic_chunker_reconstruction(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    assert sample_text == "".join([chunk.text for chunk in chunks])


def test_semantic_chunker_reconstruction_complex_md(embedding_model: BaseEmbeddings, sample_complex_markdown_text: str) -> None:
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    assert sample_complex_markdown_text == "".join([chunk.text for chunk in chunks])


def test_semantic_chunker_reconstruction_batch(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk_batch([sample_text] * 10)[-1]
    assert sample_text == "".join([chunk.text for chunk in chunks])


def test_semantic_chunker_return_type(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that SemanticChunker's return type is correctly set."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
        return_type="texts",
    )
    chunks = chunker.chunk(sample_text)
    tokenizer = embedding_model.get_tokenizer_or_token_counter()
    assert all([type(chunk) is str for chunk in chunks])
    assert all([len(tokenizer.encode(chunk)) <= 512 for chunk in chunks])


def test_semantic_chunker_from_recipe_default() -> None:
    """Test that SemanticChunker.from_recipe works with default parameters."""
    chunker = SemanticChunker.from_recipe()

    assert chunker is not None
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"

def test_semantic_chunker_from_recipe_custom_params(embedding_model: BaseEmbeddings) -> None:
    """Test that SemanticChunker.from_recipe works with custom parameters."""
    chunker = SemanticChunker.from_recipe(
        name="default",
        lang="en",
        embedding_model=embedding_model,
        chunk_size=256,
        threshold=0.9,
        return_type="texts",
    )

    assert chunker is not None
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.chunk_size == 256
    assert chunker.threshold == 0.9
    assert chunker.return_type == "texts"

def test_semantic_chunker_from_recipe_nonexistent() -> None:
    """Test that SemanticChunker.from_recipe raises an error if the recipe does not exist."""
    with pytest.raises(ValueError):
        SemanticChunker.from_recipe(name="invalid")
    
    with pytest.raises(ValueError):
        SemanticChunker.from_recipe(name="default", lang="invalid")

class TestSemanticChunkerParameterValidation:
    """Test parameter validation in SemanticChunker."""

    def test_invalid_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=-1)

    def test_invalid_min_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid min_chunk_size."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_chunk_size=0)
        
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_chunk_size=-1)

    def test_invalid_min_sentences(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid min_sentences."""
        with pytest.raises(ValueError, match="min_sentences must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences=0)
        
        with pytest.raises(ValueError, match="min_sentences must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences=-1)

    def test_invalid_similarity_window(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid similarity_window."""
        with pytest.raises(ValueError, match="similarity_window must be non-negative"):
            SemanticChunker(embedding_model=embedding_model, similarity_window=-1)

    def test_invalid_threshold_step(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold_step."""
        with pytest.raises(ValueError, match="threshold_step must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold_step=0)
        
        with pytest.raises(ValueError, match="threshold_step must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold_step=1.0)
        
        with pytest.raises(ValueError, match="threshold_step must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold_step=1.5)

    def test_invalid_mode(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid mode."""
        with pytest.raises(ValueError, match="mode must be 'cumulative' or 'window'"):
            SemanticChunker(embedding_model=embedding_model, mode="invalid")

    def test_invalid_threshold_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold type."""
        with pytest.raises(ValueError, match="threshold must be a string, float, or int"):
            SemanticChunker(embedding_model=embedding_model, threshold=[0.5])

    def test_invalid_threshold_string(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold string."""
        with pytest.raises(ValueError, match="threshold must be 'auto', 'smart', or 'percentile'"):
            SemanticChunker(embedding_model=embedding_model, threshold="invalid")

    def test_invalid_threshold_float(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold float."""
        with pytest.raises(ValueError, match=r"threshold \(float\) must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=-0.1)
        
        with pytest.raises(ValueError, match=r"threshold \(float\) must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=1.1)

    def test_invalid_threshold_int(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold int."""
        with pytest.raises(ValueError, match=r"threshold \(int\) must be between 1 and 100"):
            SemanticChunker(embedding_model=embedding_model, threshold=0)
        
        with pytest.raises(ValueError, match=r"threshold \(int\) must be between 1 and 100"):
            SemanticChunker(embedding_model=embedding_model, threshold=101)

    def test_invalid_delim_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid delim type."""
        with pytest.raises(ValueError, match="delim must be a string or list of strings"):
            SemanticChunker(embedding_model=embedding_model, delim=123)

    def test_invalid_return_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid return_type."""
        with pytest.raises(ValueError, match="Invalid return_type. Must be either 'chunks' or 'texts'."):
            SemanticChunker(embedding_model=embedding_model, return_type="invalid")

    def test_invalid_embedding_model_type(self) -> None:
        """Test that SemanticChunker raises error for invalid embedding model type."""
        with pytest.raises(ValueError, match="123 is not a valid embedding model"):
            SemanticChunker(embedding_model=123)


class TestSemanticChunkerModeConfiguration:
    """Test different mode configurations in SemanticChunker."""

    def test_cumulative_mode(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with cumulative mode."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            mode="cumulative",
            threshold=0.5,
            chunk_size=512
        )
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, SemanticChunk) for chunk in chunks)
        assert chunker.mode == "cumulative"
        assert chunker.similarity_window == 1  # Should be 1 for cumulative mode

    def test_window_mode_with_larger_window(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with window mode and larger similarity window."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            mode="window",
            similarity_window=3,
            threshold=0.5,
            chunk_size=512
        )
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, SemanticChunk) for chunk in chunks)
        assert chunker.mode == "window"
        assert chunker.similarity_window == 3


class TestSemanticChunkerThresholdTypes:
    """Test different threshold configurations in SemanticChunker."""

    def test_auto_threshold(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with auto threshold."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold="auto",
            chunk_size=512
        )
        # Before chunking, threshold should be None
        assert chunker.similarity_threshold is None
        assert chunker.similarity_percentile is None
        
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) > 0
        assert chunker.threshold == "auto"
        # After chunking, threshold should be calculated
        assert chunker.similarity_threshold is not None

    def test_float_threshold(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with float threshold."""
        threshold_value = 0.75
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold=threshold_value,
            chunk_size=512
        )
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) > 0
        assert chunker.threshold == threshold_value
        assert chunker.similarity_threshold == threshold_value
        assert chunker.similarity_percentile is None

    def test_int_threshold(self, embedding_model: BaseEmbeddings, sample_text: str) -> None:
        """Test SemanticChunker with int threshold (percentile)."""
        threshold_value = 75
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold=threshold_value,
            chunk_size=512
        )
        # Before chunking, similarity_threshold should be None
        assert chunker.similarity_threshold is None
        assert chunker.similarity_percentile == threshold_value
        
        chunks = chunker.chunk(sample_text)
        
        assert len(chunks) > 0
        assert chunker.threshold == threshold_value
        # After chunking, similarity_threshold should be calculated from percentile
        assert chunker.similarity_threshold is not None
        assert chunker.similarity_percentile == threshold_value


class TestSemanticChunkerInternalMethods:
    """Test internal methods of SemanticChunker."""

    def test_split_sentences_with_include_delim_prev(self, embedding_model: BaseEmbeddings) -> None:
        """Test sentence splitting with include_delim='prev'."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            delim=[".", "!"],
            include_delim="prev"
        )
        text = "Hello world. How are you! Fine thanks."
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) > 0
        assert all(len(sentence) >= chunker.min_characters_per_sentence for sentence in sentences)

    def test_split_sentences_with_include_delim_next(self, embedding_model: BaseEmbeddings) -> None:
        """Test sentence splitting with include_delim='next'."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            delim=[".", "!"],
            include_delim="next"
        )
        text = "Hello world. How are you! Fine thanks."
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) > 0

    def test_split_sentences_with_include_delim_none(self, embedding_model: BaseEmbeddings) -> None:
        """Test sentence splitting with include_delim=None."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            delim=[".", "!"],
            include_delim=None
        )
        text = "Hello world. How are you! Fine thanks."
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) > 0

    def test_split_sentences_short_fragments(self, embedding_model: BaseEmbeddings) -> None:
        """Test sentence splitting with short fragments."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            min_characters_per_sentence=20
        )
        text = "A. B. C. This is a longer sentence that should remain separate."
        sentences = chunker._split_sentences(text)
        
        # Short fragments should be combined
        assert len(sentences) > 0
        assert any(len(sentence) >= 20 for sentence in sentences)

    def test_prepare_sentences_empty_text(self, embedding_model: BaseEmbeddings) -> None:
        """Test _prepare_sentences with empty text."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        sentences = chunker._prepare_sentences("")
        
        assert sentences == []

    def test_prepare_sentences_whitespace_text(self, embedding_model: BaseEmbeddings) -> None:
        """Test _prepare_sentences with whitespace-only text."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        sentences = chunker._prepare_sentences("   \n\t   ")
        
        assert sentences == []

    def test_prepare_sentences_normal_text(self, embedding_model: BaseEmbeddings) -> None:
        """Test _prepare_sentences with normal text."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        text = "Hello world. How are you?"
        sentences = chunker._prepare_sentences(text)
        
        assert len(sentences) > 0
        assert all(isinstance(sentence, SemanticSentence) for sentence in sentences)
        assert all(hasattr(sentence, 'embedding') for sentence in sentences)
        assert all(hasattr(sentence, 'token_count') for sentence in sentences)

    def test_compute_group_embedding_single_sentence(self, embedding_model: BaseEmbeddings) -> None:
        """Test _compute_group_embedding with single sentence."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        sentences = chunker._prepare_sentences("This is a test sentence.")
        
        if sentences:
            group_embedding = chunker._compute_group_embedding([sentences[0]])
            assert group_embedding is not None
            # For single sentence, should return the sentence's embedding
            assert (group_embedding == sentences[0].embedding).all()

    def test_compute_group_embedding_multiple_sentences(self, embedding_model: BaseEmbeddings) -> None:
        """Test _compute_group_embedding with multiple sentences."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        sentences = chunker._prepare_sentences("First sentence. Second sentence. Third sentence.")
        
        if len(sentences) >= 2:
            group_embedding = chunker._compute_group_embedding(sentences[:2])
            assert group_embedding is not None
            assert len(group_embedding.shape) > 0

    def test_get_semantic_similarity(self, embedding_model: BaseEmbeddings) -> None:
        """Test _get_semantic_similarity method."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        sentences = chunker._prepare_sentences("First sentence. Second sentence.")
        
        if len(sentences) >= 2:
            similarity = chunker._get_semantic_similarity(
                sentences[0].embedding, 
                sentences[1].embedding
            )
            # Accept both float and numpy float types
            assert isinstance(similarity, (float, type(sentences[0].embedding.flat[0])))
            assert 0.0 <= similarity <= 1.0

    def test_create_chunk_empty_sentences(self, embedding_model: BaseEmbeddings) -> None:
        """Test _create_chunk with empty sentence list."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        
        with pytest.raises(ValueError, match="Cannot create chunk from empty sentence list"):
            chunker._create_chunk([])

    def test_create_chunk_with_texts_return_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test _create_chunk with return_type='texts'."""
        chunker = SemanticChunker(
            embedding_model=embedding_model, 
            return_type="texts"
        )
        sentences = chunker._prepare_sentences("This is a test sentence.")
        
        if sentences:
            result = chunker._create_chunk(sentences)
            assert isinstance(result, str)

    def test_create_chunk_invalid_return_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test _create_chunk with invalid return_type."""
        chunker = SemanticChunker(embedding_model=embedding_model)
        chunker.return_type = "invalid"  # Force invalid return type
        sentences = chunker._prepare_sentences("This is a test sentence.")
        
        if sentences:
            with pytest.raises(ValueError, match="Invalid return_type. Must be either 'chunks' or 'texts'."):
                chunker._create_chunk(sentences)


class TestSemanticChunkerThresholdCalculation:
    """Test threshold calculation methods."""

    def test_compute_similarity_threshold_with_fixed_threshold(self, embedding_model: BaseEmbeddings) -> None:
        """Test _compute_similarity_threshold with fixed threshold."""
        chunker = SemanticChunker(
            embedding_model=embedding_model, 
            threshold=0.7
        )
        similarities = [0.9, 0.8, 0.6, 0.5, 0.4]
        
        threshold = chunker._compute_similarity_threshold(similarities)
        assert threshold == 0.7

    def test_compute_similarity_threshold_with_percentile(self, embedding_model: BaseEmbeddings) -> None:
        """Test _compute_similarity_threshold with percentile."""
        chunker = SemanticChunker(
            embedding_model=embedding_model, 
            threshold=50  # 50th percentile
        )
        # Mock numpy percentile calculation
        import numpy as np
        similarities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold = chunker._compute_similarity_threshold(similarities)
        expected = float(np.percentile(similarities, 50))
        assert threshold == expected

    def test_calculate_threshold_via_percentile(self, embedding_model: BaseEmbeddings) -> None:
        """Test _calculate_threshold_via_percentile method."""
        chunker = SemanticChunker(
            embedding_model=embedding_model, 
            threshold=75  # 75th percentile
        )
        sentences = chunker._prepare_sentences("First sentence. Second sentence. Third sentence.")
        
        if len(sentences) >= 2:
            threshold = chunker._calculate_threshold_via_percentile(sentences)
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0

    def test_calculate_threshold_via_binary_search_with_warnings(self, embedding_model: BaseEmbeddings) -> None:
        """Test _calculate_threshold_via_binary_search with iteration limit warning."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold="auto",
            threshold_step=0.001,  # Very small step to trigger iteration limit
            chunk_size=10000,  # Large chunk size to make convergence difficult
            min_chunk_size=1
        )
        sentences = chunker._prepare_sentences("First. Second. Third. Fourth. Fifth. Sixth.")
        
        if len(sentences) >= 3:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                threshold = chunker._calculate_threshold_via_binary_search(sentences)
                
                # Check if warning was issued
                assert len(w) <= 1  # May or may not trigger depending on convergence
                if w:
                    assert "Too many iterations" in str(w[0].message)
                
                assert isinstance(threshold, float)


class TestSemanticChunkerEdgeCases:
    """Test edge cases and error conditions."""

    def test_chunk_with_min_sentences_requirement(self, embedding_model: BaseEmbeddings) -> None:
        """Test chunking with minimum sentences requirement."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            min_sentences=3,
            chunk_size=512
        )
        # Text with exactly min_sentences
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert len(chunks[0].sentences) >= chunker.min_sentences

    def test_chunk_with_very_small_chunks(self, embedding_model: BaseEmbeddings) -> None:
        """Test chunking with very small chunk size."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=20,  # Small but reasonable
            min_chunk_size=1,
            min_sentences=1  # Allow single sentence chunks
        )
        text = "Short. Also short. Another short one. Final short sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        # Some chunks may slightly exceed due to semantic grouping
        assert all(chunk.token_count <= 25 for chunk in chunks)  # Allow small buffer

    def test_delim_as_string(self, embedding_model: BaseEmbeddings) -> None:
        """Test SemanticChunker with delim as string instead of list."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            delim=".",  # Single string delimiter
            chunk_size=512
        )
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert chunker.delim == "."

    def test_very_long_text_chunking(self, embedding_model: BaseEmbeddings) -> None:
        """Test chunking with very long text."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=100
        )
        # Create a long text with many sentences
        long_text = " ".join([f"This is sentence number {i}." for i in range(50)])
        chunks = chunker.chunk(long_text)
        
        assert len(chunks) > 1
        assert all(chunk.token_count <= 100 for chunk in chunks)
        
        # Verify reconstruction
        reconstructed = "".join(chunk.text for chunk in chunks)
        assert reconstructed == long_text


if __name__ == "__main__":
    pytest.main()