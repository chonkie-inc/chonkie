"""Test the overlap refinery module."""

import pytest

from chonkie.refinery import OverlapRefinery
from chonkie.types import Chunk


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Fixture to create sample chunks."""
    return [
        Chunk(text="This is the first chunk of text.", start_index=0, end_index=31, token_count=7),
        Chunk(
            text="This is the second chunk of text.",
            start_index=32,
            end_index=64,
            token_count=7,
        ),
        Chunk(
            text="This is the third chunk of text.",
            start_index=65,
            end_index=96,
            token_count=7,
        ),
    ]


def test_overlap_refinery_initialization() -> None:
    """Test the OverlapRefinery initialization."""
    refinery = OverlapRefinery()
    assert refinery is not None
    assert isinstance(refinery, OverlapRefinery)
    assert refinery.chunk_overlap == 0
    assert refinery._overlap_mode == "token"
    assert refinery._overlap_method == "suffix"
    assert refinery._overlap_enabled is False


def test_overlap_refinery_initialization_with_int_overlap() -> None:
    """Test initialization with integer chunk_overlap."""
    refinery = OverlapRefinery(chunk_overlap=5)
    assert refinery.chunk_overlap == 5
    assert refinery._overlap_enabled is True


def test_overlap_refinery_initialization_with_float_overlap() -> None:
    """Test initialization with float chunk_overlap (fraction of chunk size)."""
    refinery = OverlapRefinery(chunk_overlap=0.3)
    assert refinery.chunk_overlap == 0.3
    assert refinery._overlap_enabled is True


def test_overlap_refinery_initialization_with_invalid_mode() -> None:
    """Test that invalid mode is accepted at init (raises at refine time)."""
    refinery = OverlapRefinery(chunk_overlap=1, overlap_mode="invalid")
    with pytest.raises(ValueError):
        chunks = [
            Chunk(text="a", start_index=0, end_index=1, token_count=1),
            Chunk(text="b", start_index=1, end_index=2, token_count=1),
        ]
        refinery.refine(chunks)


def test_overlap_refinery_initialization_with_invalid_method() -> None:
    """Test that invalid method is accepted at init (raises at refine time)."""
    refinery = OverlapRefinery(chunk_overlap=1, overlap_method="invalid")
    with pytest.raises(ValueError):
        chunks = [
            Chunk(text="a", start_index=0, end_index=1, token_count=1),
            Chunk(text="b", start_index=1, end_index=2, token_count=1),
        ]
        refinery.refine(chunks)


def test_overlap_refinery_refine_empty_chunks() -> None:
    """Test the OverlapRefinery.refine method with empty chunks."""
    refinery = OverlapRefinery()
    chunks = []
    refined_chunks = refinery.refine(chunks)
    assert refined_chunks == []


def test_overlap_refinery_refine_different_chunk_types() -> None:
    """Test the OverlapRefinery.refine method with different chunk types."""
    refinery = OverlapRefinery()

    class CustomChunk(Chunk):
        pass

    chunks = [
        Chunk(text="This is the first chunk of text.", start_index=0, end_index=31, token_count=7),
        CustomChunk(
            text="This is the second chunk of text.",
            start_index=32,
            end_index=64,
            token_count=7,
        ),
    ]

    with pytest.raises(ValueError):
        refinery.refine(chunks)


def test_overlap_refinery_token_suffix_overlap(sample_chunks) -> None:
    """Test the OverlapRefinery with token-based suffix overlap."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[0], "context")
    assert refined_chunks[0].context != ""

    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""

    # Text is updated with context (always merged now)
    assert refined_chunks[0].text.endswith(refined_chunks[0].context)
    assert refined_chunks[1].text.endswith(refined_chunks[1].context)

    # Third chunk (last) is not modified
    assert refined_chunks[2].text == "This is the third chunk of text."


def test_overlap_refinery_token_prefix_overlap(sample_chunks) -> None:
    """Test the OverlapRefinery with token-based prefix overlap."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="prefix")
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""

    assert hasattr(refined_chunks[2], "context")
    assert refined_chunks[2].context != ""

    # Text is updated with context
    assert refined_chunks[1].text.startswith(refined_chunks[1].context)
    assert refined_chunks[2].text.startswith(refined_chunks[2].context)

    # First chunk is not modified
    assert refined_chunks[0].text == "This is the first chunk of text."


def test_overlap_refinery_token_suffix_overlap_float_context(sample_chunks) -> None:
    """Test the OverlapRefinery with float chunk_overlap (fraction of chunk size)."""
    refinery = OverlapRefinery(chunk_overlap=0.3, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(sample_chunks)

    assert refinery.chunk_overlap == 0.3

    assert hasattr(refined_chunks[0], "context")
    assert refined_chunks[0].context != ""

    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""


def test_overlap_refinery_token_overlap_large_context(sample_chunks) -> None:
    """Test the OverlapRefinery with large overlap size."""
    refinery = OverlapRefinery(chunk_overlap=10, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[0], "context")
    assert refined_chunks[0].context != ""
    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""


def test_overlap_refinery_recursive_suffix_overlap(sample_chunks) -> None:
    """Test the OverlapRefinery with recursive-based suffix overlap."""
    refinery = OverlapRefinery(chunk_overlap=3, overlap_mode="recursive", overlap_method="suffix")
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[0], "context")
    assert refined_chunks[0].context != ""

    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""

    assert refined_chunks[0].text.endswith(refined_chunks[0].context)
    assert refined_chunks[1].text.endswith(refined_chunks[1].context)

    assert refined_chunks[2].text == "This is the third chunk of text."


def test_overlap_refinery_recursive_prefix_overlap(sample_chunks) -> None:
    """Test the OverlapRefinery with recursive-based prefix overlap."""
    refinery = OverlapRefinery(chunk_overlap=3, overlap_mode="recursive", overlap_method="prefix")
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[1], "context")
    assert refined_chunks[1].context != ""

    assert hasattr(refined_chunks[2], "context")
    assert refined_chunks[2].context != ""

    assert refined_chunks[1].text.startswith(refined_chunks[1].context)
    assert refined_chunks[2].text.startswith(refined_chunks[2].context)

    assert refined_chunks[0].text == "This is the first chunk of text."


def test_overlap_refinery_recursive_large_context(sample_chunks) -> None:
    """Test the OverlapRefinery with recursive mode and large context size."""
    refinery = OverlapRefinery(
        chunk_overlap=100, overlap_mode="recursive", overlap_method="suffix"
    )
    refined_chunks = refinery.refine(sample_chunks)

    assert hasattr(refined_chunks[0], "context")
    assert hasattr(refined_chunks[1], "context")


def test_overlap_refinery_recursive_with_custom_rules() -> None:
    """Test the OverlapRefinery with custom recursive rules."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    custom_rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=[". ", "! ", "? "]),
            RecursiveLevel(whitespace=True),
        ],
    )

    chunks = [
        Chunk(text="First sentence. Second sentence.", start_index=0, end_index=31, token_count=6),
        Chunk(
            text="Third sentence. Fourth sentence.",
            start_index=32,
            end_index=63,
            token_count=6,
        ),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=2,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=custom_rules,
    )

    refined_chunks = refinery.refine(chunks)

    assert hasattr(refined_chunks[0], "context")
    assert refined_chunks[0].context != ""


def test_overlap_refinery_recursive_exceeding_levels() -> None:
    """Test the OverlapRefinery when recursive levels are exceeded."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    minimal_rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=[". "]),
        ],
    )

    chunks = [
        Chunk(
            text="NoSentenceDelimitersHereJustOneString",
            start_index=0,
            end_index=37,
            token_count=10,
        ),
        Chunk(text="AnotherChunkWithoutDelimiters", start_index=38, end_index=66, token_count=8),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=5,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=minimal_rules,
    )

    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 2
    assert hasattr(refined_chunks[0], "context")


def test_overlap_refinery_recursive_empty_text() -> None:
    """Test the OverlapRefinery with empty text chunks."""
    chunks = [
        Chunk(text="", start_index=0, end_index=0, token_count=0),
        Chunk(text="Some text here", start_index=1, end_index=14, token_count=3),
    ]

    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="recursive", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 2


def test_overlap_refinery_recursive_single_chunk() -> None:
    """Test the OverlapRefinery with a single chunk in recursive mode."""
    chunks = [
        Chunk(text="Single chunk of text.", start_index=0, end_index=20, token_count=4),
    ]

    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="recursive", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 1
    assert refined_chunks[0].text == "Single chunk of text."


def test_overlap_refinery_float_context_calculation() -> None:
    """Test that float context size calculation works correctly."""
    chunks = [
        Chunk(text="Small chunk", start_index=0, end_index=10, token_count=2),
        Chunk(text="Medium sized chunk here", start_index=11, end_index=33, token_count=5),
        Chunk(
            text="Very long chunk with many tokens here for testing",
            start_index=34,
            end_index=83,
            token_count=10,
        ),
    ]

    # 0.5 * max(10) = 5 tokens
    refinery = OverlapRefinery(chunk_overlap=0.5, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert refinery.chunk_overlap == 0.5
    assert len(refined_chunks) == 3


def test_overlap_refinery_very_small_chunks() -> None:
    """Test with very small chunks that might cause issues."""
    chunks = [
        Chunk(text="A", start_index=0, end_index=0, token_count=1),
        Chunk(text="B", start_index=1, end_index=1, token_count=1),
        Chunk(text="C", start_index=2, end_index=2, token_count=1),
    ]

    refinery = OverlapRefinery(chunk_overlap=1, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 3
    assert hasattr(refined_chunks[0], "context")
    assert hasattr(refined_chunks[1], "context")


def test_overlap_refinery_context_larger_than_chunk() -> None:
    """Test when context size is larger than the chunk itself."""
    chunks = [
        Chunk(text="Short", start_index=0, end_index=4, token_count=1),
        Chunk(text="Also short", start_index=5, end_index=14, token_count=2),
    ]

    refinery = OverlapRefinery(chunk_overlap=10, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)
    assert len(refined_chunks) == 2


def test_overlap_refinery_recursive_with_only_whitespace() -> None:
    """Test recursive mode with text that only has whitespace delimiters."""
    chunks = [
        Chunk(text="word1 word2 word3", start_index=0, end_index=16, token_count=3),
        Chunk(text="word4 word5 word6", start_index=17, end_index=33, token_count=3),
    ]

    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="recursive", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 2
    assert hasattr(refined_chunks[0], "context")


def test_overlap_refinery_invalid_refine_method() -> None:
    """Test that invalid method parameter is caught."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="suffix")
    refinery._overlap_method = "invalid_method"

    chunks = [
        Chunk(text="Test chunk", start_index=0, end_index=9, token_count=2),
        Chunk(text="Another chunk", start_index=10, end_index=22, token_count=2),
    ]

    with pytest.raises(ValueError):
        refinery.refine(chunks)


def test_overlap_refinery_invalid_mode_in_overlap() -> None:
    """Test that invalid mode parameter is caught in overlap methods."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="suffix")
    refinery._overlap_mode = "invalid_mode"

    chunks = [
        Chunk(text="Test chunk", start_index=0, end_index=9, token_count=2),
        Chunk(text="Another chunk", start_index=10, end_index=22, token_count=2),
    ]

    with pytest.raises(ValueError):
        refinery.refine(chunks)


def test_overlap_refinery_stress_test_many_chunks() -> None:
    """Stress test with many chunks."""
    chunks = [
        Chunk(
            text=f"Chunk number {i} with some text",
            start_index=i * 30,
            end_index=(i + 1) * 30 - 1,
            token_count=6,
        )
        for i in range(100)
    ]

    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert len(refined_chunks) == 100
    for i in range(99):
        assert hasattr(refined_chunks[i], "context")


def test_overlap_refinery_recursive_stress_deep_nesting() -> None:
    """Test recursive mode with content that forces deep recursion."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    deep_rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=["|||"]),
            RecursiveLevel(delimiters=[":::"]),
        ],
    )

    chunks = [
        Chunk(text="NoSpecialDelimitersHereAtAll", start_index=0, end_index=27, token_count=5),
        Chunk(text="AnotherChunkWithoutDelimiters", start_index=28, end_index=56, token_count=5),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=3,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=deep_rules,
    )

    refined_chunks = refinery.refine(chunks)
    assert len(refined_chunks) == 2


def test_overlap_refinery_index_preservation() -> None:
    """Test that start_index and end_index are preserved when adding context."""
    chunks = [
        Chunk(text="Hello world!", start_index=0, end_index=11, token_count=2),
        Chunk(text="How are you", start_index=13, end_index=23, token_count=3),
        Chunk(text="today?", start_index=25, end_index=30, token_count=1),
    ]

    # Test suffix overlap
    refinery_suffix = OverlapRefinery(
        chunk_overlap=1, overlap_mode="token", overlap_method="suffix"
    )
    suffix_chunks = refinery_suffix.refine([chunk.copy() for chunk in chunks])

    assert suffix_chunks[0].start_index == 0
    assert suffix_chunks[0].end_index == 11
    assert suffix_chunks[1].start_index == 13
    assert suffix_chunks[1].end_index == 23
    assert suffix_chunks[2].start_index == 25
    assert suffix_chunks[2].end_index == 30

    # Test prefix overlap
    refinery_prefix = OverlapRefinery(
        chunk_overlap=1, overlap_mode="token", overlap_method="prefix"
    )
    prefix_chunks = refinery_prefix.refine([chunk.copy() for chunk in chunks])

    assert prefix_chunks[0].start_index == 0
    assert prefix_chunks[0].end_index == 11
    assert prefix_chunks[1].start_index == 13
    assert prefix_chunks[1].end_index == 23
    assert prefix_chunks[2].start_index == 25
    assert prefix_chunks[2].end_index == 30


def test_overlap_refinery_recursive_index_preservation() -> None:
    """Test that indices are preserved in recursive mode."""
    chunks = [
        Chunk(text="First sentence. Second part.", start_index=0, end_index=27, token_count=5),
        Chunk(text="Third sentence here.", start_index=29, end_index=48, token_count=3),
    ]

    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="recursive", overlap_method="suffix")
    refined_chunks = refinery.refine(chunks)

    assert refined_chunks[0].start_index == 0
    assert refined_chunks[0].end_index == 27
    assert refined_chunks[1].start_index == 29
    assert refined_chunks[1].end_index == 48


def test_overlap_refinery_recursive_delimiter_modes() -> None:
    """Test recursive mode with different delimiter inclusion modes."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    chunks = [
        Chunk(text="First. Second.", start_index=0, end_index=13, token_count=3),
        Chunk(text="Third. Fourth.", start_index=15, end_index=28, token_count=3),
    ]

    # Test include_delim="prev"
    rules_prev = RecursiveRules(
        levels=[RecursiveLevel(delimiters=[". "], include_delim="prev")],
    )
    refinery_prev = OverlapRefinery(
        chunk_overlap=2,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=rules_prev,
    )
    refined_prev = refinery_prev.refine([chunk.copy() for chunk in chunks])
    assert len(refined_prev) == 2

    # Test include_delim="next"
    rules_next = RecursiveRules(
        levels=[RecursiveLevel(delimiters=[". "], include_delim="next")],
    )
    refinery_next = OverlapRefinery(
        chunk_overlap=2,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=rules_next,
    )
    refined_next = refinery_next.refine([chunk.copy() for chunk in chunks])
    assert len(refined_next) == 2

    # Test include_delim=None (removes delimiters)
    rules_none = RecursiveRules(
        levels=[RecursiveLevel(delimiters=[". "], include_delim=None)],
    )
    refinery_none = OverlapRefinery(
        chunk_overlap=2,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=rules_none,
    )
    refined_none = refinery_none.refine([chunk.copy() for chunk in chunks])
    assert len(refined_none) == 2


def test_overlap_refinery_empty_text_recursive() -> None:
    """Test recursive overlap with empty text."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    chunks = [
        Chunk(text="", start_index=0, end_index=0, token_count=0),
        Chunk(text="Some text", start_index=1, end_index=9, token_count=2),
    ]

    rules = RecursiveRules(levels=[RecursiveLevel(delimiters=[". "])])
    refinery = OverlapRefinery(
        chunk_overlap=1,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=rules,
    )
    refined_chunks = refinery.refine(chunks)
    assert len(refined_chunks) == 2

    chunks2 = [
        Chunk(text="No delimiters", start_index=0, end_index=12, token_count=2),
        Chunk(text="", start_index=13, end_index=13, token_count=0),
    ]
    refined_chunks2 = refinery.refine(chunks2)
    assert len(refined_chunks2) == 2


def test_overlap_refinery_context_size_larger_than_chunk() -> None:
    """When context_size exceeds chunk's token count, entire chunk text is returned as context."""
    chunks = [
        Chunk(text="A", start_index=0, end_index=0, token_count=1),
        Chunk(text="B", start_index=1, end_index=1, token_count=1),
    ]

    # Suffix: context from chunk 1 ("B") appended to chunk 0
    refinery_suffix = OverlapRefinery(
        chunk_overlap=100, overlap_mode="token", overlap_method="suffix"
    )
    refined_chunks = refinery_suffix.refine(chunks)

    assert refined_chunks[0].text == "AB"
    assert refined_chunks[0].context == "B"
    assert len(refined_chunks) == 2

    # Prefix: context from chunk 0 ("A") prepended to chunk 1
    fresh_chunks = [
        Chunk(text="A", start_index=0, end_index=0, token_count=1),
        Chunk(text="B", start_index=1, end_index=1, token_count=1),
    ]
    refinery_prefix = OverlapRefinery(
        chunk_overlap=100, overlap_mode="token", overlap_method="prefix"
    )
    refined_chunks = refinery_prefix.refine(fresh_chunks)

    assert refined_chunks[1].text == "AB"
    assert refined_chunks[1].context == "A"
    assert len(refined_chunks) == 2


def test_overlap_refinery_invalid_modes() -> None:
    """Test error handling for invalid modes."""
    chunks = [
        Chunk(text="Test", start_index=0, end_index=3, token_count=1),
        Chunk(text="More", start_index=4, end_index=7, token_count=1),
    ]

    refinery = OverlapRefinery(chunk_overlap=1, overlap_mode="token", overlap_method="prefix")
    refinery._overlap_mode = "invalid_mode"

    with pytest.raises(ValueError, match="Mode must be one of: token, recursive"):
        refinery.refine(chunks)


def test_overlap_refinery_context_size_reuse_correctness() -> None:
    """Test that reusing OverlapRefinery with float chunk_overlap works correctly with different chunk sets."""
    refinery = OverlapRefinery(chunk_overlap=0.3, overlap_mode="token", overlap_method="suffix")

    # First set: small token counts -> context_size should be 0.3 * 5 = 1.5 -> 1
    small_chunks = [
        Chunk(text="Short text", start_index=0, end_index=9, token_count=2),
        Chunk(text="Another brief chunk here", start_index=10, end_index=33, token_count=5),
    ]
    refined_small = refinery.refine([c.copy() for c in small_chunks])

    # Second set: large token counts -> context_size should be 0.3 * 20 = 6
    large_chunks = [
        Chunk(
            text="This is a significantly longer text chunk with many more tokens",
            start_index=0,
            end_index=62,
            token_count=12,
        ),
        Chunk(
            text="Another very long chunk with substantial content that contains many words and tokens for testing",
            start_index=63,
            end_index=159,
            token_count=20,
        ),
    ]
    refined_large = refinery.refine([c.copy() for c in large_chunks])

    small_context = getattr(refined_small[0], "context", "")
    large_context = getattr(refined_large[0], "context", "")

    assert len(refined_small) == 2
    assert len(refined_large) == 2
    assert small_context is not None and small_context != ""
    assert large_context is not None and large_context != ""


def test_overlap_refinery_repr() -> None:
    """Test the OverlapRefinery.__repr__ method."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_mode="token", overlap_method="suffix")
    repr_str = repr(refinery)
    assert "OverlapRefinery" in repr_str
    assert "chunk_overlap=2" in repr_str
    assert "overlap_mode=token" in repr_str
    assert "overlap_method=suffix" in repr_str


def test_overlap_refinery_float_context_size_preservation() -> None:
    """Test that float chunk_overlap is preserved and recalculated for each chunk set."""
    refinery = OverlapRefinery(chunk_overlap=0.4, overlap_mode="token", overlap_method="suffix")

    assert refinery.chunk_overlap == 0.4

    small_chunks = [
        Chunk(text="Small", start_index=0, end_index=4, token_count=2),
        Chunk(text="Test", start_index=5, end_index=8, token_count=1),
    ]

    large_chunks = [
        Chunk(text="This is a much longer text chunk", start_index=0, end_index=32, token_count=8),
        Chunk(
            text="Another long chunk with more content",
            start_index=33,
            end_index=68,
            token_count=7,
        ),
    ]

    refinery.refine(small_chunks)
    refinery.refine(large_chunks)

    assert refinery.chunk_overlap == 0.4
    assert isinstance(refinery.chunk_overlap, float)


# ---- Tests for explicit tokenizer ----


def test_overlap_refinery_with_explicit_tokenizer() -> None:
    """Test that overlap_tokenizer parameter works correctly."""
    chunks = [
        Chunk(text="Hello world, this is chunk one.", start_index=0, end_index=29, token_count=6),
        Chunk(text="And this is chunk number two.", start_index=30, end_index=58, token_count=6),
        Chunk(text="Finally the third chunk here.", start_index=59, end_index=87, token_count=6),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=3,
        overlap_mode="token",
        overlap_method="suffix",
        overlap_tokenizer="character",
    )

    refined = refinery.refine(chunks)

    # With character tokenizer and chunk_overlap=3, suffix takes 3 chars from start of next chunk
    assert refined[0].context == "And"
    assert refined[1].context == "Fin"
    assert refined[2].text == "Finally the third chunk here."


def test_overlap_refinery_with_word_tokenizer() -> None:
    """Test overlap with word tokenizer."""
    chunks = [
        Chunk(text="alpha beta gamma", start_index=0, end_index=15, token_count=3),
        Chunk(text="delta epsilon zeta", start_index=16, end_index=33, token_count=3),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=1,
        overlap_mode="token",
        overlap_method="suffix",
        overlap_tokenizer="word",
    )

    refined = refinery.refine(chunks)

    assert refined[0].context == "delta"
    assert refined[0].text == "alpha beta gammadelta"


def test_overlap_refinery_prefix_with_tokenizer() -> None:
    """Test prefix overlap with explicit tokenizer."""
    chunks = [
        Chunk(text="alpha beta gamma", start_index=0, end_index=15, token_count=3),
        Chunk(text="delta epsilon zeta", start_index=16, end_index=33, token_count=3),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=2,
        overlap_mode="token",
        overlap_method="prefix",
        overlap_tokenizer="word",
    )

    refined = refinery.refine(chunks)

    assert refined[1].context == "beta gamma"
    assert refined[1].text.startswith("beta gamma")


def test_overlap_refinery_recursive_mode_with_rules() -> None:
    """Test recursive overlap mode with custom rules and tokenizer."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=[". "]),
            RecursiveLevel(whitespace=True),
        ],
    )

    chunks = [
        Chunk(text="First part here.", start_index=0, end_index=15, token_count=4),
        Chunk(text="Second part. Third part.", start_index=16, end_index=39, token_count=6),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=3,
        overlap_mode="recursive",
        overlap_method="suffix",
        overlap_rules=rules,
        overlap_tokenizer="word",
    )

    refined = refinery.refine(chunks)

    assert refined[0].context != ""
    assert refined[0].text.endswith(refined[0].context)


def test_overlap_refinery_all_params_together() -> None:
    """Integration test: all overlap params work together correctly."""
    from chonkie.types import RecursiveLevel, RecursiveRules

    rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=[". ", "! "]),
            RecursiveLevel(whitespace=True),
        ],
    )

    chunks = [
        Chunk(
            text="The quick brown fox. Jumps over the lazy dog!",
            start_index=0,
            end_index=45,
            token_count=10,
        ),
        Chunk(
            text="Another sentence here. And one more thing.",
            start_index=46,
            end_index=87,
            token_count=8,
        ),
        Chunk(
            text="Final chunk of text. Nothing else to say.",
            start_index=88,
            end_index=128,
            token_count=9,
        ),
    ]

    refinery = OverlapRefinery(
        chunk_overlap=4,
        overlap_mode="recursive",
        overlap_method="prefix",
        overlap_rules=rules,
        overlap_tokenizer="word",
    )

    refined = refinery.refine(chunks)

    # Context is set from previous chunks (prefix method)
    assert refined[1].context != ""
    assert refined[2].context != ""

    # Text is merged with context
    assert refined[1].text.startswith(refined[1].context)
    assert refined[2].text.startswith(refined[2].context)


def test_overlap_refinery_clear_cache() -> None:
    """Test that clear_overlap_cache works without error."""
    refinery = OverlapRefinery(chunk_overlap=2, overlap_tokenizer="character")

    chunks = [
        Chunk(text="Hello", start_index=0, end_index=4, token_count=1),
        Chunk(text="World", start_index=5, end_index=9, token_count=1),
    ]

    refinery.refine(chunks)
    refinery.clear_overlap_cache()

    # Should still work after clearing cache
    fresh_chunks = [
        Chunk(text="Foo", start_index=0, end_index=2, token_count=1),
        Chunk(text="Bar", start_index=3, end_index=5, token_count=1),
    ]
    refined = refinery.refine(fresh_chunks)
    assert len(refined) == 2


def test_overlap_pipeline_refinery_class() -> None:
    """Test the _OverlapRefineryRefinery pipeline wrapper class."""
    from chonkie.refinery.overlap import _OverlapRefineryRefinery

    r = _OverlapRefineryRefinery(
        chunk_overlap=3,
        mode="token",
        method="suffix",
        tokenizer="character",
    )

    chunks = [
        Chunk(text="AAABBB", start_index=0, end_index=5, token_count=6),
        Chunk(text="CCCDDD", start_index=6, end_index=11, token_count=6),
    ]

    refined = r.refine(chunks)

    # Suffix takes 3 chars from start of next chunk
    assert refined[0].context == "CCC"
    assert refined[0].text == "AAABBBCCC"


def test_overlap_pipeline_refinery_refine_document() -> None:
    """Test the _OverlapRefineryRefinery.refine_document method."""
    from chonkie.refinery.overlap import _OverlapRefineryRefinery
    from chonkie.types import Document

    r = _OverlapRefineryRefinery(
        chunk_overlap=2,
        mode="token",
        method="suffix",
        tokenizer="character",
    )

    doc = Document(
        content="Hello World",
        chunks=[
            Chunk(text="Hello", start_index=0, end_index=4, token_count=5),
            Chunk(text="World", start_index=5, end_index=9, token_count=5),
        ],
    )

    result = r.refine_document(doc)

    assert result is doc
    assert result.chunks[0].text == "HelloWo"
    assert result.chunks[0].context == "Wo"


def test_overlap_pipeline_refinery_empty_document() -> None:
    """Test refine_document with empty document."""
    from chonkie.refinery.overlap import _OverlapRefineryRefinery
    from chonkie.types import Document

    r = _OverlapRefineryRefinery(chunk_overlap=2)
    doc = Document(content="", chunks=[])

    result = r.refine_document(doc)
    assert result is doc
    assert result.chunks == []
