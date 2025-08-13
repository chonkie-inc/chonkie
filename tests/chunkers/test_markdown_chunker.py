"""Tests for the MarkdownChunker class."""

from __future__ import annotations

import re
from typing import List

import pytest

from chonkie import Chunk, MarkdownChunker


@pytest.fixture
def sample_markdown_text() -> str:
    """Return a sample markdown text with headings and content."""
    text = """# Heading 1

Intro paragraph with some **bold** and *italic* text and `inline code`.

## Subheading A
Some details with a [link](https://example.com).

### Sub-subheading A.1
More details here.

## Subheading B
Final section with more content.
"""
    return text


@pytest.fixture
def sample_complex_markdown_text() -> str:
    """Return a complex markdown text containing lists and code blocks."""
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


def verify_chunk_indices(chunks: List[Chunk], original_text: str) -> None:
    """Verify that chunk indices correctly map to the original text."""
    reconstructed = ""
    for i, chunk in enumerate(chunks):
        extracted = original_text[chunk.start_index : chunk.end_index]
        assert chunk.text == extracted, (
            f"Chunk {i} text mismatch\n"
            f"Chunk text: {repr(chunk.text)}\n"
            f"Slice text: {repr(extracted)}\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )
        reconstructed += chunk.text
    assert reconstructed == original_text


def test_markdown_chunker_initialization() -> None:
    """Test MarkdownChunker initialization parameters."""
    chunker = MarkdownChunker(
        tokenizer_or_token_counter="character",
        chunk_size=256,
        heading_level=3,
        min_characters_per_chunk=32,
        max_characters_per_section=1000,
        clean_text=False,
    )
    assert chunker is not None
    assert chunker.chunk_size == 256
    assert chunker.heading_level == 3
    assert chunker.min_characters_per_chunk == 32
    assert chunker.max_characters_per_section == 1000
    assert chunker.clean_text is False


def test_markdown_chunker_chunking_basic(sample_markdown_text: str) -> None:
    """Test basic chunking of markdown content."""
    chunker = MarkdownChunker(chunk_size=256)
    chunks = chunker.chunk(sample_markdown_text)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(isinstance(c.text, str) and len(c.text) > 0 for c in chunks)
    assert all(isinstance(c.start_index, int) and c.start_index >= 0 for c in chunks)
    assert all(isinstance(c.end_index, int) and c.end_index > c.start_index for c in chunks)
    assert all(isinstance(c.token_count, int) and c.token_count > 0 for c in chunks)


def test_markdown_chunker_reconstruction_no_cleaning(sample_complex_markdown_text: str) -> None:
    """Without cleaning, concatenating chunks should reconstruct the original text."""
    chunker = MarkdownChunker(chunk_size=200, clean_text=False)
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


def test_markdown_chunker_clean_text_removes_markdown(sample_markdown_text: str) -> None:
    """With cleaning enabled, inline markdown and URLs should be removed from chunks."""
    chunker = MarkdownChunker(chunk_size=256, clean_text=True)
    chunks = chunker.chunk(sample_markdown_text)

    assert len(chunks) > 0
    combined = "\n".join(c.text for c in chunks)
    # No inline link syntax or obvious markdown emphasis/code remains
    assert "[" not in combined and "](" not in combined
    assert "**" not in combined and "`" not in combined


def test_markdown_chunker_respects_chunk_size(sample_complex_markdown_text: str) -> None:
    """Chunks should respect the configured chunk_size limit (character tokenizer)."""
    chunk_size = 120
    chunker = MarkdownChunker(chunk_size=chunk_size)
    chunks = chunker.chunk(sample_complex_markdown_text)
    assert len(chunks) > 0
    assert all(c.token_count <= chunk_size for c in chunks)


def test_markdown_chunker_empty_text() -> None:
    """Empty input should yield no chunks."""
    chunker = MarkdownChunker()
    assert chunker.chunk("") == []


def test_markdown_chunker_from_recipe_default() -> None:
    """Test that from_recipe returns a configured instance."""
    chunker = MarkdownChunker.from_recipe()
    assert isinstance(chunker, MarkdownChunker)
    assert chunker.chunk_size == 1000


def test_markdown_chunker_repr() -> None:
    """Test the string representation lists key parameters."""
    chunker = MarkdownChunker(chunk_size=300, min_characters_per_chunk=40, clean_text=True)
    rep = repr(chunker)
    assert "MarkdownChunker" in rep
    assert "chunk_size=300" in rep
    assert "min_characters_per_chunk=40" in rep
    assert "clean_text=True" in rep



