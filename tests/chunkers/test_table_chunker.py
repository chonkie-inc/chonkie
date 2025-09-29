"""Tests for the TableChunker class."""

from __future__ import annotations

import pytest

from chonkie import Chunk, TableChunker


@pytest.fixture
def sample_table() -> str:
    """Fixture that returns a sample markdown table for testing."""
    table = """| Name | Age | City | Country | Occupation |
|------|-----|------|---------|------------|
| John | 25 | New York | USA | Engineer |
| Alice | 30 | London | UK | Designer |
| Bob | 35 | Paris | France | Manager |
| Carol | 28 | Tokyo | Japan | Developer |
| David | 40 | Berlin | Germany | Architect |
| Eva | 32 | Sydney | Australia | Analyst |
| Frank | 45 | Toronto | Canada | Consultant |
| Grace | 29 | Rome | Italy | Writer |
| Henry | 38 | Madrid | Spain | Teacher |
| Iris | 33 | Amsterdam | Netherlands | Researcher |"""
    return table


@pytest.fixture
def large_table() -> str:
    """Fixture that returns a large markdown table that should be chunked."""
    header = """| ID | First Name | Last Name | Email | Phone | Address | City | State | ZIP | Country | Department | Position | Salary | Start Date |
|-----|------------|-----------|-------|-------|---------|------|-------|-----|---------|------------|----------|--------|------------|"""

    rows = []
    for i in range(20):
        rows.append(f"| {i+1:03d} | Person{i+1} | Lastname{i+1} | person{i+1}@email.com | 555-{i+1:04d} | {i+1} Main St | City{i+1} | ST | {10000+i} | Country{i+1} | Dept{i+1} | Position{i+1} | ${50000+i*1000} | 2023-01-{(i%28)+1:02d} |")

    return header + "\n" + "\n".join(rows)


def test_table_chunker_initialization() -> None:
    """Test that the TableChunker can be initialized with default parameters."""
    chunker = TableChunker()

    assert chunker is not None
    assert chunker.chunk_size == 2048
    assert hasattr(chunker, 'tokenizer')


def test_table_chunker_initialization_with_params() -> None:
    """Test that the TableChunker can be initialized with custom parameters."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    assert chunker is not None
    assert chunker.chunk_size == 500


def test_table_chunker_invalid_chunk_size() -> None:
    """Test that the TableChunker raises an error for invalid chunk size."""
    with pytest.raises(ValueError, match="Chunk size must be greater than 0"):
        TableChunker(chunk_size=0)

    with pytest.raises(ValueError, match="Chunk size must be greater than 0"):
        TableChunker(chunk_size=-1)


def test_table_chunker_small_table(sample_table: str) -> None:
    """Test that a small table returns a single chunk."""
    chunker = TableChunker(tokenizer="character", chunk_size=2048)
    chunks = chunker.chunk(sample_table)

    assert len(chunks) == 1
    assert chunks[0].text == sample_table
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == len(sample_table)
    assert chunks[0].token_count == len(sample_table)


def test_table_chunker_large_table(large_table: str) -> None:
    """Test that a large table gets chunked into multiple pieces."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)
    chunks = chunker.chunk(large_table)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 500 for chunk in chunks)

    # Verify all chunks have the header
    header_lines = large_table.split('\n')[:2]
    expected_header = '\n'.join(header_lines)

    for chunk in chunks:
        chunk_lines = chunk.text.split('\n')
        actual_header = '\n'.join(chunk_lines[:2])
        assert actual_header == expected_header, f"Chunk missing proper header: {chunk.text[:100]}..."


def test_table_chunker_index_calculation(large_table: str) -> None:
    """Test that index calculations are correct when headers are added to chunks."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)
    chunks = chunker.chunk(large_table)

    # Verify indices are sequential and non-overlapping (except for headers)
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        chunks[i + 1]

        # For all chunks after the first, the start should be where previous ended
        if i > 0:
            prev_chunk = chunks[i - 1]
            assert current_chunk.start_index == prev_chunk.end_index

        # Verify we can extract meaningful text from the original using indices
        if i == 0:
            # First chunk should start at 0
            assert current_chunk.start_index == 0

        # Check that end index is reasonable
        assert current_chunk.end_index > current_chunk.start_index


def test_table_chunker_preserves_content() -> None:
    """Test that all original data rows are preserved across chunks."""
    table = """| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |
| C | 3 |
| D | 4 |
| E | 5 |"""

    chunker = TableChunker(tokenizer="character", chunk_size=50)  # Force chunking
    chunks = chunker.chunk(table)

    # Extract all data rows from chunks (skip header rows)
    all_data_rows = []
    for chunk in chunks:
        lines = chunk.text.split('\n')
        data_lines = lines[2:]  # Skip header and separator
        # Filter out empty strings that come from trailing newlines
        data_lines = [line for line in data_lines if line.strip()]
        all_data_rows.extend(data_lines)

    # Get original data rows
    original_lines = table.split('\n')
    original_data = original_lines[2:]

    # Should have same data rows (accounting for duplicates from multiple chunks)
    unique_data_rows = list(dict.fromkeys(all_data_rows))  # Remove duplicates, preserve order
    assert set(unique_data_rows) == set(original_data)


def test_table_chunker_invalid_table() -> None:
    """Test that the TableChunker handles invalid tables appropriately."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    # Table with no rows (just header)
    invalid_table = """| Name | Value |
|------|-------|"""

    with pytest.raises(ValueError, match="Table must have at least a header and one row"):
        chunker.chunk(invalid_table)

    # Single line (no table structure)
    with pytest.raises(ValueError, match="Table must have at least a header and one row"):
        chunker.chunk("Just a single line")


def test_table_chunker_empty_input() -> None:
    """Test that the TableChunker handles empty input."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    with pytest.raises(ValueError, match="Table must have at least a header and one row"):
        chunker.chunk("")


def test_table_chunker_exact_chunk_size() -> None:
    """Test table chunking when rows exactly fit the chunk size."""
    # Create a table where each row is exactly a known size
    table = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |"""

    # Set chunk size to fit header + one row
    header_size = len("| A | B |\n|---|---|")
    row_size = len("\n| 1 | 2 |")
    chunk_size = header_size + row_size

    chunker = TableChunker(tokenizer="character", chunk_size=chunk_size)
    chunks = chunker.chunk(table)

    # Should create multiple chunks since we have 2 data rows
    assert len(chunks) >= 2

    # Each chunk should have the header
    for chunk in chunks:
        assert "| A | B |" in chunk.text
        assert "|---|---|" in chunk.text


def verify_chunk_indices(chunks: list[Chunk], original_text: str) -> None:
    """Verify that chunk indices correctly represent positions in original text."""
    # For table chunker, we need to account for the fact that indices represent
    # logical positions in the original table, not literal string positions
    # since headers are repeated in each chunk

    # Basic sanity checks
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index > chunk.start_index for chunk in chunks)

    # Indices should be increasing (non-overlapping content)
    for i in range(len(chunks) - 1):
        assert chunks[i].end_index <= chunks[i + 1].start_index


def test_table_chunker_indices_consistency(large_table: str) -> None:
    """Test that TableChunker's indices are consistent and reasonable."""
    chunker = TableChunker(tokenizer="character", chunk_size=400)
    chunks = chunker.chunk(large_table)

    verify_chunk_indices(chunks, large_table)


def test_table_chunker_call_method(sample_table: str) -> None:
    """Test that the TableChunker can be called directly."""
    chunker = TableChunker(tokenizer="character", chunk_size=2048)
    chunks = chunker(sample_table)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == sample_table


def test_table_chunker_repr() -> None:
    """Test that the TableChunker has a string representation."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    repr_str = repr(chunker)
    assert "TableChunker" in repr_str
    assert "500" in repr_str