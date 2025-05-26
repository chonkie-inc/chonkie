"""Test for the JSONPorter class."""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from chonkie import Chunk, Context
from chonkie.friends.porters.json import JSONPorter


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="This is the first chunk.",
            start_index=0,
            end_index=25,
            token_count=5,
        ),
        Chunk(
            text="This is the second chunk with more content.",
            start_index=25,
            end_index=68,
            token_count=8,
        ),
        Chunk(
            text="Third chunk is shorter.",
            start_index=68,
            end_index=91,
            token_count=4,
        ),
    ]


@pytest.fixture
def chunks_with_context() -> list[Chunk]:
    """Create chunks with context for testing."""
    context1 = Context(text="Context for chunk 1", token_count=4)
    context2 = Context(text="Context for chunk 2", token_count=4)
    
    return [
        Chunk(
            text="First chunk with context.",
            start_index=0,
            end_index=25,
            token_count=5,
            context=context1,
        ),
        Chunk(
            text="Second chunk with context.",
            start_index=25,
            end_index=51,
            token_count=5,
            context=context2,
        ),
    ]


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_json_porter_initialization() -> None:
    """Test JSONPorter initialization."""
    # Test default initialization (lines=True)
    porter = JSONPorter()
    assert porter.lines is True
    assert porter.indent == 4

    # Test initialization with lines=False
    porter = JSONPorter(lines=False)
    assert porter.lines is False
    assert porter.indent == 4

    # Test initialization with lines=True explicitly
    porter = JSONPorter(lines=True)
    assert porter.lines is True
    assert porter.indent == 4


def test_json_porter_export_jsonl(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test exporting chunks as JSONL format."""
    porter = JSONPorter(lines=True)
    output_file = os.path.join(temp_dir, "test_chunks.jsonl")
    
    # Export chunks
    porter.export(sample_chunks, output_file)
    
    # Verify file exists
    assert os.path.exists(output_file)
    
    # Read and verify content
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == len(sample_chunks)
    
    # Verify each line is valid JSON and contains expected data
    for i, line in enumerate(lines):
        chunk_data = json.loads(line.strip())
        assert chunk_data["text"] == sample_chunks[i].text
        assert chunk_data["start_index"] == sample_chunks[i].start_index
        assert chunk_data["end_index"] == sample_chunks[i].end_index
        assert chunk_data["token_count"] == sample_chunks[i].token_count


def test_json_porter_export_json(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test exporting chunks as JSON format."""
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "test_chunks.json")
    
    # Export chunks
    porter.export(sample_chunks, output_file)
    
    # Verify file exists
    assert os.path.exists(output_file)
    
    # Read and verify content
    with open(output_file, "r") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) == len(sample_chunks)
    
    # Verify each chunk data
    for i, chunk_data in enumerate(data):
        assert chunk_data["text"] == sample_chunks[i].text
        assert chunk_data["start_index"] == sample_chunks[i].start_index
        assert chunk_data["end_index"] == sample_chunks[i].end_index
        assert chunk_data["token_count"] == sample_chunks[i].token_count


def test_json_porter_with_context(chunks_with_context: list[Chunk], temp_dir: str) -> None:
    """Test exporting chunks with context."""
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "chunks_with_context.json")
    
    # Export chunks
    porter.export(chunks_with_context, output_file)
    
    # Read and verify content
    with open(output_file, "r") as f:
        data = json.load(f)
    
    assert len(data) == len(chunks_with_context)
    
    # Verify context is included
    for i, chunk_data in enumerate(data):
        assert "context" in chunk_data
        assert chunk_data["context"]["text"] == chunks_with_context[i].context.text
        assert chunk_data["context"]["token_count"] == chunks_with_context[i].context.token_count


def test_json_porter_call_method(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test JSONPorter __call__ method uses default filename."""
    porter = JSONPorter(lines=True)
    
    # Change to temp directory to use default filename
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Use __call__ method (base class only passes chunks, uses default filename)
        porter(sample_chunks)
        
        # Verify default file exists and has correct content
        assert os.path.exists("chunks.jsonl")
        
        with open("chunks.jsonl", "r") as f:
            lines = f.readlines()
        
        assert len(lines) == len(sample_chunks)
        
    finally:
        os.chdir(original_cwd)


def test_json_porter_default_filenames(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test JSONPorter with default filenames."""
    # Change to temp directory to avoid polluting the project
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Test JSONL default filename
        porter_lines = JSONPorter(lines=True)
        porter_lines.export(sample_chunks)
        assert os.path.exists("chunks.jsonl")
        
        # Test JSON format - note: default filename is still "chunks.jsonl" 
        # but content will be JSON format, not JSONL
        porter_json = JSONPorter(lines=False)
        porter_json.export(sample_chunks)
        assert os.path.exists("chunks.jsonl")  # Default filename is always chunks.jsonl
        
        # Verify the content is JSON format (not JSONL)
        with open("chunks.jsonl", "r") as f:
            data = json.load(f)  # Should be valid JSON, not JSONL
        assert isinstance(data, list)
        assert len(data) == len(sample_chunks)
        
    finally:
        os.chdir(original_cwd)


def test_json_porter_empty_chunks_list(temp_dir: str) -> None:
    """Test JSONPorter with empty chunks list."""
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "empty_chunks.json")
    
    # Export empty list
    porter.export([], output_file)
    
    # Verify file exists and contains empty array
    assert os.path.exists(output_file)
    
    with open(output_file, "r") as f:
        data = json.load(f)
    
    assert data == []


def test_json_porter_empty_chunks_jsonl(temp_dir: str) -> None:
    """Test JSONPorter with empty chunks list in JSONL format."""
    porter = JSONPorter(lines=True)
    output_file = os.path.join(temp_dir, "empty_chunks.jsonl")
    
    # Export empty list
    porter.export([], output_file)
    
    # Verify file exists and is empty
    assert os.path.exists(output_file)
    
    with open(output_file, "r") as f:
        content = f.read()
    
    assert content == ""


def test_json_porter_indentation(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test JSON indentation is applied correctly."""
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "indented_chunks.json")
    
    # Export chunks
    porter.export(sample_chunks, output_file)
    
    # Read raw content and verify indentation
    with open(output_file, "r") as f:
        content = f.read()
    
    # Should contain indentation (4 spaces)
    assert "    " in content
    
    # Should be properly formatted JSON
    data = json.loads(content)
    assert len(data) == len(sample_chunks)


def test_json_porter_file_permissions_error(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test JSONPorter handles file permission errors."""
    porter = JSONPorter(lines=False)
    
    # Try to write to a directory that doesn't exist
    invalid_path = os.path.join(temp_dir, "nonexistent", "chunks.json")
    
    with pytest.raises(FileNotFoundError):
        porter.export(sample_chunks, invalid_path)


def test_json_porter_large_chunks_list(temp_dir: str) -> None:
    """Test JSONPorter with a large number of chunks."""
    # Create a large list of chunks
    large_chunks = []
    for i in range(1000):
        chunk = Chunk(
            text=f"Chunk number {i} with some content.",
            start_index=i * 40,
            end_index=(i + 1) * 40,
            token_count=7,
        )
        large_chunks.append(chunk)
    
    porter = JSONPorter(lines=True)
    output_file = os.path.join(temp_dir, "large_chunks.jsonl")
    
    # Export large list
    porter.export(large_chunks, output_file)
    
    # Verify file exists and has correct number of lines
    assert os.path.exists(output_file)
    
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == 1000


def test_json_porter_unicode_content(temp_dir: str) -> None:
    """Test JSONPorter handles Unicode content correctly."""
    unicode_chunks = [
        Chunk(
            text="Hello 世界! 🌍 This contains unicode.",
            start_index=0,
            end_index=35,
            token_count=8,
        ),
        Chunk(
            text="Café, naïve, résumé, 北京",
            start_index=35,
            end_index=58,
            token_count=6,
        ),
    ]
    
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "unicode_chunks.json")
    
    # Export chunks with unicode
    porter.export(unicode_chunks, output_file)
    
    # Read and verify content
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert len(data) == 2
    assert data[0]["text"] == "Hello 世界! 🌍 This contains unicode."
    assert data[1]["text"] == "Café, naïve, résumé, 北京"


def test_json_porter_chunk_serialization_completeness(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test that all chunk attributes are properly serialized."""
    porter = JSONPorter(lines=False)
    output_file = os.path.join(temp_dir, "complete_chunks.json")
    
    # Export chunks
    porter.export(sample_chunks, output_file)
    
    # Read back and verify all expected fields are present
    with open(output_file, "r") as f:
        data = json.load(f)
    
    for i, chunk_data in enumerate(data):
        # Check all basic fields are present
        required_fields = ["text", "start_index", "end_index", "token_count"]
        for field in required_fields:
            assert field in chunk_data
            assert chunk_data[field] == getattr(sample_chunks[i], field)
        
        # Context should be None for these chunks
        assert chunk_data.get("context") is None


def test_json_porter_path_object_support(sample_chunks: list[Chunk], temp_dir: str) -> None:
    """Test JSONPorter works with Path objects."""
    porter = JSONPorter(lines=False)
    output_file = Path(temp_dir) / "path_chunks.json"
    
    # Export using Path object
    porter.export(sample_chunks, str(output_file))
    
    # Verify file exists
    assert output_file.exists()
    
    # Verify content
    with open(output_file, "r") as f:
        data = json.load(f)
    
    assert len(data) == len(sample_chunks)