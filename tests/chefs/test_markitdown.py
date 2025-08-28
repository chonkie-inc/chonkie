"""Tests for MarkitdownChef."""

import os
import pytest
from unittest.mock import patch, MagicMock

from chonkie.chefs import (
    MarkitdownChef,
    ProcessingResult,
    ProcessingStatus,
    ContentExtractionError,
)


@pytest.fixture
def test_markdown():
    """Create a test markdown file."""
    content = """# Test Markdown

This is a test markdown file.

## Features
- Feature 1
- Feature 2

```python
def hello():
    print("Hello, World!")
```
"""
    file_path = "test.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    os.remove(file_path)


def test_markitdown_chef_initialization():
    """Test MarkitdownChef initialization."""
    chef = MarkitdownChef()
    assert chef.name == "MarkitdownChef"
    assert chef.version == "1.0.0"
    assert chef.supported_formats == ["md", "markdown"]


def test_markitdown_chef_missing_dependencies():
    """Test behavior when markdown dependencies are missing."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError) as exc_info:
            MarkitdownChef()
        assert "Markdown dependencies not found" in str(exc_info.value)


def test_markitdown_chef_process_success(test_markdown):
    """Test successful markdown processing."""
    chef = MarkitdownChef()
    result = chef.process(test_markdown)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert result.document.text is not None
    assert "html_content" in result.document.metadata
    assert result.metadata["processing_engine"] == "markdown"
    assert result.metadata["file_path"] == test_markdown


def test_markitdown_chef_process_failure():
    """Test markdown processing failure."""
    chef = MarkitdownChef()
    result = chef.process("nonexistent.md")

    assert result.status == ProcessingStatus.FAILED
    assert "Invalid markdown file" in result.error


def test_markitdown_chef_invalid_file():
    """Test processing invalid file."""
    # Create a non-markdown file
    file_path = "test.txt"
    with open(file_path, "w") as f:
        f.write("This is not markdown")
    
    chef = MarkitdownChef()
    result = chef.process(file_path)
    
    assert result.status == ProcessingStatus.FAILED
    assert "Invalid markdown file" in result.error
    
    os.remove(file_path)


def test_markitdown_chef_content_extraction_error(test_markdown):
    """Test content extraction error handling."""
    chef = MarkitdownChef()
    
    # Mock markdown.markdown to raise an exception
    with patch.object(chef.markdown, "markdown", side_effect=Exception("Test error")):
        with pytest.raises(ContentExtractionError) as exc_info:
            chef.process(test_markdown)
        assert "Failed to process markdown file" in str(exc_info.value) 