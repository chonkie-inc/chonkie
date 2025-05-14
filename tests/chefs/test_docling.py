"""Tests for DoclingChef."""

import os
import pytest
from unittest.mock import patch, MagicMock

from chonkie.chefs import (
    DoclingChef,
    ProcessingResult,
    ProcessingStatus,
    ContentExtractionError,
)


@pytest.fixture
def test_doc():
    """Create a test documentation file."""
    content = """# Test Documentation

This is a test documentation file.

## Features
- Feature 1
- Feature 2

```python
def hello():
    print("Hello, World!")
```

## Usage
Here's how to use it:

```python
from test import hello
hello()
```
"""
    file_path = "test.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def test_rst():
    """Create a test reStructuredText file."""
    content = """Test Documentation
================

This is a test documentation file.

Features
--------
- Feature 1
- Feature 2

Usage
-----
Here's how to use it::

    from test import hello
    hello()
"""
    file_path = "test.rst"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    os.remove(file_path)


def test_docling_chef_initialization():
    """Test DoclingChef initialization."""
    chef = DoclingChef()
    assert chef.name == "DoclingChef"
    assert chef.version == "1.0.0"
    assert chef.supported_formats == ["md", "markdown", "rst", "txt"]


def test_docling_chef_missing_dependencies():
    """Test behavior when docutils dependencies are missing."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError) as exc_info:
            DoclingChef()
        assert "Dependencies not found" in str(exc_info.value)


def test_docling_chef_process_success(test_doc):
    """Test successful documentation processing."""
    chef = DoclingChef()
    result = chef.process(test_doc)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert result.document.text is not None
    assert "sections" in result.document.metadata
    assert "code_blocks" in result.document.metadata
    assert result.metadata["processing_engine"] == "docling"
    assert result.metadata["file_path"] == test_doc
    assert result.metadata["sections_count"] > 0
    assert result.metadata["code_blocks_count"] > 0


def test_docling_chef_process_rst(test_rst):
    """Test processing reStructuredText file."""
    chef = DoclingChef()
    result = chef.process(test_rst)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert "html_content" in result.document.metadata
    assert result.metadata["processing_engine"] == "docling"


def test_docling_chef_process_failure():
    """Test documentation processing failure."""
    chef = DoclingChef()
    result = chef.process("nonexistent.md")

    assert result.status == ProcessingStatus.FAILED
    assert "Invalid documentation file" in result.error


def test_docling_chef_invalid_file():
    """Test processing invalid file."""
    # Create a non-documentation file
    file_path = "test.py"
    with open(file_path, "w") as f:
        f.write("print('Hello, World!')")
    
    chef = DoclingChef()
    result = chef.process(file_path)
    
    assert result.status == ProcessingStatus.FAILED
    assert "Invalid documentation file" in result.error
    
    os.remove(file_path)


def test_docling_chef_content_extraction_error(test_rst):
    """Test content extraction error handling."""
    chef = DoclingChef()
    
    # Mock publish_parts to raise an exception
    with patch.object(chef, "publish_parts", side_effect=Exception("Test error")):
        with pytest.raises(ContentExtractionError) as exc_info:
            chef.process(test_rst)
        assert "Failed to process documentation file" in str(exc_info.value)


def test_docling_chef_section_extraction(test_doc):
    """Test section extraction functionality."""
    chef = DoclingChef()
    result = chef.process(test_doc)
    
    sections = result.document.metadata["sections"]
    assert len(sections) > 0
    
    # Check first section
    assert sections[0]["title"] == "Test Documentation"
    assert sections[0]["level"] == 1
    
    # Check subsection
    assert any(section["title"] == "Features" and section["level"] == 2 
              for section in sections)


def test_docling_chef_code_block_extraction(test_doc):
    """Test code block extraction functionality."""
    chef = DoclingChef()
    result = chef.process(test_doc)
    
    code_blocks = result.document.metadata["code_blocks"]
    assert len(code_blocks) > 0
    
    # Check first code block
    assert code_blocks[0]["language"] == "python"
    assert "def hello()" in code_blocks[0]["code"] 