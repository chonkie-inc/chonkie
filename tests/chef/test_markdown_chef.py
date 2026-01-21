"""Tests for the MarkdownChef class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from chonkie.chef import MarkdownChef
from chonkie.types import MarkdownDocument


class TestMarkdownChef:
    """Test suite for MarkdownChef class."""

    @pytest.fixture
    def markdown_chef(self) -> MarkdownChef:
        """Fixture that returns a MarkdownChef instance."""
        return MarkdownChef()

    @pytest.fixture
    def simple_markdown(self) -> str:
        """Fixture that returns simple markdown text."""
        return """# Heading

This is a simple paragraph."""

    @pytest.fixture
    def markdown_with_table(self) -> str:
        """Fixture that returns markdown with a table."""
        return """# Document

Some text before the table.

| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |

Some text after the table."""

    @pytest.fixture
    def markdown_with_code(self) -> str:
        """Fixture that returns markdown with code blocks."""
        return """# Code Example

Here's some Python code:

```python
def hello():
    print("Hello, World!")
```

And some JavaScript:

```javascript
console.log("Hello");
```"""

    @pytest.fixture
    def markdown_with_images(self) -> str:
        """Fixture that returns markdown with images."""
        return """# Images

Here's an image:

![Alt text](path/to/image.png)

Another image with link:

[![Linked image](image2.jpg)](https://example.com)"""

    @pytest.fixture
    def complex_markdown(self) -> str:
        """Fixture that returns complex markdown with tables, code, and images."""
        return """# Complex Document

Introduction paragraph.

## Data Table

| ID | Name | Value |
|----|------|-------|
| 1  | A    | 100   |
| 2  | B    | 200   |

## Code Sample

```python
x = 42
```

## Image

![Chart](chart.png)

Conclusion text."""

    # ==================== Tests for parse() method ====================

    def test_parse_simple_markdown(
        self,
        markdown_chef: MarkdownChef,
        simple_markdown: str,
    ) -> None:
        """Test parsing simple markdown text."""
        result = markdown_chef.parse(simple_markdown)

        assert isinstance(result, MarkdownDocument)
        assert result.content == simple_markdown
        assert len(result.chunks) >= 1

    def test_parse_markdown_with_table(
        self,
        markdown_chef: MarkdownChef,
        markdown_with_table: str,
    ) -> None:
        """Test parsing markdown with a table."""
        result = markdown_chef.parse(markdown_with_table)

        assert isinstance(result, MarkdownDocument)
        assert result.content == markdown_with_table
        assert len(result.tables) == 1
        assert "| Name | Age | City |" in result.tables[0].content
        assert result.tables[0].start_index >= 0
        assert result.tables[0].end_index > result.tables[0].start_index

    def test_parse_markdown_with_code(
        self,
        markdown_chef: MarkdownChef,
        markdown_with_code: str,
    ) -> None:
        """Test parsing markdown with code blocks."""
        result = markdown_chef.parse(markdown_with_code)

        assert isinstance(result, MarkdownDocument)
        assert len(result.code) == 2
        assert result.code[0].language == "python"
        assert 'print("Hello, World!")' in result.code[0].content
        assert result.code[1].language == "javascript"
        assert 'console.log("Hello")' in result.code[1].content

    def test_parse_markdown_with_images(
        self,
        markdown_chef: MarkdownChef,
        markdown_with_images: str,
    ) -> None:
        """Test parsing markdown with images."""
        result = markdown_chef.parse(markdown_with_images)

        assert isinstance(result, MarkdownDocument)
        assert len(result.images) == 2
        assert result.images[0].alias == "Alt text"
        assert result.images[0].content == "path/to/image.png"
        assert result.images[1].content == "image2.jpg"
        assert result.images[1].link == "https://example.com"

    def test_parse_complex_markdown(
        self,
        markdown_chef: MarkdownChef,
        complex_markdown: str,
    ) -> None:
        """Test parsing complex markdown with multiple elements."""
        result = markdown_chef.parse(complex_markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) == 1
        assert len(result.code) == 1
        assert len(result.images) == 1
        assert len(result.chunks) >= 1

    def test_parse_empty_string(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing empty string."""
        result = markdown_chef.parse("")

        assert isinstance(result, MarkdownDocument)
        assert result.content == ""
        assert len(result.tables) == 0
        assert len(result.code) == 0
        assert len(result.images) == 0

    def test_parse_whitespace_only(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing whitespace-only string."""
        whitespace = "   \n\n   \t  \n"
        result = markdown_chef.parse(whitespace)

        assert isinstance(result, MarkdownDocument)
        assert result.content == whitespace

    def test_parse_markdown_with_unicode(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown with unicode characters."""
        unicode_md = """# 日本語

これはテストです。

| 名前 | 年齢 |
|------|------|
| 太郎 | 25   |

🎉 Emoji support!"""

        result = markdown_chef.parse(unicode_md)

        assert isinstance(result, MarkdownDocument)
        assert "日本語" in result.content
        assert "🎉" in result.content
        assert len(result.tables) == 1

    def test_parse_multiple_tables(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown with multiple tables."""
        markdown = """# Multiple Tables

First table:

| A | B |
|---|---|
| 1 | 2 |

Second table:

| X | Y | Z |
|---|---|---|
| 3 | 4 | 5 |

Third table:

| Col1 |
|------|
| Data |"""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) == 3

    def test_parse_multiple_code_blocks(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown with multiple code blocks."""
        markdown = """# Code Examples

```python
x = 1
```

```java
int y = 2;
```

```
plain code
```"""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.code) == 3
        assert result.code[0].language == "python"
        assert result.code[1].language == "java"
        assert result.code[2].language is None

    def test_parse_code_without_language(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing code block without language specification."""
        markdown = """```
code without language
```"""

        result = markdown_chef.parse(markdown)

        assert len(result.code) == 1
        assert result.code[0].language is None
        assert "code without language" in result.code[0].content

    def test_parse_malformed_table(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing malformed tables."""
        markdown = """# Document

| A | B |
| 1 | 2 |

This is not a valid table."""

        result = markdown_chef.parse(markdown)

        # Should not crash, but may not extract the malformed table
        assert isinstance(result, MarkdownDocument)

    def test_parse_nested_code_in_table(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown with code-like content in tables."""
        markdown = """| Code | Description |
|------|-------------|
| `x = 1` | Variable |"""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) == 1

    def test_parse_indices_consistency(
        self,
        markdown_chef: MarkdownChef,
        complex_markdown: str,
    ) -> None:
        """Test that all extracted elements have valid indices."""
        result = markdown_chef.parse(complex_markdown)

        # Check tables
        for table in result.tables:
            assert 0 <= table.start_index < len(result.content)
            assert table.start_index < table.end_index <= len(result.content)
            assert result.content[table.start_index : table.end_index] == table.content

        # Check code
        for code in result.code:
            assert 0 <= code.start_index < len(result.content)
            assert code.start_index < code.end_index <= len(result.content)

        # Check images
        for image in result.images:
            assert 0 <= image.start_index < len(result.content)
            assert image.start_index < image.end_index <= len(result.content)

        # Check chunks
        for chunk in result.chunks:
            assert 0 <= chunk.start_index < len(result.content)
            assert chunk.start_index < chunk.end_index <= len(result.content)

    # ==================== Tests for process() method ====================

    def test_process_file(self, markdown_chef: MarkdownChef, simple_markdown: str) -> None:
        """Test processing a markdown file."""
        with patch("builtins.open", mock_open(read_data=simple_markdown)):
            result = markdown_chef.process("test.md")

            assert isinstance(result, MarkdownDocument)
            assert result.content == simple_markdown

    def test_process_file_with_path_object(
        self,
        markdown_chef: MarkdownChef,
        simple_markdown: str,
    ) -> None:
        """Test processing a file with Path object."""
        path_obj = Path("test.md")
        with patch("builtins.open", mock_open(read_data=simple_markdown)):
            result = markdown_chef.process(path_obj)

            assert isinstance(result, MarkdownDocument)
            assert result.content == simple_markdown

    def test_process_calls_parse(self, markdown_chef: MarkdownChef, simple_markdown: str) -> None:
        """Test that process() calls parse() internally."""
        with patch("builtins.open", mock_open(read_data=simple_markdown)):
            with patch.object(markdown_chef, "parse", wraps=markdown_chef.parse) as mock_parse:
                result = markdown_chef.process("test.md")

                mock_parse.assert_called_once_with(simple_markdown)
                assert isinstance(result, MarkdownDocument)

    def test_process_file_not_found(self, markdown_chef: MarkdownChef) -> None:
        """Test handling of FileNotFoundError."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                markdown_chef.process("nonexistent.md")

    def test_process_batch(self, markdown_chef: MarkdownChef, simple_markdown: str) -> None:
        """Test batch processing of multiple markdown files."""
        paths = ["file1.md", "file2.md", "file3.md"]
        with patch("builtins.open", mock_open(read_data=simple_markdown)):
            results = markdown_chef.process_batch(paths)

            assert len(results) == 3
            assert all(isinstance(r, MarkdownDocument) for r in results)
            assert all(r.content == simple_markdown for r in results)

    # ==================== Edge Cases ====================

    def test_parse_very_long_markdown(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing very long markdown document."""
        long_markdown = "# Header\n\n" + ("This is a paragraph. " * 1000)
        result = markdown_chef.parse(long_markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.content) > 20000

    def test_parse_markdown_with_special_characters(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown with special characters."""
        markdown = """# Special Characters

< > & " ' ` @ # $ % ^ * ( ) [ ] { } | \\ / ? ! ~ + = - _ ;"""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert "<" in result.content
        assert "&" in result.content

    def test_parse_markdown_with_html(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing markdown containing HTML."""
        markdown = """# Document

<div class="container">
    <p>HTML content</p>
</div>

Regular markdown text."""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert "<div" in result.content

    def test_parse_table_with_alignment(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing table with column alignment."""
        markdown = """| Left | Center | Right |
|:-----|:------:|------:|
| L    | C      | R     |"""

        result = markdown_chef.parse(markdown)

        assert len(result.tables) == 1
        assert ":---" in result.tables[0].content or "|:-----|" in result.tables[0].content

    def test_call_method(self, markdown_chef: MarkdownChef, simple_markdown: str) -> None:
        """Test that __call__ method works."""
        with patch("builtins.open", mock_open(read_data=simple_markdown)):
            result = markdown_chef("test.md")

            assert isinstance(result, MarkdownDocument)
            assert result.content == simple_markdown

    def test_repr(self, markdown_chef: MarkdownChef) -> None:
        """Test __repr__ method."""
        repr_str = repr(markdown_chef)
        assert "MarkdownChef" in repr_str

    # ==================== Integration Tests ====================

    def test_parse_then_chunker_integration(self, markdown_chef: MarkdownChef) -> None:
        """Test that parsed MarkdownDocument can be used with chunkers."""
        markdown = """# Document

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |

Some text content."""

        result = markdown_chef.parse(markdown)

        # Verify structure is ready for chunker processing
        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) > 0
        assert all(hasattr(t, "start_index") and hasattr(t, "end_index") for t in result.tables)

    def test_parse_real_world_markdown(self, markdown_chef: MarkdownChef) -> None:
        """Test parsing realistic markdown document."""
        markdown = """# Project README

## Overview

This is a sample project.

## Installation

```bash
pip install package
```

## Usage

Basic usage:

```python
from package import Module

module = Module()
module.run()
```

## Data

| Feature | Support |
|---------|---------|
| Tables  | ✓       |
| Code    | ✓       |
| Images  | ✓       |

## License

MIT License

![Badge](https://img.shields.io/badge/status-active-green)"""

        result = markdown_chef.parse(markdown)

        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) >= 1
        assert len(result.code) >= 2
        assert len(result.images) >= 1
        assert len(result.chunks) >= 1
