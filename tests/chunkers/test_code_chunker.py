"""Test the CodeChunker class."""

import pytest

from chonkie import CodeChunker
from chonkie.types import Chunk
from chonkie.types.markdown import MarkdownCode, MarkdownDocument


@pytest.fixture
def python_code() -> str:
    """Return a sample Python code snippet."""
    return """
import os
import sys

def hello_world(name: str):
    \"\"\"Prints a greeting.\"\"\"
    print(f"Hello, {name}!")

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

if __name__ == "__main__":
    hello_world("World")
    instance = MyClass(10)
    print(instance.get_value())
"""


@pytest.fixture
def js_code() -> str:
    """Return a sample JavaScript code snippet."""
    return """
function greet(name) {
  console.log(`Hello, ${name}!`);
}

class Calculator {
  add(a, b) {
    return a + b;
  }
}

const calc = new Calculator();
greet('Developer');
console.log(calc.add(5, 3));
"""


def test_code_chunker_initialization() -> None:
    """Test CodeChunker initialization."""
    chunker = CodeChunker(language="python", chunk_size=128)
    assert chunker.chunk_size == 128
    assert chunker.language == "python"


def test_code_chunker_chunking_python(python_code: str) -> None:
    """Test basic chunking of Python code."""
    chunker = CodeChunker(language="python", chunk_size=50, include_nodes=True)
    chunks = chunker.chunk(python_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.text is not None for chunk in chunks)
    assert all(chunk.start_index is not None for chunk in chunks)
    assert all(chunk.end_index is not None for chunk in chunks)
    assert all(chunk.token_count is not None for chunk in chunks)
    # Note: nodes attribute is no longer part of base Chunk


def test_code_chunker_reconstruction_python(python_code: str) -> None:
    """Test if the original Python code can be reconstructed from chunks."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == python_code


def test_code_chunker_chunk_size_python(python_code: str) -> None:
    """Test if Python code chunks mostly adhere to chunk_size."""
    chunk_size = 50
    chunker = CodeChunker(language="python", chunk_size=chunk_size)
    chunks = chunker.chunk(python_code)
    # Allow for some leeway as splitting happens at node boundaries
    assert all(
        chunk.token_count < chunk_size + 20 for chunk in chunks[:-1]
    )  # Check all but last chunk rigorously
    assert chunks[-1].token_count > 0  # Last chunk must have content


def test_code_chunker_indices_python(python_code: str) -> None:
    """Test the start and end indices of Python code chunks."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    current_index = 0
    for chunk in chunks:
        assert chunk.start_index == current_index
        assert chunk.end_index == current_index + len(chunk.text)
        assert chunk.text == python_code[chunk.start_index : chunk.end_index]
        current_index = chunk.end_index
    assert current_index == len(python_code)


def test_code_chunker_return_type_chunks(python_code: str) -> None:
    """Test that chunker returns Chunk objects."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == python_code


def test_code_chunker_empty_input() -> None:
    """Test chunking an empty string."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("")
    assert chunks == []

    # Test with default chunker (returns chunks)
    chunker_default = CodeChunker(language="python")
    chunks_default = chunker_default.chunk("")
    assert chunks_default == []


def test_code_chunker_whitespace_input() -> None:
    """Test chunking a string with only whitespace."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("   \n\t\n  ")
    assert chunks == []

    # Test with default chunker (returns chunks)
    chunker_default = CodeChunker(language="python")
    chunks_default = chunker_default.chunk("   \n\t\n  ")
    assert chunks_default == []


def test_code_chunker_chunking_javascript(js_code: str) -> None:
    """Test basic chunking of JavaScript code."""
    chunker = CodeChunker(language="javascript", chunk_size=30)
    chunks = chunker.chunk(js_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == js_code


def test_code_chunker_reconstruction_javascript(js_code: str) -> None:
    """Test if the original JavaScript code can be reconstructed."""
    chunker = CodeChunker(language="javascript", chunk_size=30)
    chunks = chunker.chunk(js_code)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == js_code


def test_code_chunker_chunk_size_javascript(js_code: str) -> None:
    """Test if JavaScript code chunks mostly adhere to chunk_size."""
    chunk_size = 30
    chunker = CodeChunker(language="javascript", chunk_size=chunk_size)
    chunks = chunker.chunk(js_code)
    # Allow for some leeway
    assert all(chunk.token_count < chunk_size + 15 for chunk in chunks[:-1])
    assert chunks[-1].token_count > 0


def test_code_chunker_markdown_document() -> None:
    """Test that CodeChunker handles MarkdownDocument code blocks."""
    python_block = 'def hello():\n    print("world")\n\ndef foo():\n    for i in range(100):\n        print(i)\n'
    js_block = "function greet(name) {\n  console.log(`Hello, ${name}!`);\n}\n"

    doc = MarkdownDocument(
        content=f"# Title\n\nSome text.\n\n```python\n{python_block}```\n\nMore text.\n\n```javascript\n{js_block}```\n",
        chunks=[Chunk(text="Some text.", start_index=10, end_index=20, token_count=2)],
        code=[
            MarkdownCode(
                content=python_block,
                language="python",
                start_index=35,
                end_index=35 + len(python_block),
            ),
            MarkdownCode(
                content=js_block,
                language="javascript",
                start_index=150,
                end_index=150 + len(js_block),
            ),
        ],
    )

    chunker = CodeChunker(language="auto", chunk_size=50)
    result = chunker.chunk_document(doc)

    # Original text chunks should be preserved
    assert any(c.text == "Some text." for c in result.chunks)
    # Code blocks should be chunked and added
    assert len(result.chunks) > 1
    # All code chunk indices should be offset by the code block's start_index
    code_chunks = [c for c in result.chunks if c.text != "Some text."]
    assert len(code_chunks) > 0
    for c in code_chunks:
        assert c.start_index >= 35
    # Chunks should be sorted by start_index
    for i in range(len(result.chunks) - 1):
        assert result.chunks[i].start_index <= result.chunks[i + 1].start_index


def test_code_chunker_markdown_document_empty_code() -> None:
    """Test that CodeChunker skips empty code blocks in MarkdownDocument."""
    doc = MarkdownDocument(
        content="# Title\n\nSome text.\n",
        chunks=[Chunk(text="Some text.", start_index=10, end_index=20, token_count=2)],
        code=[
            MarkdownCode(content="   \n  ", language="python", start_index=50, end_index=60),
        ],
    )

    chunker = CodeChunker(language="python", chunk_size=50)
    result = chunker.chunk_document(doc)

    # Only the original text chunk should remain
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Some text."


def test_code_chunker_markdown_document_no_code() -> None:
    """Test that CodeChunker is a no-op on MarkdownDocument with no code blocks."""
    doc = MarkdownDocument(
        content="# Title\n\nSome text.\n",
        chunks=[Chunk(text="Some text.", start_index=10, end_index=20, token_count=2)],
        code=[],
    )

    chunker = CodeChunker(language="python", chunk_size=50)
    result = chunker.chunk_document(doc)

    # Existing chunks should be preserved untouched
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Some text."


def test_code_chunker_plain_document() -> None:
    """Test that CodeChunker still works with plain Document (non-markdown)."""
    from chonkie.types import Document

    code = 'def hello():\n    print("world")\n'
    doc = Document(content=code)

    chunker = CodeChunker(language="python", chunk_size=512)
    result = chunker.chunk_document(doc)

    assert len(result.chunks) > 0
    reconstructed = "".join(c.text for c in result.chunks)
    assert reconstructed == code
