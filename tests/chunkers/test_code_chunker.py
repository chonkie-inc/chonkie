"""Test the CodeChunker class."""
import pytest

from chonkie import CodeChunker
from chonkie.types import Chunk


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
    assert chunker.parser is not None


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
    assert all(chunk.token_count < chunk_size + 20 for chunk in chunks[:-1]) # Check all but last chunk rigorously
    assert chunks[-1].token_count > 0 # Last chunk must have content


def test_code_chunker_indices_python(python_code: str) -> None:
    """Test the start and end indices of Python code chunks."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    current_index = 0
    for chunk in chunks:
        assert chunk.start_index == current_index
        assert chunk.end_index == current_index + len(chunk.text)
        assert chunk.text == python_code[chunk.start_index:chunk.end_index]
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


def test_code_chunker_adds_line_numbers(python_code: str) -> None:
    """Test that the CodeChunker correctly adds start and end line numbers to chunks."""
    # Use a chunk size that will reliably split the python_code fixture
    chunker = CodeChunker(language="python", chunk_size=100)
    chunks = chunker.chunk(python_code)

    assert len(chunks) > 0, "Chunker should produce at least one chunk"

    # Verify that every chunk has valid, 1-indexed line numbers
    for chunk in chunks:
        assert chunk.start_line is not None, "start_line should not be None"
        assert chunk.end_line is not None, "end_line should not be None"
        assert isinstance(chunk.start_line, int)
        assert isinstance(chunk.end_line, int)
        assert chunk.start_line > 0, "Line numbers should be 1-indexed and positive"
        assert chunk.end_line >= chunk.start_line, "End line must be greater than or equal to start line"

    # The python_code fixture starts with a newline, so its content begins on line 2.
    # The code content ends on line 19 of the input string.
    assert chunks[0].start_line == 2, "The first chunk should start on the first line of code (line 2)"
    assert chunks[-1].end_line == 19, "The last chunk should end on the last line of code (line 19)"