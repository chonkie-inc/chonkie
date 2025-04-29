"""Test the CodeChunker class."""
import pytest

from chonkie import CodeChunker
from chonkie.types.code import CodeChunk


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
    assert chunker.return_type == "chunks"
    assert chunker.parser is not None


def test_code_chunker_chunking_python(python_code: str) -> None:
    """Test basic chunking of Python code."""
    chunker = CodeChunker(language="python", chunk_size=50, include_nodes=True)
    chunks = chunker.chunk(python_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
    assert all(chunk.text is not None for chunk in chunks)
    assert all(chunk.start_index is not None for chunk in chunks)
    assert all(chunk.end_index is not None for chunk in chunks)
    assert all(chunk.token_count is not None for chunk in chunks)
    assert all(chunk.nodes is not None for chunk in chunks)


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


def test_code_chunker_return_type_texts(python_code: str) -> None:
    """Test return_type='texts'."""
    chunker = CodeChunker(language="python", chunk_size=50, return_type="texts")
    texts = chunker.chunk(python_code)
    assert isinstance(texts, list)
    assert len(texts) > 0
    assert all(isinstance(text, str) for text in texts)
    reconstructed_text = "".join(texts)
    assert reconstructed_text == python_code


def test_code_chunker_empty_input() -> None:
    """Test chunking an empty string."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("")
    assert chunks == []

    # Test return_type='texts'
    chunker = CodeChunker(language="python", return_type="texts")
    texts = chunker.chunk("")
    assert texts == []


def test_code_chunker_whitespace_input() -> None:
    """Test chunking a string with only whitespace."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("   \n\t\n  ")
    assert chunks == []

    # Test return_type='texts'
    chunker = CodeChunker(language="python", return_type="texts")
    texts = chunker.chunk("   \n\t\n  ")
    assert texts == []


def test_code_chunker_chunking_javascript(js_code: str) -> None:
    """Test basic chunking of JavaScript code."""
    chunker = CodeChunker(language="javascript", chunk_size=30)
    chunks = chunker.chunk(js_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
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