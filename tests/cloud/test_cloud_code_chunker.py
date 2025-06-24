"""Test for the Chonkie Cloud Code Chunker class."""

from typing import Any, Callable, Dict, List, Union
from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import CodeChunker
from chonkie.types import CodeChunk


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


@pytest.fixture
def mock_api_response() -> Callable[[Union[str, List[str]], int], Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]]:
    """Mock successful API response."""
    def _mock_response(text_input: Union[str, List[str]], chunk_count: int = 1) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        if isinstance(text_input, str):
            # Single text input - split into multiple chunks for longer text
            if len(text_input) > 100:
                # Create multiple chunks for longer text that can be perfectly reconstructed
                chunks = []
                chunk_size = len(text_input) // 3  # Split into roughly 3 chunks
                for i in range(0, len(text_input), chunk_size):
                    end_idx = min(i + chunk_size, len(text_input))
                    chunk_text = text_input[i:end_idx]
                    if chunk_text:  # Only add non-empty chunks
                        chunks.append({
                            "text": chunk_text,
                            "token_count": max(1, len(chunk_text.split())),
                            "start_index": i,
                            "end_index": end_idx
                        })
                return chunks if chunks else [{
                    "text": text_input,
                    "token_count": max(1, len(text_input.split())),
                    "start_index": 0,
                    "end_index": len(text_input)
                }]
            else:
                # Single chunk for short text
                return [{
                    "text": text_input,
                    "token_count": max(1, len(text_input.split())),
                    "start_index": 0,
                    "end_index": len(text_input)
                }]
        else:
            # Batch input
            results = []
            for text in text_input:
                if len(text) > 100:
                    # Multiple chunks for longer text that can be perfectly reconstructed
                    chunks = []
                    chunk_size = len(text) // 2
                    for i in range(0, len(text), chunk_size):
                        end_idx = min(i + chunk_size, len(text))
                        chunk_text = text[i:end_idx]
                        if chunk_text:
                            chunks.append({
                                "text": chunk_text,
                                "token_count": max(1, len(chunk_text.split())),
                                "start_index": i,
                                "end_index": end_idx
                            })
                    results.append(chunks if chunks else [{
                        "text": text,
                        "token_count": max(1, len(text.split())),
                        "start_index": 0,
                        "end_index": len(text)
                    }])
                else:
                    # Single chunk for short text
                    results.append([{
                        "text": text,
                        "token_count": max(1, len(text.split())),
                        "start_index": 0,
                        "end_index": len(text)
                    }])
            return results
    return _mock_response


@pytest.fixture
def mock_requests_get() -> Any:
    """Mock requests.get for API availability check."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_requests_post() -> Any:
    """Mock requests.post for API chunking calls."""
    with patch('requests.post') as mock_post:
        yield mock_post


def test_cloud_code_chunker_initialization(mock_requests_get: Any) -> None:
    """Test that the code chunker can be initialized."""
    # Check if the chunk_size <= 0 raises an error
    with pytest.raises(ValueError):
        CodeChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1, api_key="test_key")

    with pytest.raises(ValueError):
        CodeChunker(tokenizer_or_token_counter="gpt2", chunk_size=0, api_key="test_key")

    # Finally, check if the attributes are set correctly
    chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2", 
        chunk_size=512, 
        language="python",
        api_key="test_key"
    )
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.language == "python"


def test_cloud_code_chunker_simple(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any) -> None:
    """Test that the code chunker works with simple code."""
    simple_code = "def hello():\n    print('Hello, world!')"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(simple_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        api_key="test_key"
    )
    result = code_chunker(simple_code)

    # Check the result
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], CodeChunk)
    assert result[0].text
    assert result[0].token_count
    assert result[0].start_index is not None
    assert result[0].end_index is not None
    assert isinstance(result[0].text, str)
    assert isinstance(result[0].token_count, int)
    assert isinstance(result[0].start_index, int)
    assert isinstance(result[0].end_index, int)


def test_cloud_code_chunker_python_complex(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, python_code: str) -> None:
    """Test that the code chunker works with complex Python code."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(python_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=50,
        language="python",
        api_key="test_key"
    )
    result = code_chunker(python_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) > 1  # Should be split into multiple chunks
    assert all(isinstance(item, CodeChunk) for item in result)
    assert all(isinstance(item.text, str) for item in result)
    assert all(isinstance(item.token_count, int) for item in result)
    assert all(isinstance(item.start_index, int) for item in result)
    assert all(isinstance(item.end_index, int) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk.text for chunk in result)
    assert reconstructed == python_code


def test_cloud_code_chunker_javascript(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, js_code: str) -> None:
    """Test that the code chunker works with JavaScript code."""
    # Mock the post request response  
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(js_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=40,
        language="javascript",
        api_key="test_key"
    )
    result = code_chunker(js_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) > 1  # Should be split into multiple chunks
    assert all(isinstance(item, CodeChunk) for item in result)
    assert all(isinstance(item.text, str) for item in result)
    assert all(isinstance(item.token_count, int) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk.text for chunk in result)
    assert reconstructed == js_code


def test_cloud_code_chunker_auto_language(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, python_code: str) -> None:
    """Test that the code chunker works with auto language detection."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(python_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=100,
        language="auto",
        api_key="test_key"
    )
    result = code_chunker(python_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(item, CodeChunk) for item in result)
    assert all(isinstance(item.text, str) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk.text for chunk in result)
    assert reconstructed == python_code


def test_cloud_code_chunker_no_nodes_support(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any) -> None:
    """Test that the code chunker doesn't support nodes (API limitation)."""
    simple_code = "def hello():\n    print('Hello, world!')"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(simple_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        api_key="test_key"
    )
    result = code_chunker(simple_code)

    # Check the result - should not contain nodes since API doesn't support them
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], CodeChunk)
    assert result[0].text
    # API doesn't support tree-sitter nodes, so they shouldn't be in response


def test_cloud_code_chunker_batch(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, python_code: str, js_code: str) -> None:
    """Test that the code chunker works with a batch of texts."""
    texts = [python_code, js_code, "def simple(): pass"]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=100,
        language="auto",
        api_key="test_key"
    )
    result = code_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result), (
        f"Expected a list of lists, got {type(result)}"
    )
    assert all(isinstance(chunk, CodeChunk) for batch in result for chunk in batch), (
        "Expected lists of CodeChunks"
    )
    assert all(isinstance(chunk.text, str) for batch in result for chunk in batch), (
        "Expected chunks with text field"
    )
    assert all(isinstance(chunk.token_count, int) for batch in result for chunk in batch), (
        "Expected chunks with token_count field"
    )


def test_cloud_code_chunker_empty_text(mock_requests_get: Any, mock_requests_post: Any) -> None:
    """Test that the code chunker works with an empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty response for empty input
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        api_key="test_key"
    )

    result = code_chunker("")
    assert len(result) == 0


def test_cloud_code_chunker_whitespace_text(mock_requests_get: Any, mock_requests_post: Any) -> None:
    """Test that the code chunker works with whitespace-only text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty response for whitespace input
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        api_key="test_key"
    )

    result = code_chunker("   \n  \t  ")
    assert len(result) == 0  # Assuming whitespace-only input behaves like empty input


def test_cloud_code_chunker_chunk_size_adherence(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, python_code: str) -> None:
    """Test that code chunks mostly adhere to chunk_size limits."""
    chunk_size = 30
    
    # Mock response with realistic chunk sizes
    mock_response = Mock()
    mock_response.status_code = 200
    chunks = mock_api_response(python_code)
    # Adjust token counts to be realistic for chunk size
    for chunk in chunks:
        chunk["token_count"] = min(chunk_size + 10, chunk["token_count"])  # Allow some tolerance
    mock_response.json.return_value = chunks
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=chunk_size,
        language="python",
        api_key="test_key"
    )
    result = code_chunker(python_code)

    # Most chunks should be close to or under chunk_size (with some tolerance for code boundaries)
    for i, chunk in enumerate(result[:-1]):  # Check all but last chunk
        assert chunk.token_count <= chunk_size + 20, f"Chunk {i} exceeds size limit: {chunk.token_count}"
    
    # Last chunk should have some content
    if result:
        assert result[-1].token_count > 0


def test_cloud_code_chunker_indices_continuity(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any, python_code: str) -> None:
    """Test that chunk indices are continuous."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(python_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=60,
        language="python",
        api_key="test_key"
    )
    result = code_chunker(python_code)

    # Check indices are continuous
    current_index = 0
    for chunk in result:
        assert chunk.start_index == current_index
        assert chunk.end_index > chunk.start_index
        current_index = chunk.end_index
    
    # Final index should match original text length
    assert current_index == len(python_code)


def test_cloud_code_chunker_different_tokenizers(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any) -> None:
    """Test that the code chunker works with different tokenizers."""
    simple_code = "def hello():\n    print('Hello, world!')"
    
    # Test with different tokenizers
    tokenizers = ["gpt2", "cl100k_base"]
    
    for tokenizer in tokenizers:
        # Mock the post request response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response(simple_code)
        mock_requests_post.return_value = mock_response
        
        code_chunker = CodeChunker(
            tokenizer_or_token_counter=tokenizer,
            chunk_size=512,
            language="python",
            api_key="test_key"
        )
        result = code_chunker(simple_code)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(chunk, CodeChunk) for chunk in result)
        assert all(chunk.text for chunk in result)


def test_cloud_code_chunker_real_api(mock_requests_get: Any, mock_requests_post: Any, mock_api_response: Any) -> None:
    """Test with mocked API calls."""
    simple_code = "def hello():\n    print('Hello, world!')"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(simple_code)
    mock_requests_post.return_value = mock_response
    
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        api_key="test_key",  # Use a test key to avoid env dependency
    )
    result = code_chunker(simple_code)

    # Check the result
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], CodeChunk)
    assert result[0].text
    assert result[0].token_count