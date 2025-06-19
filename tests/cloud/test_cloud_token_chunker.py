"""Test for the Chonkie Cloud Token Chunker class."""

from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import TokenChunker
from chonkie.types import Chunk


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    def _mock_response(text_input, chunk_count=1):
        if isinstance(text_input, str):
            if not text_input.strip():
                return []
            # Single text input
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
                if not text.strip():
                    results.append([])
                else:
                    results.append([{
                        "text": text,
                        "token_count": max(1, len(text.split())),
                        "start_index": 0,
                        "end_index": len(text)
                    }])
            return results
    return _mock_response


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for API availability check."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for API chunking calls."""
    with patch('requests.post') as mock_post:
        yield mock_post


def test_cloud_token_chunker_initialization(mock_requests_get) -> None:
    """Test that the token chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="gpt2", chunk_size=-1, chunk_overlap=0, api_key="test_key")

    # Check if the chunk_overlap < 0 raises an error
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=-1, api_key="test_key")

    # Finally, check if the attributes are set correctly
    chunker = TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=0, api_key="test_key")
    assert chunker.tokenizer == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 0


def test_cloud_token_chunker_simple(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the token chunker works."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=512,
        chunk_overlap=0,
        api_key="test_key"
    )
    result = token_chunker(text)

    # Check the result
    assert isinstance(result, list) and isinstance(result[0], Chunk) and len(result) == 1
    assert result[0].text == "Hello, world!"
    assert result[0].token_count == 2  # Based on simple word split
    assert result[0].start_index == 0
    assert result[0].end_index == 13


def test_cloud_token_chunker_multiple_sentences(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the token chunker works with a complex text."""
    text = "This is one sentence. This is another sentence. This is a third sentence."
    
    # Mock response for complex text that gets split into multiple chunks
    mock_response = Mock()
    mock_response.status_code = 200
    # Create multiple chunks for longer text
    chunks = [
        {
            "text": "This is one",
            "token_count": 3,
            "start_index": 0,
            "end_index": 11
        },
        {
            "text": " sentence. This is",
            "token_count": 3,
            "start_index": 11,
            "end_index": 29
        },
        {
            "text": " another sentence. This",
            "token_count": 3,
            "start_index": 29,
            "end_index": 52
        },
        {
            "text": " is a third sentence.",
            "token_count": 4,
            "start_index": 52,
            "end_index": len(text)
        }
    ]
    mock_response.json.return_value = chunks
    mock_requests_post.return_value = mock_response
    
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=5,
        chunk_overlap=0,
        api_key="test_key"
    )
    result = token_chunker(text)

    # Check the result
    assert len(result) > 1
    assert isinstance(result, list)
    assert all(isinstance(item, Chunk) for item in result)
    assert all(isinstance(item.text, str) for item in result)
    assert all(isinstance(item.token_count, int) for item in result)
    assert all(isinstance(item.start_index, int) for item in result)
    assert all(isinstance(item.end_index, int) for item in result)


def test_cloud_token_chunker_batch(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the token chunker works with a batch of texts."""
    texts = ["Hello, world!", "This is another sentence.", "This is a third sentence."]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=512,
        chunk_overlap=0,
        api_key="test_key"
    )
    result = token_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result), (
        f"Expected a list of lists, got {type(result)}"
    )
    assert all(isinstance(item, Chunk) for item in result[0]), (
        f"Expected a list of dictionaries, got {type(result[0])}"
    )
    assert all(isinstance(item.text, str) for item in result[0]), (
        f"Expected a list of dictionaries with a 'text' key, got {type(result[0])}"
    )
    assert all(isinstance(item.token_count, int) for item in result[0]), (
        f"Expected a list of dictionaries with a 'token_count' key, got {type(result[0])}"
    )
    assert all(isinstance(item.start_index, int) for item in result[0]), (
        f"Expected a list of dictionaries with a 'start_index' key, got {type(result[0])}"
    )
    assert all(isinstance(item.end_index, int) for item in result[0]), (
        f"Expected a list of dictionaries with a 'end_index' key, got {type(result[0])}"
    )
