"""Test the Chonkie Cloud Recursive Chunker."""

from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import RecursiveChunker


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


def test_cloud_recursive_chunker_initialization(mock_requests_get) -> None:
    """Test that the recursive chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1, api_key="test_key")

    # Check if the min_characters_per_chunk < 1 raises an error
    with pytest.raises(ValueError):
        RecursiveChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=512,
            min_characters_per_chunk=-1,
            api_key="test_key"
        )

    # Finally, check if the attributes are set correctly
    chunker = RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, api_key="test_key")
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_chunk == 12


def test_cloud_recursive_chunker_single_sentence(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Recursive Chunker works with a single sentence."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )

    result = recursive_chunker(text)
    assert len(result) == 1
    assert result[0].text == "Hello, world!"
    assert result[0].token_count == 2  # Mocked value based on word count
    assert result[0].start_index == 0
    assert result[0].end_index == 13


def test_cloud_recursive_chunker_batch(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Recursive Chunker works with a batch of texts."""
    texts = [
        "Hello, world!",
        "This is another sentence.",
        "This is a third sentence.",
    ]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )

    result = recursive_chunker(texts)
    assert len(result) == 3
    assert result[0][0].text == "Hello, world!"
    assert result[0][0].token_count == 2
    assert result[0][0].start_index == 0
    assert result[0][0].end_index == 13


def test_cloud_recursive_chunker_empty_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Recursive Chunker works with an empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response("")
    mock_requests_post.return_value = mock_response
    
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )

    result = recursive_chunker("")
    assert len(result) == 0


def test_cloud_recursive_chunker_real_api(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test with mocked API calls."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )

    result = recursive_chunker(text)
    assert len(result) >= 1
    assert result[0].text == "Hello, world!"