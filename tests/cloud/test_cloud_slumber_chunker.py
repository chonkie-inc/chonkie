"""Test the Chonkie Cloud Slumber Chunker."""

import os
from unittest.mock import Mock, patch

import pytest
from pytest import MonkeyPatch

from chonkie.cloud import SlumberChunker
from chonkie.types import RecursiveLevel, RecursiveRules


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


def test_cloud_slumber_chunker_no_api_key(monkeypatch: MonkeyPatch) -> None:
    """Test SlumberChunker initialization fails without API key."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        SlumberChunker()


def test_cloud_slumber_chunker_initialization(mock_requests_get) -> None:
    """Test that the slumber chunker can be initialized."""
    # Check if chunk_size <= 0 raises an error
    with pytest.raises(ValueError, match="Chunk size must be greater than 0."):
        SlumberChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1, api_key="test_key")

    # Check if candidate_size <= 0 raises an error
    with pytest.raises(ValueError, match="Candidate size must be greater than 0."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            candidate_size=-1,
            api_key="test_key"
        )

    # Check if min_characters_per_chunk < 1 raises an error
    with pytest.raises(ValueError, match="Minimum characters per chunk must be greater than 0."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            min_characters_per_chunk=0,
            api_key="test_key"
        )

    # Check if return_type is not "texts" or "chunks" raises an error
    with pytest.raises(ValueError, match="Return type must be either 'texts' or 'chunks'."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            return_type="not_a_valid_type",
            api_key="test_key"
        )

    # Check default initialization
    chunker = SlumberChunker(api_key="test_key")
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 1024  # Default is 1024, not 512
    assert chunker.candidate_size == 128  # Default is 128, not 512
    assert chunker.min_characters_per_chunk == 24  # Default is 24, not 10
    assert chunker.return_type == "chunks"
    assert isinstance(chunker.rules, RecursiveRules)

    # Check initialization with custom parameters
    custom_levels = [RecursiveLevel(delimiters=["\n\n", "\n", ". "])]
    custom_rules = RecursiveRules(levels=custom_levels)
    custom_chunker = SlumberChunker(
        tokenizer_or_token_counter="cl100k_base",
        chunk_size=1024,
        candidate_size=1024,
        min_characters_per_chunk=20,
        return_type="texts",
        rules=custom_rules,
        api_key="test_key"
    )
    assert custom_chunker.tokenizer_or_token_counter == "cl100k_base"
    assert custom_chunker.chunk_size == 1024
    assert custom_chunker.candidate_size == 1024
    assert custom_chunker.min_characters_per_chunk == 20
    assert custom_chunker.return_type == "texts"
    assert custom_chunker.rules == custom_rules


def test_cloud_slumber_chunker_single_sentence(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Slumber Chunker works with a single sentence."""
    text = "This is a simple sentence for testing the slumber chunker."
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )
    result = slumber_chunker(text)

    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    assert "token_count" in result[0]
    assert "start_index" in result[0]
    assert "end_index" in result[0]
    assert isinstance(result[0]["text"], str)
    assert isinstance(result[0]["token_count"], int)
    assert isinstance(result[0]["start_index"], int)
    assert isinstance(result[0]["end_index"], int)


def test_cloud_slumber_chunker_batch(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Slumber Chunker works with a batch of texts."""
    texts = [
        "Hello, world!",
        "This is another sentence for batch processing.",
        "And a third one to ensure multiple inputs are handled.",
    ]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )
    result = slumber_chunker(texts)

    # Expect a list of lists of dictionaries, one inner list per input text
    assert isinstance(result, list)
    assert len(result) == len(texts)
    for i, text_result in enumerate(result):
        assert isinstance(text_result, list)
        assert len(text_result) >= 1
        for chunk in text_result:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "token_count" in chunk
            assert "start_index" in chunk
            assert "end_index" in chunk


def test_cloud_slumber_chunker_from_recipe() -> None:
    """Test that the Slumber Chunker from_recipe method availability."""
    # Check if from_recipe method exists, if not skip the test
    if not hasattr(SlumberChunker, 'from_recipe'):
        pytest.skip("SlumberChunker.from_recipe method not implemented yet")
    
    # If it exists, test it
    sample_recipe_path = "tests/samples/recipe.json"
    if os.path.exists(sample_recipe_path):
        slumber_chunker = SlumberChunker.from_recipe(sample_recipe_path, api_key="test_key")
        assert slumber_chunker is not None
    else:
        pytest.skip("Sample recipe file not found")


def test_cloud_slumber_chunker_return_type_texts(mock_requests_get, mock_requests_post) -> None:
    """Test that the Slumber Chunker works with return_type='texts'."""
    text = "This is a test for return type texts."
    
    # Mock the post request response for text return type
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [text]  # Return type texts returns list of strings
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        return_type="texts",
        api_key="test_key"
    )
    result = slumber_chunker(text)

    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


def test_cloud_slumber_chunker_return_type_texts_batch(mock_requests_get, mock_requests_post) -> None:
    """Test that the Slumber Chunker works with return_type='texts' for batch processing."""
    texts = [
        "First text for batch processing.",
        "Second text for batch processing.",
    ]
    
    # Mock the post request response for batch text return type
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [texts[0], texts[1]]  # Flat list for texts return type
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        return_type="texts",
        api_key="test_key"
    )
    result = slumber_chunker(texts)

    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


def test_cloud_slumber_chunker_empty_text(mock_requests_get, mock_requests_post) -> None:
    """Test that the Slumber Chunker works with empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty response for empty input
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        api_key="test_key"
    )
    result = slumber_chunker("")

    assert isinstance(result, list)
    assert len(result) == 0


def test_cloud_slumber_chunker_long_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Slumber Chunker works with long text."""
    long_text = "This is a longer text for testing. " * 50
    
    # Mock response for long text that gets split into multiple chunks
    mock_response = Mock()
    mock_response.status_code = 200
    # Create multiple chunks for long text
    chunks = [
        {
            "text": long_text[:len(long_text)//2],
            "token_count": len(long_text[:len(long_text)//2].split()),
            "start_index": 0,
            "end_index": len(long_text)//2
        },
        {
            "text": long_text[len(long_text)//2:],
            "token_count": len(long_text[len(long_text)//2:].split()),
            "start_index": len(long_text)//2,
            "end_index": len(long_text)
        }
    ]
    mock_response.json.return_value = chunks
    mock_requests_post.return_value = mock_response
    
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=100,  # Small chunk size to force splitting
        api_key="test_key"
    )
    result = slumber_chunker(long_text)

    assert isinstance(result, list)
    assert len(result) >= 2  # Should be split into multiple chunks
    for chunk in result:
        assert isinstance(chunk, dict)
        assert "text" in chunk
        assert "token_count" in chunk


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set - Skipping real API test",
)
def test_cloud_slumber_chunker_real_api() -> None:
    """Test with real API if CHONKIE_API_KEY is available."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
    )
    text = "This is a test sentence for the real API."
    result = slumber_chunker(text)

    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    assert "token_count" in result[0]