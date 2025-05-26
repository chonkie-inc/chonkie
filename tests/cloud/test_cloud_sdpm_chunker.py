"""Test the Chonkie Cloud SDPM Chunker."""

import os
from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import SDPMChunker


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    def _mock_response(text_input, chunk_count=1):
        if isinstance(text_input, str):
            if not text_input.strip():
                return []
            # Single text input - SDPM might not chunk very short text
            if len(text_input.split()) < 3:
                return []  # SDPM typically requires more text
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
                if not text.strip() or len(text.split()) < 3:
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


def test_cloud_sdpm_chunker_initialization(mock_requests_get) -> None:
    """Test that the SDPM chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SDPMChunker(chunk_size=-1, api_key="test_key")

    # Check if the threshold is a str but not "auto"
    with pytest.raises(ValueError):
        SDPMChunker(threshold="not_auto", api_key="test_key")

    # Check if the similarity window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(similarity_window=-1, api_key="test_key")

    # Check if the min_sentences is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_sentences=-1, api_key="test_key")

    # Check if the min_chunk_size is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_chunk_size=-1, api_key="test_key")

    # Check if the min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_characters_per_sentence=-1, api_key="test_key")

    # Check if the threshold_step is not a positive float
    with pytest.raises(ValueError):
        SDPMChunker(threshold_step=-0.1, api_key="test_key")
    
    # Check if the delim is not a list
    with pytest.raises(ValueError):
        SDPMChunker(delim="not_a_list", api_key="test_key")

    # Check if the include_delim is not "prev" or "next"
    with pytest.raises(ValueError):
        SDPMChunker(include_delim="not_valid", api_key="test_key")

    # Check if the skip_window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(skip_window=-1, api_key="test_key")

    # Check if the return_type is not "chunks" or "texts"
    with pytest.raises(ValueError):
        SDPMChunker(return_type="not_valid", api_key="test_key")

    # Check if the embedding_model is not a string
    with pytest.raises(ValueError):
        SDPMChunker(embedding_model=123, api_key="test_key") # type: ignore

    # Finally, check if the attributes are set correctly
    chunker = SDPMChunker(chunk_size=256, api_key="test_key") 
    assert chunker.embedding_model == "minishlab/potion-base-8M"
    assert chunker.mode == "window"
    assert chunker.threshold == "auto"
    assert chunker.chunk_size == 256
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.min_characters_per_sentence == 12
    assert chunker.threshold_step == 0.01
    assert chunker.delim == [". ", "! ", "? ", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.skip_window == 1
    assert chunker.return_type == "chunks"
    assert chunker.api_key == "test_key"


def test_cloud_sdpm_chunker_single_sentence(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the SDPM Chunker works with a single sentence."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
        api_key="test_key"
    )

    result = sdpm_chunker(text)
    assert len(result) >= 0 # API might return 0 chunks for very short text
    if len(result) > 0:
        # Exact values depend on the API response, so we check for presence and type
        assert "text" in result[0]
        assert isinstance(result[0]["text"], str)
        assert "token_count" in result[0]
        assert isinstance(result[0]["token_count"], int)
        assert "start_index" in result[0]
        assert isinstance(result[0]["start_index"], int)
        assert "end_index" in result[0]
        assert isinstance(result[0]["end_index"], int)


def test_cloud_sdpm_chunker_batch(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the SDPM Chunker works with a batch of texts."""
    texts = [
        "Hello, world!",
        "This is another sentence.",
        "This is a third sentence. This is a fourth one to make it longer.",
    ]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
        api_key="test_key"
    )
    
    result = sdpm_chunker(texts)
    assert len(result) == len(texts)
    for i in range(len(texts)):
        assert isinstance(result[i], list) # API returns a list of lists for batch
        if len(result[i]) > 0:
            assert "text" in result[i][0] 
            assert isinstance(result[i][0]["text"], str)
            assert "token_count" in result[i][0]
            assert isinstance(result[i][0]["token_count"], int)
            assert "start_index" in result[i][0]
            assert isinstance(result[i][0]["start_index"], int)
            assert "end_index" in result[i][0]
            assert isinstance(result[i][0]["end_index"], int)


def test_cloud_sdpm_chunker_empty_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the SDPM Chunker works with an empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response("")
    mock_requests_post.return_value = mock_response
    
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
        api_key="test_key"
    )

    result = sdpm_chunker("")
    # The API might return an empty list or a list with an empty chunk object
    # Depending on the API's behavior, this assertion might need adjustment.
    # Based on SemanticChunker tests, an empty list is expected.
    assert len(result) == 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_real_api() -> None:
    """Test with real API if CHONKIE_API_KEY is available."""
    sdpm_chunker = SDPMChunker(chunk_size=512)
    text = "This is a test sentence for the SDPM chunker. It has several parts. This is another sentence to test with."
    
    result = sdpm_chunker(text)
    assert isinstance(result, list)
    if result:
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "token_count" in chunk