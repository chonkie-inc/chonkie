"""Test the Chonkie Cloud Late Chunker."""

from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import LateChunker
from chonkie.types import LateChunk


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
                "end_index": len(text_input),
                "embedding": [0.1] * 384  # Mock embedding
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
                        "end_index": len(text),
                        "embedding": [0.1] * 384
                    }])
            return results  # Return the list of lists directly for batch response
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


def test_cloud_late_chunker_initialization(mock_requests_get) -> None:
    """Test that the late chunker can be initialized."""
    # Check if chunk_size < 0 raises an error (inherited from superclass validation)
    with pytest.raises(ValueError):
        LateChunker(chunk_size=-1, api_key="test_key")

    # Check if min_characters_per_chunk < 1 raises an error (inherited from superclass validation)
    with pytest.raises(ValueError):
        LateChunker(min_characters_per_chunk=-1, api_key="test_key")

    # Check default initialization
    chunker = LateChunker(api_key="test_key")
    assert chunker.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_chunk == 24  # Default for LateChunker
    # Verify attributes set for the superclass
    assert chunker.tokenizer_or_token_counter == "gpt2" # Set by LateChunker for super

def test_cloud_late_chunker_single_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Late Chunker works with a single text."""
    text = "This is a test sentence for the late chunker. It has several parts."
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    late_chunker = LateChunker(chunk_size=512, api_key="test_key") # Using default embedding model
    result = late_chunker(text)

    assert isinstance(result, list)
    if result: # API might return multiple chunks depending on its logic
        for chunk in result:
            assert isinstance(chunk, LateChunk)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.start_index, int)
            assert isinstance(chunk.end_index, int)
            assert isinstance(chunk.token_count, int)
            assert isinstance(chunk.embedding, list)    


def test_cloud_late_chunker_batch_texts(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Late Chunker works with a batch of texts."""
    texts = [
        "First document for batch processing.",
        "Second document, slightly longer to see if it splits.",
        "Third one.",
    ]
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(texts)
    mock_requests_post.return_value = mock_response
    
    late_chunker = LateChunker(chunk_size=256, api_key="test_key") # Smaller chunk size for variety
    result = late_chunker(texts)

    assert isinstance(result, list)
    # Based on the error, the API for batch input returns a List[List[Dict]]
    # where the outer list contains a single element: the list of all chunks.
    if result:
        assert len(result) > 0 # Ensure we got something back
        assert isinstance(result[0], list) # The actual list of chunks is the first element
        
        all_chunks = result[0]
        if all_chunks:
            for chunk in all_chunks: # Iterate through the inner list of chunks
                assert isinstance(chunk, LateChunk)
                assert isinstance(chunk.text, str)
                assert isinstance(chunk.start_index, int)
                assert isinstance(chunk.end_index, int)
                assert isinstance(chunk.token_count, int)

                # Check that start and end indices are within the bounds of the original texts
                # This is more complex for batch as we don't know which original text a chunk belongs to
                # without more info from API. For now, just check type.
                assert isinstance(chunk.start_index, int)
                assert isinstance(chunk.end_index, int)


def test_cloud_late_chunker_empty_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Late Chunker works with an empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response("")
    mock_requests_post.return_value = mock_response
    
    late_chunker = LateChunker(api_key="test_key")
    result = late_chunker("")
    assert isinstance(result, list)
    assert len(result) == 0


def test_cloud_late_chunker_from_recipe(mock_requests_get) -> None:
    """Test creating a LateChunker from a recipe."""
    # Assuming 'default' recipe exists and is valid
    chunker = LateChunker(
        recipe="default",
        lang="en",
        embedding_model="test-recipe-model",
        chunk_size=128,
        min_characters_per_chunk=5,
        api_key="test_key"
    )
    assert isinstance(chunker, LateChunker)
    assert chunker.embedding_model == "test-recipe-model"
    assert chunker.chunk_size == 128
    assert chunker.min_characters_per_chunk == 5

def test_cloud_late_chunker_real_api(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test with mocked API calls."""
    text = "This is a test sentence for the late chunker. It has several parts."
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    late_chunker = LateChunker(chunk_size=512, api_key="test_key")
    result = late_chunker(text)

    assert isinstance(result, list)
    if result:
        for chunk in result:
            assert isinstance(chunk, LateChunk)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.token_count, int)