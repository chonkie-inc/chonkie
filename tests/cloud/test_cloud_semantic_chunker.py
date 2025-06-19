"""Test the Chonkie Cloud Semantic Chunker."""

from unittest.mock import Mock, patch

import pytest

from chonkie.cloud import SemanticChunker


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


def test_cloud_semantic_chunker_initialization(mock_requests_get) -> None:
    """Test that the semantic chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=-1, api_key="test_key")

    # Check if the threshold is a str but not "auto"
    with pytest.raises(ValueError):
        SemanticChunker(threshold="not_auto", api_key="test_key")

    # Check if the threshold is a number but not between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold=1.1, api_key="test_key")

    # Check if the threshold is a number but not between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold=-0.1, api_key="test_key")

    # Check if the similarity window is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(similarity_window=-1, api_key="test_key")

    # Check if the min_sentences is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_sentences=-1, api_key="test_key")

    # Check if the min_chunk_size is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_chunk_size=-1, api_key="test_key")

    # Check if the min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_characters_per_sentence=-1, api_key="test_key")

    # Check if the threshold_step is not a number between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold_step=-0.1, api_key="test_key")

    # Check if the threshold_step is not a number between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold_step=1.1, api_key="test_key")

    # Check if the delim is not a string or a list of strings
    with pytest.raises(ValueError):
        SemanticChunker(delim=1, api_key="test_key")

    # Check if the include_delim is not a string or a list of strings
    with pytest.raises(ValueError):
        SemanticChunker(include_delim=1, api_key="test_key")

    # Finally, check if the attributes are set correctly
    chunker = SemanticChunker(chunk_size=512, api_key="test_key")
    assert chunker.embedding_model == "minishlab/potion-base-32M"
    assert chunker.chunk_size == 512
    assert chunker.threshold == "auto"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.min_characters_per_sentence == 12
    assert chunker.threshold_step == 0.01
    assert chunker.delim == [". ", "! ", "? ", "\n"]
    assert chunker.include_delim == "prev"


def test_cloud_semantic_chunker_single_sentence(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Semantic Chunker works with a single sentence."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    semantic_chunker = SemanticChunker(
        chunk_size=512,
        api_key="test_key"
    )

    result = semantic_chunker(text)
    assert len(result) == 1
    assert result[0].text == "Hello, world!"
    assert result[0].token_count == 2  # Based on simple word split
    assert result[0].start_index == 0
    assert result[0].end_index == 13


def test_cloud_semantic_chunker_batch(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Semantic Chunker works with a batch of texts."""
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
    
    semantic_chunker = SemanticChunker(
        chunk_size=512,
        api_key="test_key"
    )
    
    result = semantic_chunker(texts)
    assert len(result) == 3
    assert result[0][0].text == "Hello, world!"
    assert result[0][0].token_count == 2
    assert result[0][0].start_index == 0
    assert result[0][0].end_index == 13


def test_cloud_semantic_chunker_empty_text(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test that the Semantic Chunker works with an empty text."""
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response("")
    mock_requests_post.return_value = mock_response
    
    semantic_chunker = SemanticChunker(
        chunk_size=512,
        api_key="test_key"
    )

    result = semantic_chunker("")
    assert len(result) == 0


def test_cloud_semantic_chunker_real_api(mock_requests_get, mock_requests_post, mock_api_response) -> None:
    """Test with mocked API calls."""
    text = "Hello, world!"
    
    # Mock the post request response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response(text)
    mock_requests_post.return_value = mock_response
    
    semantic_chunker = SemanticChunker(chunk_size=512, api_key="test_key")
    result = semantic_chunker(text)
    assert len(result) >= 1
    assert result[0].text == "Hello, world!"