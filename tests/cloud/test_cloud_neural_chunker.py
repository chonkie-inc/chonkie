"""Test the Chonkie Cloud Neural Chunker."""

import os
from typing import Any, Dict, List

import pytest
import requests  # Import requests to mock its methods

from chonkie.cloud.chunker import NeuralChunker


@pytest.fixture
def mock_requests_get_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock requests.get to return a successful response."""
    class MockResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self) -> dict:
            return {} # Or some other relevant JSON if needed

    def mock_get(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(200)

    monkeypatch.setattr(requests, "get", mock_get)


@pytest.fixture
def mock_requests_post_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock requests.post to return a successful response with dummy chunk data."""
    class MockResponse:
        def __init__(self, status_code: int, json_data: List[Dict[str, Any]]):
            self.status_code = status_code
            self._json_data = json_data

        def json(self) -> List[Dict[str, Any]]:
            return self._json_data

    # Default dummy response for post, can be customized per test if needed
    dummy_chunks: List[Dict[str, Any]] = [
        {"text": "chunk1", "start_index": 0, "end_index": 6, "token_count": 1},
        {"text": "chunk2", "start_index": 7, "end_index": 13, "token_count": 1},
    ]
    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(200, dummy_chunks)

    monkeypatch.setattr(requests, "post", mock_post)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_initialization(mock_requests_get_success: Any) -> None:
    """Test that the neural chunker can be initialized."""
    # Check if model not in SUPPORTED_MODELS raises an error
    with pytest.raises(ValueError):
        NeuralChunker(model="unsupported-model")

    # Check if min_characters_per_chunk < 1 raises an error
    with pytest.raises(ValueError):
        NeuralChunker(min_characters_per_chunk=0)

    # Check if return_type not in ["texts", "chunks"] raises an error
    with pytest.raises(ValueError):
        NeuralChunker(return_type="invalid_type") # type: ignore

    # Check default initialization
    chunker = NeuralChunker()
    assert chunker.model == NeuralChunker.DEFAULT_MODEL
    assert chunker.min_characters_per_chunk == 10
    assert chunker.stride == NeuralChunker.SUPPORTED_MODEL_STRIDES[NeuralChunker.DEFAULT_MODEL]
    assert chunker.return_type == "chunks"
    assert chunker.api_key == os.getenv("CHONKIE_API_KEY")

    # Check initialization with custom parameters
    custom_model = "mirth/chonky_modernbert_base_1"
    custom_chunker = NeuralChunker(
        model=custom_model,
        min_characters_per_chunk=5,
        stride=128,
        return_type="texts",
        api_key="test_key",
    )
    assert custom_chunker.model == custom_model
    assert custom_chunker.min_characters_per_chunk == 5
    assert custom_chunker.stride == 128
    assert custom_chunker.return_type == "texts"
    assert custom_chunker.api_key == "test_key"

    # Test initialization without API key (should raise ValueError if not in env)
    original_api_key = os.environ.pop("CHONKIE_API_KEY", None)
    with pytest.raises(ValueError, match="No API key provided"):
        NeuralChunker(api_key=None) # Explicitly pass None
    if original_api_key: # Restore API key if it was present
        os.environ["CHONKIE_API_KEY"] = original_api_key


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_initialization_api_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NeuralChunker initialization when the API is down."""
    class MockResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code
    
    def mock_get_api_down(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(500) # Simulate API being down

    monkeypatch.setattr(requests, "get", mock_get_api_down)

    with pytest.raises(ValueError, match="Oh no! You caught Chonkie at a bad time"):
        NeuralChunker()


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_single_text(mock_requests_get_success: Any, mock_requests_post_success: Any) -> None:
    """Test that the Neural Chunker works with a single text."""
    chunker = NeuralChunker()
    text = "This is a test sentence for the neural chunker."
    result = chunker(text)

    assert isinstance(result, list)
    if result:
        for chunk_dict in result: # API returns List[Dict] directly
            assert isinstance(chunk_dict, dict)
            assert "text" in chunk_dict
            assert isinstance(chunk_dict["text"], str)
            assert "start_index" in chunk_dict
            assert isinstance(chunk_dict["start_index"], int)
            assert "end_index" in chunk_dict
            assert isinstance(chunk_dict["end_index"], int)
            # token_count is part of the mocked response
            assert "token_count" in chunk_dict
            assert isinstance(chunk_dict["token_count"], int)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_batch_texts(mock_requests_get_success: Any, mock_requests_post_success: Any) -> None:
    """Test that the Neural Chunker works with a batch of texts."""
    chunker = NeuralChunker()
    texts = [
        "First document for batch processing.",
        "Second document, slightly longer.",
    ]
    # The mock_requests_post_success by default returns a flat list of chunks,
    # which is the expected behavior for NeuralChunker for both single and batch.
    result = chunker(texts)

    assert isinstance(result, list)
    if result:
        for chunk_dict in result: # API returns List[Dict] directly
            assert isinstance(chunk_dict, dict)
            assert "text" in chunk_dict
            assert "start_index" in chunk_dict
            assert "end_index" in chunk_dict
            assert "token_count" in chunk_dict


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_empty_text(mock_requests_get_success: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Neural Chunker works with an empty text."""
    class MockResponse:
        def __init__(self, status_code: int, json_data: List[Dict[str, Any]]):
            self.status_code = status_code
            self._json_data = json_data

        def json(self) -> List[Dict[str, Any]]:
            return self._json_data

    # Mock post to return an empty list for empty text
    def mock_post_empty_result(*args: Any, **kwargs: Any) -> MockResponse:
        payload: Dict[str, Any] = kwargs.get("json", {})
        if payload.get("text") == "": # Assuming this simplified check is intended for the mock
            return MockResponse(200, []) # API should return empty list for empty string
        # Fallback to a default non-empty response if needed for other calls in the same test scope
        return MockResponse(200, [{"text": "default", "start_index": 0, "end_index": 7, "token_count": 1}])


    monkeypatch.setattr(requests, "post", mock_post_empty_result)
    
    chunker = NeuralChunker()
    result = chunker("")
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_neural_chunker_api_error_on_chunk(mock_requests_get_success: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NeuralChunker's chunk method when the API returns an error."""
    class MockResponse:
        def __init__(self, status_code: int, content: str = "Error content"):
            self.status_code = status_code
            self.content = content # Store content for error messages

        def json(self) -> List[Dict[str, Any]]: # Typehinted to what the caller expects
            raise requests.exceptions.JSONDecodeError("Mock JSON decode error", "doc", 0)

    def mock_post_api_error(*args: Any, **kwargs: Any) -> MockResponse:
        # Simulate an API error (e.g., 500 internal server error or 400 bad request)
        # For this test, we'll focus on the JSON decode error.
        return MockResponse(200) # Status code might be 200 but content is bad

    monkeypatch.setattr(requests, "post", mock_post_api_error)
    
    chunker = NeuralChunker()
    with pytest.raises(ValueError, match="Oh no! The Chonkie API returned an invalid response"):
        chunker("Some text to chunk")
