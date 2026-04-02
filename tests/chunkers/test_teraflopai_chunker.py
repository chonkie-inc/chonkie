"""Tests for the TeraflopAIChunker class."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from chonkie import Chunk, TeraflopAIChunker  # noqa: I001

# Create a fake teraflopai module so that lazy import works without
# the real package installed.
_fake_teraflopai = MagicMock()
_fake_TeraflopAI_cls = MagicMock()
_fake_teraflopai.TeraflopAI = _fake_TeraflopAI_cls


@pytest.fixture
def mock_client():
    """Create a mock TeraflopAI client."""
    client = MagicMock()
    return client


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return (
        "UNITED STATES of America, Appellee, v. Daniel Dee VEON, Appellant. "
        "No. 72-1889. United States Court of Appeals, Ninth Circuit. "
        "Feb. 12, 1973."
    )


@pytest.fixture
def sample_segments() -> list[str]:
    """Return sample segments that the API might return."""
    return [
        "UNITED STATES of America, Appellee, v. Daniel Dee VEON, Appellant.",
        "No. 72-1889.",
        "United States Court of Appeals, Ninth Circuit.",
        "Feb. 12, 1973.",
    ]


@pytest.fixture
def chunker_with_mock(mock_client) -> TeraflopAIChunker:
    """Create a TeraflopAIChunker with a mock client."""
    with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
        chunker = TeraflopAIChunker(client=mock_client)
    return chunker


class TestTeraflopAIChunkerInitialization:
    """Tests for TeraflopAIChunker initialization."""

    def test_import_error_when_teraflopai_not_installed(self) -> None:
        """Test that ImportError is raised when teraflopai is not installed."""
        with patch.dict(sys.modules, {"teraflopai": None}):
            with pytest.raises(ImportError, match="teraflopai is not installed"):
                TeraflopAIChunker(api_key="test_key")

    def test_init_with_client(self, mock_client) -> None:
        """Test initialization with an existing client."""
        with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
            chunker = TeraflopAIChunker(client=mock_client)
            assert chunker.client is mock_client

    def test_init_with_api_key(self) -> None:
        """Test initialization with an API key."""
        _fake_TeraflopAI_cls.reset_mock()
        with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
            chunker = TeraflopAIChunker(api_key="test_key")
            _fake_TeraflopAI_cls.assert_called_once_with(
                api_key="test_key",
                url="https://api.segmentation.teraflopai.com/v1/segmentation/free",
            )
            assert chunker is not None, "Chunker should be initialized successfully with API key."

    def test_init_with_env_api_key(self, monkeypatch) -> None:
        """Test initialization with API key from environment variable."""
        monkeypatch.setenv("TERAFLOPAI_API_KEY", "env_test_key")
        _fake_TeraflopAI_cls.reset_mock()
        with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
            chunker = TeraflopAIChunker()
            _fake_TeraflopAI_cls.assert_called_once_with(
                api_key="env_test_key",
                url="https://api.segmentation.teraflopai.com/v1/segmentation/free",
            )
            assert chunker is not None, (
                "Chunker should be initialized successfully with env API key."
            )

    def test_init_without_api_key_raises(self, monkeypatch) -> None:
        """Test that ValueError is raised when no API key is provided."""
        monkeypatch.delenv("TERAFLOPAI_API_KEY", raising=False)
        with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
            with pytest.raises(ValueError, match="API key is required"):
                TeraflopAIChunker()

    def test_init_with_custom_url(self) -> None:
        """Test initialization with a custom URL."""
        _fake_TeraflopAI_cls.reset_mock()
        custom_url = "https://custom.api.teraflopai.com/v1/segmentation"
        with patch.dict(sys.modules, {"teraflopai": _fake_teraflopai}):
            chunker = TeraflopAIChunker(api_key="test_key", url=custom_url)
            _fake_TeraflopAI_cls.assert_called_once_with(api_key="test_key", url=custom_url)
            assert chunker is not None, (
                "Chunker should be initialized successfully with custom URL."
            )


class TestTeraflopAIChunkerChunking:
    """Tests for TeraflopAIChunker chunking functionality."""

    def test_chunk_returns_chunks(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that chunk returns a list of Chunk objects."""
        mock_client.segment.return_value = {"results": sample_segments}
        chunks = chunker_with_mock.chunk(sample_text)

        assert isinstance(chunks, list)
        assert len(chunks) == len(sample_segments)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)

    def test_chunk_text_content(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that chunk text contains each segment."""
        mock_client.segment.return_value = {"results": sample_segments}
        chunks = chunker_with_mock.chunk(sample_text)

        for chunk, expected_text in zip(chunks, sample_segments):
            assert expected_text in chunk.text

    def test_chunk_indices(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that chunk start and end indices are correct and contiguous."""
        mock_client.segment.return_value = {"results": sample_segments}
        chunks = chunker_with_mock.chunk(sample_text)

        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(sample_text)
            assert chunk.end_index > chunk.start_index
            # Verify the text at the indices matches the chunk text
            assert sample_text[chunk.start_index : chunk.end_index] == chunk.text

        # Verify chunks are contiguous: no gaps between consecutive chunks
        assert chunks[0].start_index == 0
        for i in range(1, len(chunks)):
            assert chunks[i].start_index == chunks[i - 1].end_index

    def test_chunk_token_counts(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that chunk token counts are positive."""
        mock_client.segment.return_value = {"results": sample_segments}
        chunks = chunker_with_mock.chunk(sample_text)

        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_empty_text(self, chunker_with_mock) -> None:
        """Test that chunking empty text returns an empty list."""
        chunks = chunker_with_mock.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self, chunker_with_mock) -> None:
        """Test that chunking whitespace-only text returns an empty list."""
        chunks = chunker_with_mock.chunk("   \n\t  ")
        assert chunks == []

    def test_chunk_no_results(self, chunker_with_mock, mock_client) -> None:
        """Test fallback when API returns no results."""
        mock_client.segment.return_value = {"results": []}
        text = "Some text that the API did not segment."
        chunks = chunker_with_mock.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == len(text)

    def test_chunk_single_segment(self, chunker_with_mock, mock_client) -> None:
        """Test when API returns a single segment."""
        text = "A single sentence."
        mock_client.segment.return_value = {"results": [text]}
        chunks = chunker_with_mock.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text


class TestTeraflopAIChunkerBatch:
    """Tests for TeraflopAIChunker batch and callable functionality."""

    def test_callable_single_text(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that calling the chunker with a single string works."""
        mock_client.segment.return_value = {"results": sample_segments}
        result = chunker_with_mock(sample_text)

        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_callable_batch(
        self, chunker_with_mock, mock_client, sample_text, sample_segments
    ) -> None:
        """Test that calling the chunker with a list of strings works."""
        mock_client.segment.return_value = {"results": sample_segments}
        result = chunker_with_mock([sample_text, sample_text])

        assert isinstance(result, list)
        assert len(result) == 2
        for batch_result in result:
            assert isinstance(batch_result, list)
            assert all(isinstance(c, Chunk) for c in batch_result)


class TestTeraflopAIChunkerRepr:
    """Tests for TeraflopAIChunker string representation."""

    def test_repr(self, chunker_with_mock) -> None:
        """Test __repr__ output."""
        repr_str = repr(chunker_with_mock)
        assert "TeraflopAIChunker" in repr_str
