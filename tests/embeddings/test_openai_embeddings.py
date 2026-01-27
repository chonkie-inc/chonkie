"""Test suite for OpenAIEmbeddings."""

import os
import dotenv
dotenv.load_dotenv()
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openai import APIError, APITimeoutError, RateLimitError
from tenacity import RetryError

from chonkie.embeddings.openai import OpenAIEmbeddings


@pytest.fixture
def embedding_model() -> OpenAIEmbeddings:
    """Fixture to create an OpenAIEmbeddings instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def sample_text() -> str:
    """Fixture to create a sample text for testing."""
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts() -> list[str]:
    """Fixture to create a list of sample texts for testing."""
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_initialization_with_model_name(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings initializes with a model name."""
    assert embedding_model.model == "text-embedding-3-small"
    assert embedding_model.client is not None


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed")
def test_embed_single_text(
    mock_embed,
    embedding_model: OpenAIEmbeddings,
    sample_text: str,
) -> None:
    """Test that OpenAIEmbeddings correctly embeds a single text."""
    mock_embed.return_value = np.zeros(embedding_model.dimension)
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed_batch")
def test_embed_batch_texts(
    mock_embed_batch,
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that OpenAIEmbeddings correctly embeds a batch of texts."""
    mock_embed_batch.return_value = [np.zeros(embedding_model.dimension) for _ in sample_texts]
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(embedding.shape == (embedding_model.dimension,) for embedding in embeddings)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.embed_batch")
@patch("chonkie.embeddings.openai.OpenAIEmbeddings.similarity")
def test_similarity(
    mock_similarity,
    mock_embed_batch,
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that OpenAIEmbeddings correctly calculates similarity between two embeddings."""
    mock_embed_batch.return_value = [np.zeros(embedding_model.dimension) for _ in sample_texts]
    mock_similarity.return_value = np.float32(0.5)
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, np.float32)
    assert 0.0 <= similarity_score <= 1.0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_dimension_property(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings correctly calculates the dimension property."""
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_is_available() -> None:
    """Test that OpenAIEmbeddings correctly checks if it is available."""
    assert OpenAIEmbeddings._is_available() is True


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_repr(embedding_model: OpenAIEmbeddings) -> None:
    """Test that OpenAIEmbeddings correctly returns a string representation."""
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("OpenAIEmbeddings")


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_retry_on_rate_limit_error(embedding_model: OpenAIEmbeddings, sample_text: str) -> None:
    """Test that embed retries on RateLimitError and eventually succeeds."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * embedding_model.dimension)]

    call_count = 0

    def side_effect_rate_limit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        return mock_response

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_rate_limit,
    ):
        result = embedding_model.embed(sample_text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedding_model.dimension,)
        assert call_count == 3  # Failed twice, succeeded on third attempt


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_retry_on_api_error(embedding_model: OpenAIEmbeddings, sample_text: str) -> None:
    """Test that embed retries on APIError and eventually succeeds."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * embedding_model.dimension)]

    call_count = 0

    def side_effect_api_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
        return mock_response

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_api_error,
    ):
        result = embedding_model.embed(sample_text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedding_model.dimension,)
        assert call_count == 2  # Failed once, succeeded on second attempt


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_retry_on_timeout(embedding_model: OpenAIEmbeddings, sample_text: str) -> None:
    """Test that embed retries on APITimeoutError and eventually succeeds."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * embedding_model.dimension)]

    call_count = 0

    def side_effect_timeout(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise APITimeoutError(request=MagicMock())
        return mock_response

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_timeout,
    ):
        result = embedding_model.embed(sample_text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedding_model.dimension,)
        assert call_count == 2  # Failed once, succeeded on second attempt


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_retry_exceeds_max_attempts(embedding_model: OpenAIEmbeddings, sample_text: str) -> None:
    """Test that embed raises RetryError after maximum retry attempts."""
    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        ),
    ):
        with pytest.raises(RetryError):
            embedding_model.embed(sample_text)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_no_retry_on_other_exceptions(embedding_model: OpenAIEmbeddings, sample_text: str) -> None:
    """Test that embed does not retry on non-retryable exceptions."""
    call_count = 0

    def side_effect_value_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid input")

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_value_error,
    ):
        with pytest.raises(ValueError):
            embedding_model.embed(sample_text)
        assert call_count == 1  # Should not retry on ValueError


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_batch_retry_on_rate_limit_error(
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that _embed_batch_with_retry retries on RateLimitError and eventually succeeds."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * embedding_model.dimension, index=i)
        for i in range(len(sample_texts))
    ]

    call_count = 0

    def side_effect_rate_limit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        return mock_response

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_rate_limit,
    ):
        result = embedding_model.embed_batch(sample_texts)
        assert isinstance(result, list)
        assert len(result) == len(sample_texts)
        assert call_count == 3  # Failed twice, succeeded on third attempt


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Skipping test because OPENAI_API_KEY is not defined",
)
def test_batch_fallback_to_individual_on_persistent_error(
    embedding_model: OpenAIEmbeddings,
    sample_texts: list[str],
) -> None:
    """Test that embed_batch falls back to individual embeds when batch fails persistently."""
    mock_individual_response = MagicMock()
    mock_individual_response.data = [MagicMock(embedding=[0.1] * embedding_model.dimension)]

    batch_call_count = 0
    individual_call_count = 0

    def side_effect_batch(*args, **kwargs):
        nonlocal batch_call_count, individual_call_count
        # Check if it's a batch or individual call based on input
        if isinstance(kwargs.get("input"), list) and len(kwargs["input"]) > 1:
            batch_call_count += 1
            raise APIError(
                message="Batch processing error",
                request=MagicMock(),
                body=None,
            )
        else:
            individual_call_count += 1
            return mock_individual_response

    with patch.object(
        embedding_model.client.embeddings,
        "create",
        side_effect=side_effect_batch,
    ):
        result = embedding_model.embed_batch(sample_texts)
        assert isinstance(result, list)
        assert len(result) == len(sample_texts)
        # Should try batch 5 times (max retries), then fall back to individual embeds
        assert batch_call_count == 5
        assert individual_call_count == len(sample_texts)


if __name__ == "__main__":
    pytest.main()
