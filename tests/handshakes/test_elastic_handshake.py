"""Test the ElasticHandshake class."""

import uuid
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Try to import elasticsearch, skip all tests in this module if it's not installed
try:
    import elasticsearch
except ImportError:
    pass

from chonkie.embeddings import BaseEmbeddings
from chonkie.handshakes.elastic import ElasticHandshake
from chonkie.types import Chunk

# Mark all tests in this module to be skipped if elasticsearch is not installed
pytestmark = pytest.mark.skipif(
    elasticsearch is None, reason="elasticsearch-py not installed"
)

# ---- Fixtures ----


@pytest.fixture
def mock_embeddings() -> Generator[MagicMock, None, None]:
    """Mock AutoEmbeddings to avoid downloading models and to provide consistent results."""
    with patch(
        "chonkie.embeddings.AutoEmbeddings.get_embeddings"
    ) as mock_get_embeddings:
        mock_embedding_model = MagicMock(spec=BaseEmbeddings)
        mock_embedding_model.dimension = 128  # Use a consistent dimension for tests
        import numpy as np

        mock_embedding_model.embed.return_value = np.array([0.1] * 128)
        mock_embedding_model.embed_batch.return_value = [[0.1] * 128, [0.2] * 128]
        mock_get_embeddings.return_value = mock_embedding_model
        yield mock_embedding_model


@pytest.fixture
def sample_chunk() -> Chunk:
    """Provide a single sample Chunk."""
    return Chunk(
        text="This is a test chunk.", start_index=0, end_index=22, token_count=5
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide a list of sample Chunks."""
    return [
        Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
        Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
    ]


# ---- Elastic Client and Bulk Mock Fixture ----


@pytest.fixture
def mock_elastic_client_and_bulk(monkeypatch):
    """Mock the Elasticsearch client and the bulk helper."""
    # Patch elasticsearch.Elasticsearch globally
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = False
    mock_client.indices.create.return_value = None
    mock_client.search.return_value = {"hits": {"hits": []}}
    mock_bulk = MagicMock()
    mock_bulk.return_value = (1, [])
    monkeypatch.setattr(
        "elasticsearch.Elasticsearch", lambda *args, **kwargs: mock_client
    )
    monkeypatch.setattr("elasticsearch.helpers.bulk", mock_bulk)
    return mock_client, mock_bulk


# ---- Initialization Tests ----


def test_elastic_handshake_init_existing_index(
    mock_elastic_client_and_bulk: tuple[MagicMock, MagicMock],
) -> None:
    """Test that an index is NOT created if it already exists."""
    mock_client, _ = mock_elastic_client_and_bulk
    mock_client.indices.exists.return_value = True  # Simulate index already exists

    handshake = ElasticHandshake(index_name="my-existing-index")

    mock_client.indices.exists.assert_called_once_with(index="my-existing-index")
    mock_client.indices.create.assert_not_called()
    assert handshake.index_name == "my-existing-index"


# ---- Write Tests ----


def test_write_single_chunk(
    mock_elastic_client_and_bulk: tuple[MagicMock, MagicMock],
    mock_embeddings: MagicMock,
    sample_chunk: Chunk,
) -> None:
    """Test writing a single chunk."""
    mock_client, mock_bulk = mock_elastic_client_and_bulk
    handshake = ElasticHandshake()
    handshake.write(sample_chunk)

    # Check that the bulk helper was called correctly
    mock_bulk.assert_called_once()
    # Inspect the arguments passed to bulk()
    args, _ = mock_bulk.call_args
    assert args[0] == mock_client  # First arg is the client
    actions = args[1]
    assert len(actions) == 1
    action = actions[0]
    assert action["_index"] == handshake.index_name
    assert "_id" in action
    assert action["_source"]["text"] == sample_chunk.text
    assert (
        action["_source"]["embedding"] == [0.1] * 128
    )  # From mock_embeddings.embed_batch


def test_write_multiple_chunks(
    mock_elastic_client_and_bulk: tuple[MagicMock, MagicMock],
    mock_embeddings: MagicMock,
    sample_chunks: list[Chunk],
) -> None:
    """Test writing multiple chunks."""
    mock_client, mock_bulk = mock_elastic_client_and_bulk
    # Configure mock to return correct number of successes
    mock_bulk.return_value = (len(sample_chunks), [])

    handshake = ElasticHandshake()
    handshake.write(sample_chunks)

    mock_bulk.assert_called_once()
    args, _ = mock_bulk.call_args
    actions = args[1]
    assert len(actions) == len(sample_chunks)
    assert actions[0]["_source"]["text"] == sample_chunks[0].text
    assert actions[1]["_source"]["text"] == sample_chunks[1].text
    assert actions[0]["_source"]["embedding"] == [0.1] * 128
    assert actions[1]["_source"]["embedding"] == [0.2] * 128


# ---- Helper Method Tests ----


def test_generate_id(sample_chunk: Chunk) -> None:
    """Test the _generate_id method for consistency and validity."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    handshake = ElasticHandshake(client=mock_client, index_name="test-id-gen")
    generated_id = handshake._generate_id(0, sample_chunk)
    assert isinstance(generated_id, str)
    # Check if it's a valid UUID string
    try:
        uuid.UUID(generated_id)
    except ValueError:
        pytest.fail(f"Generated ID '{generated_id}' is not a valid UUID.")

    # Check for consistency
    assert handshake._generate_id(0, sample_chunk) == generated_id


# ---- Search Tests ----


def test_search_with_query(
    mock_elastic_client_and_bulk: tuple[MagicMock, MagicMock],
    mock_embeddings: MagicMock,
) -> None:
    """Test the search method with a text query."""
    mock_client, _ = mock_elastic_client_and_bulk

    # Define a mock Elasticsearch search response
    mock_es_response = {
        "hits": {
            "hits": [
                {
                    "_id": "mock-id-1",
                    "_score": 0.99,
                    "_source": {
                        "text": "A relevant document text.",
                        "start_index": 10,
                        "end_index": 40,
                        "token_count": 7,
                    },
                }
            ]
        }
    }
    mock_client.search.return_value = mock_es_response

    handshake = ElasticHandshake()
    results = handshake.search(query="find me something", limit=1)

    # 1. Assert search was called on the client with the correct KNN query
    mock_embeddings.embed.assert_called_once_with("find me something")
    expected_knn_query = {
        "field": "embedding",
        "query_vector": mock_embeddings.embed.return_value.tolist(),
        "k": 1,
        "num_candidates": 100,
    }
    mock_client.search.assert_called_once_with(
        index=handshake.index_name, knn=expected_knn_query, size=1
    )

    # 2. Assert the results are formatted correctly
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "mock-id-1"
    assert result["score"] == 0.99
    assert result["text"] == "A relevant document text."
    assert result["start_index"] == 10
