"""Test the MilvusHandshake class."""

from typing import Generator
from unittest.mock import MagicMock

import pytest
import numpy as np

# Try to import pymilvus, skip all tests in this module if it's not installed
try:
    import pymilvus
except ImportError:
    pymilvus = None

from chonkie.embeddings import BaseEmbeddings
from chonkie.handshakes.milvus import MilvusHandshake
from chonkie.types import Chunk

pytestmark = pytest.mark.skipif(pymilvus is None, reason="pymilvus-client not installed")

# ---- Fixtures ----

@pytest.fixture
def mock_pymilvus_modules(monkeypatch) -> MagicMock:
    """Mocks all required pymilvus modules and classes."""
    mock_connections = MagicMock()
    mock_utility = MagicMock()
    mock_collection = MagicMock()
    mock_collection_schema = MagicMock()
    mock_field_schema = MagicMock()

    # Default behaviors
    mock_utility.has_collection.return_value = False
    mock_collection.return_value.insert.return_value = MagicMock(insert_count=2)

    monkeypatch.setattr("chonkie.handshakes.milvus.connections", mock_connections)
    monkeypatch.setattr("chonkie.handshakes.milvus.utility", mock_utility)
    monkeypatch.setattr("chonkie.handshakes.milvus.Collection", mock_collection)
    monkeypatch.setattr("chonkie.handshakes.milvus.CollectionSchema", mock_collection_schema)
    monkeypatch.setattr("chonkie.handshakes.milvus.FieldSchema", mock_field_schema)

    # Return the top-level utility mock for assertions
    return mock_utility


@pytest.fixture
def mock_embeddings() -> Generator[MagicMock, None, None]:
    """Mock AutoEmbeddings to provide consistent numpy array results."""
    with patch("chonkie.embeddings.AutoEmbeddings.get_embeddings") as mock_get_embeddings:
        mock_embedding_model = MagicMock(spec=BaseEmbeddings)
        mock_embedding_model.dimension = 128
        mock_embedding_model.embed.return_value = np.array([0.1] * 128)
        # embed_batch should return a list of numpy arrays or a 2D numpy array
        mock_embedding_model.embed_batch.return_value = np.array([[0.1] * 128, [0.2] * 128])
        mock_get_embeddings.return_value = mock_embedding_model
        yield mock_embedding_model


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide a list of sample Chunks."""
    return [
        Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
        Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
    ]

# ---- Tests ----

def test_milvus_handshake_init_creates_collection(mock_pymilvus_modules, mock_embeddings):
    """Test that a new collection and index are created if one doesn't exist."""
    mock_utility = mock_pymilvus_modules
    mock_utility.has_collection.return_value = False

    handshake = MilvusHandshake()

    mock_utility.has_collection.assert_called_with(handshake.collection_name, using=handshake.alias)
    assert handshake.collection.create_index.call_count == 1
    handshake.collection.load.assert_called_once()


def test_milvus_handshake_init_uses_existing_collection(mock_pymilvus_modules, mock_embeddings):
    """Test that a new collection is NOT created if it already exists."""
    mock_utility = mock_pymilvus_modules
    mock_utility.has_collection.return_value = True

    handshake = MilvusHandshake(collection_name="my-existing-collection")

    mock_utility.has_collection.assert_called_with("my-existing-collection", using=handshake.alias)
    assert handshake.collection.create_index.call_count == 0
    handshake.collection.load.assert_called_once()


def test_write_multiple_chunks(mock_pymilvus_modules, mock_embeddings, sample_chunks):
    """Test writing multiple chunks with correct columnar formatting."""
    handshake = MilvusHandshake()
    handshake.write(sample_chunks)

    handshake.collection.insert.assert_called_once()
    handshake.collection.flush.assert_called_once()

    # Verify the data was passed in the correct columnar format
    args, _ = handshake.collection.insert.call_args
    inserted_data = args[0]
    assert len(inserted_data) == 5  # pk (auto) + 4 metadata fields + embedding
    # Check texts
    assert inserted_data[0] == [c.text for c in sample_chunks]
    # Check embeddings
    assert np.array_equal(inserted_data[4], mock_embeddings.embed_batch.return_value)


def test_search_with_query(mock_pymilvus_modules, mock_embeddings):
    """Test the search method formats the query and parses results correctly."""
    # Define a mock Milvus search response
    mock_hit = MagicMock()
    mock_hit.id = "mock_pk_1"
    mock_hit.distance = 0.98
    mock_hit.entity = {"text": "A relevant doc", "start_index": 0}
    mock_results = [[mock_hit]]
    
    # Get the mock Collection instance
    handshake = MilvusHandshake()
    handshake.collection.search.return_value = mock_results

    results = handshake.search(query="find me something", limit=1)

    # 1. Assert search was called on the collection with the correct parameters
    mock_embeddings.embed.assert_called_once_with("find me something")
    expected_query_vector = [mock_embeddings.embed.return_value.tolist()]

    handshake.collection.search.assert_called_once()
    search_args, search_kwargs = handshake.collection.search.call_args
    assert search_kwargs["data"] == expected_query_vector
    assert search_kwargs["limit"] == 1
    assert "text" in search_kwargs["output_fields"]

    # 2. Assert the results are formatted correctly
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "mock_pk_1"
    assert result["score"] == 0.98
    assert result["text"] == "A relevant doc"
    assert result["start_index"] == 0