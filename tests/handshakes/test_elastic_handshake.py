"""Test the ElasticHandshake class."""
import uuid
from unittest.mock import MagicMock, patch, call

import pytest

# Try to import elasticsearch, skip all tests in this module if it's not installed
try:
    import elasticsearch
except ImportError:
    elasticsearch = None

from chonkie.handshakes.elastic import ElasticHandshake
from chonkie.types import Chunk
from chonkie.embeddings import BaseEmbeddings

# Mark all tests in this module to be skipped if elasticsearch is not installed
pytestmark = pytest.mark.skipif(elasticsearch is None, reason="elasticsearch-py not installed")

# ---- Fixtures ----

@pytest.fixture
def mock_elastic_client_and_bulk():
    """Mocks both the Elasticsearch client and the bulk helper function."""
    # Patch the client class to control its instances
    with patch("chonkie.handshakes.elastic.Elasticsearch", autospec=True) as mock_es_class:
        # Create a mock instance that the class will return
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False  # Default: index does not exist
        mock_es_class.return_value = mock_client

        # Patch the bulk helper function used for writing
        with patch("chonkie.handshakes.elastic.bulk", autospec=True) as mock_bulk_func:
            # Default: return 1 success, 0 errors
            mock_bulk_func.return_value = (1, [])
            # Yield both mocks so tests can inspect them
            yield mock_client, mock_bulk_func

@pytest.fixture
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models and to provide consistent results."""
    with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get_embeddings:
        mock_embedding_model = MagicMock(spec=BaseEmbeddings)
        mock_embedding_model.dimension = 128  # Use a consistent dimension for tests
        mock_embedding_model.embed.return_value = [0.1] * 128
        mock_embedding_model.embed_batch.return_value = [[0.1] * 128, [0.2] * 128]
        mock_get_embeddings.return_value = mock_embedding_model
        yield mock_embedding_model

@pytest.fixture
def sample_chunk() -> Chunk:
    """Provide a single sample Chunk."""
    return Chunk(text="This is a test chunk.", start_index=0, end_index=22, token_count=5)

@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide a list of sample Chunks."""
    return [
        Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
        Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
    ]

# ---- Initialization Tests ----

def test_elastic_handshake_init_defaults(mock_elastic_client_and_bulk, mock_embeddings):
    """Test initialization with default parameters."""
    mock_client, _ = mock_elastic_client_and_bulk
    handshake = ElasticHandshake()

    # 1. Assert client was initialized correctly
    mock_client.indices.exists.assert_called_once()
    assert handshake.client == mock_client

    # 2. Assert index was created with correct mapping
    expected_mapping = {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": mock_embeddings.dimension},
            "text": {"type": "text"},
            "start_index": {"type": "integer"},
            "end_index": {"type": "integer"},
            "token_count": {"type": "integer"},
        }
    }
    mock_client.indices.create.assert_called_once_with(index=handshake.index_name, mappings=expected_mapping)

def test_elastic_handshake_init_with_cloud_id(mock_elastic_client_and_bulk):
    """Test initialization using cloud_id and api_key."""
    with patch("chonkie.handshakes.elastic.Elasticsearch") as mock_es_class:
        ElasticHandshake(cloud_id="my_cloud_id", api_key="my_api_key")
        mock_es_class.assert_called_once_with(cloud_id="my_cloud_id", api_key="my_api_key")

def test_elastic_handshake_init_existing_index(mock_elastic_client_and_bulk):
    """Test that an index is NOT created if it already exists."""
    mock_client, _ = mock_elastic_client_and_bulk
    mock_client.indices.exists.return_value = True  # Simulate index already exists

    handshake = ElasticHandshake(index_name="my-existing-index")

    mock_client.indices.exists.assert_called_once_with(index="my-existing-index")
    mock_client.indices.create.assert_not_called()
    assert handshake.index_name == "my-existing-index"

# ---- Write Tests ----

def test_write_single_chunk(mock_elastic_client_and_bulk, mock_embeddings, sample_chunk):
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
    assert action["_source"]["embedding"] == [0.1] * 128  # From mock_embeddings.embed_batch

def test_write_multiple_chunks(mock_elastic_client_and_bulk, mock_embeddings, sample_chunks):
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

def test_generate_id(sample_chunk):
    """Test the _generate_id method for consistency and validity."""
    handshake = ElasticHandshake(index_name="test-id-gen")
    generated_id = handshake._generate_id(0, sample_chunk)
    assert isinstance(generated_id, str)
    # Check if it's a valid UUID string
    try:
        uuid.UUID(generated_id)
    except ValueError:
        pytest.fail(f"Generated ID '{generated_id}' is not a valid UUID.")
    
    # Check for consistency
    assert handshake._generate_id(0, sample_chunk) == generated_id

def test_create_bulk_actions(mock_embeddings, sample_chunks):
    """Test the internal helper for creating bulk actions."""
    handshake = ElasticHandshake(index_name="test-bulk-actions")
    actions = handshake._create_bulk_actions(sample_chunks)

    assert len(actions) == 2
    assert actions[0]["_index"] == "test-bulk-actions"
    assert actions[0]["_source"]["text"] == sample_chunks[0].text
    assert actions[0]["_source"]["embedding"] == [0.1] * 128 # First vector from batch
    assert actions[1]["_source"]["text"] == sample_chunks[1].text
    assert actions[1]["_source"]["embedding"] == [0.2] * 128 # Second vector from batch

# ---- Search Tests ----

def test_search_with_query(mock_elastic_client_and_bulk, mock_embeddings):
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
                        "token_count": 7
                    }
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
        "query_vector": mock_embeddings.embed.return_value,
        "k": 1,
        "num_candidates": 100,
    }
    mock_client.search.assert_called_once_with(index=handshake.index_name, knn=expected_knn_query, size=1)

    # 2. Assert the results are formatted correctly
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "mock-id-1"
    assert result["score"] == 0.99
    assert result["text"] == "A relevant document text."
    assert result["start_index"] == 10