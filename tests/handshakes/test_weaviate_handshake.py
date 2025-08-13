"""Test the WeaviateHandshake class."""

import uuid
from typing import List
from unittest.mock import Mock, patch

import pytest

# Try to import weaviate, skip tests if unavailable
try:
    import weaviate
except ImportError:
    weaviate = None

from chonkie.friends.handshakes.weaviate import WeaviateHandshake
from chonkie.types import Chunk

# Mark all tests in this module to be skipped if weaviate is not installed
pytestmark = pytest.mark.skipif(weaviate is None, reason="weaviate not installed")


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models in CI."""
    with patch(
        "chonkie.embeddings.AutoEmbeddings.get_embeddings"
    ) as mock_get_embeddings:
        # Create a mock embedding model
        mock_embedding = Mock()

        # Mock the embed method to return consistent results
        def mock_embed_single(text):
            return [0.1] * 128

        # Mock the embed_batch method to return the right number of embeddings
        def mock_embed_batch(texts):
            return [[0.1] * 128] * len(texts)  # Return one embedding per text

        mock_embedding.embed.side_effect = mock_embed_single
        mock_embedding.embed_batch.side_effect = mock_embed_batch
        mock_embedding.dimension = 128
        # Make sure __str__ returns a string, not a callable
        mock_embedding.__str__ = Mock(return_value="MockEmbeddingModel")
        mock_get_embeddings.return_value = mock_embedding
        yield mock_get_embeddings


@pytest.fixture(autouse=True)
def mock_weaviate_client():
    """Mock Weaviate client to avoid needing a real Weaviate instance."""
    with patch("weaviate.connect_to_custom", autospec=True) as mock_connect:
        # Create a mock client
        mock_client = Mock()

        # Mock collections
        mock_collections = Mock()
        mock_client.collections = mock_collections

        # Mock collection
        mock_collection = Mock()
        mock_collections.get.return_value = mock_collection

        # Mock batch with proper context manager behavior
        mock_batch = Mock()
        mock_collection.batch = mock_batch

        # Create a context manager for batch.fixed_size()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_batch)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_batch.fixed_size.return_value = mock_context_manager

        # Set up batch properties
        mock_batch.failed_objects = []
        # Set number_errors to 0 by default for all tests
        mock_batch.number_errors = 0

        # Mock exists method
        mock_collections.exists.return_value = False

        # Mock create method
        mock_collections.create.return_value = mock_collection

        # Return the mock client
        mock_connect.return_value = mock_client
        yield mock_client


# Sample Chunks for testing
SAMPLE_CHUNKS: List[Chunk] = [
    Chunk(text="This is the first chunk.", start_index=0, end_index=25, token_count=6),
    Chunk(
        text="This is the second chunk.", start_index=26, end_index=52, token_count=6
    ),
    Chunk(text="Another chunk follows.", start_index=53, end_index=75, token_count=4),
]


def test_weaviate_handshake_init_default(mock_weaviate_client) -> None:
    """Test WeaviateHandshake initialization with default settings (random collection name)."""
    handshake = WeaviateHandshake()
    assert handshake.client is not None
    assert isinstance(handshake.collection_name, str)
    assert len(handshake.collection_name) > 0
    # Check if the collection was created
    mock_weaviate_client.collections.create.assert_called_once()
    assert handshake.vector_dimensions == 128


def test_weaviate_handshake_init_specific_collection(mock_weaviate_client) -> None:
    """Test WeaviateHandshake initialization with a specific collection name."""
    collection_name = "test-collection-chonkie"

    # Set up mock to indicate collection doesn't exist
    mock_weaviate_client.collections.exists.return_value = False

    handshake = WeaviateHandshake(collection_name=collection_name)
    assert handshake.collection_name == collection_name

    # Check if the collection was created
    mock_weaviate_client.collections.create.assert_called_once()

    # Test get_or_create_collection logic (initialize again with same name)
    # Reset mock and set it to indicate collection exists
    mock_weaviate_client.collections.create.reset_mock()
    mock_weaviate_client.collections.exists.return_value = True

    handshake_again = WeaviateHandshake(collection_name=collection_name)
    assert handshake_again.collection_name == collection_name

    # Check that collection was not created again
    mock_weaviate_client.collections.create.assert_not_called()


def test_weaviate_handshake_is_available(mock_weaviate_client) -> None:
    """Test the _is_available check."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)
    assert handshake._is_available() is True


def test_weaviate_handshake_custom_connection_params(mock_weaviate_client) -> None:
    """Test WeaviateHandshake initialization with custom connection parameters."""
    # Reset mock to clear previous call records
    mock_weaviate_client.reset_mock()

    # Get the mock object for connect_to_custom
    from unittest.mock import patch

    # Use patch context manager to temporarily replace weaviate.connect_to_custom
    with patch(
        "weaviate.connect_to_custom", return_value=mock_weaviate_client
    ) as mock_connect:
        # Test with custom HTTP and gRPC parameters
        WeaviateHandshake(
            url="https://example.com:8443",
            http_secure=True,
            grpc_host="grpc.example.com",
            grpc_port=50052,
            grpc_secure=True,
        )

        # Check that connect_to_custom was called with the correct parameters
        mock_connect.assert_called_once()

        call_kwargs = mock_connect.call_args.kwargs

        assert call_kwargs["http_host"] == "example.com"
        assert call_kwargs["http_port"] == 8443
        assert call_kwargs["http_secure"] is True
        assert call_kwargs["grpc_host"] == "grpc.example.com"
        assert call_kwargs["grpc_port"] == 50052
        assert call_kwargs["grpc_secure"] is True


def test_weaviate_handshake_cloud_init():
    """Test WeaviateHandshake initialization with cloud parameters (url and api_key)."""
    from unittest.mock import Mock, patch

    mock_client = Mock()
    with (
        patch(
            "weaviate.connect_to_weaviate_cloud", return_value=mock_client
        ) as mock_cloud_connect,
        patch("weaviate.auth.Auth.api_key", return_value="mock_auth") as mock_auth,
    ):
        url = "https://cloud.weaviate.io"
        key = "test-api-key"
        handshake = WeaviateHandshake(url=url, api_key=key)
        mock_cloud_connect.assert_called_once_with(
            cluster_url=url, auth_credentials="mock_auth"
        )
        mock_auth.assert_called_once_with(key)
        assert handshake.client == mock_client


def test_weaviate_handshake_generate_id(mock_weaviate_client) -> None:
    """Test the _generate_id method."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)
    chunk = SAMPLE_CHUNKS[0]
    index = 0
    expected_id_str = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            f"{handshake.collection_name}::chunk-{index}:{chunk.text}",
        )
    )
    generated_id = handshake._generate_id(index, chunk)
    assert generated_id == expected_id_str


def test_weaviate_handshake_generate_properties(mock_weaviate_client) -> None:
    """Test the _generate_properties method."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)
    chunk = SAMPLE_CHUNKS[0]
    expected_properties = {
        "text": chunk.text,
        "start_index": chunk.start_index,
        "end_index": chunk.end_index,
        "token_count": chunk.token_count,
        "chunk_type": type(chunk).__name__,
    }
    generated_properties = handshake._generate_properties(chunk)
    assert generated_properties == expected_properties


def test_weaviate_handshake_write_single_chunk(mock_weaviate_client) -> None:
    """Test writing a single Chunk."""
    handshake = WeaviateHandshake()
    chunk = SAMPLE_CHUNKS[0]

    # Set up mock batch
    mock_batch = mock_weaviate_client.collections.get.return_value.batch.fixed_size.return_value.__enter__.return_value

    # Call write method
    result = handshake.write(chunk)

    # Check that add_object was called once
    mock_batch.add_object.assert_called_once()

    # Check that the result is a list with one ID
    assert isinstance(result, list)
    assert len(result) == 1

    # Check that the ID is a string
    assert isinstance(result[0], str)


def test_weaviate_handshake_write_multiple_chunks(mock_weaviate_client) -> None:
    """Test writing multiple Chunks."""
    handshake = WeaviateHandshake()

    # Set up mock batch
    mock_batch = mock_weaviate_client.collections.get.return_value.batch.fixed_size.return_value.__enter__.return_value

    # Call write method
    result = handshake.write(SAMPLE_CHUNKS)

    # Check that add_object was called for each chunk
    assert mock_batch.add_object.call_count == len(SAMPLE_CHUNKS)

    # Check that the result is a list with the right number of IDs
    assert isinstance(result, list)
    assert len(result) == len(SAMPLE_CHUNKS)

    # Check that each ID is a string
    for id_str in result:
        assert isinstance(id_str, str)


def test_weaviate_handshake_write_upsert(mock_weaviate_client) -> None:
    """Test the upsert behavior of the write method."""
    handshake = WeaviateHandshake()
    chunk = SAMPLE_CHUNKS[0]

    # Set up mock batch
    mock_batch = mock_weaviate_client.collections.get.return_value.batch.fixed_size.return_value.__enter__.return_value

    # Call write method twice with the same chunk
    first_result = handshake.write(chunk)
    mock_batch.add_object.reset_mock()  # Reset the mock to count calls separately
    second_result = handshake.write(chunk)

    # Check that add_object was called once for each write
    assert mock_batch.add_object.call_count == 1

    # Check that the IDs are the same
    assert first_result[0] == second_result[0]


def test_weaviate_handshake_repr(mock_weaviate_client) -> None:
    """Test the __repr__ method."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)
    expected_repr = f"WeaviateHandshake(collection_name={handshake.collection_name}, vector_dimensions={handshake.vector_dimensions})"
    assert repr(handshake) == expected_repr


def test_weaviate_handshake_delete_collection(mock_weaviate_client) -> None:
    """Test the delete_collection method."""
    handshake = WeaviateHandshake()

    # Set up mock to indicate collection exists
    mock_weaviate_client.collections.exists.return_value = True

    # Call delete_collection method
    handshake.delete_collection()

    # Check that delete was called
    mock_weaviate_client.collections.delete.assert_called_once_with(
        handshake.collection_name
    )


def test_weaviate_handshake_get_collection_info(mock_weaviate_client) -> None:
    """Test the get_collection_info method."""
    handshake = WeaviateHandshake()

    # Set up mock to indicate collection exists
    mock_weaviate_client.collections.exists.return_value = True

    # Set up mock schema
    mock_schema = Mock()
    mock_schema.properties = [
        Mock(name="text"),
        Mock(name="start_index"),
        Mock(name="end_index"),
        Mock(name="token_count"),
        Mock(name="chunk_type"),
    ]
    mock_weaviate_client.collections.get.return_value.config.get.return_value = (
        mock_schema
    )

    # Call get_collection_info method
    info = handshake.get_collection_info()

    # Check the result
    assert info["name"] == handshake.collection_name
    assert info["exists"] is True
    assert info["vector_dimensions"] == 128
    assert len(info["properties"]) == 5
    assert "text" in info["properties"]
    assert "start_index" in info["properties"]
    assert "end_index" in info["properties"]
    assert "token_count" in info["properties"]
    assert "chunk_type" in info["properties"]


def test_weaviate_handshake_batch_error_handling(mock_weaviate_client) -> None:
    """Test error handling during batch processing."""
    handshake = WeaviateHandshake()

    # Set up mock batch with failed objects
    mock_batch = mock_weaviate_client.collections.get.return_value.batch.fixed_size.return_value.__enter__.return_value
    mock_failed_object = Mock()
    mock_failed_object.error = "Test error"
    mock_batch.failed_objects = [mock_failed_object] * 11  # More than max_errors
    mock_batch.number_errors = 11  # Set number_errors to match failed_objects count

    # Call write method and expect RuntimeError
    with pytest.raises(RuntimeError):
        handshake.write(SAMPLE_CHUNKS)


def test_weaviate_handshake_generate_properties_with_optional_fields(
    mock_weaviate_client,
) -> None:
    """Test the _generate_properties method with optional fields."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)

    # Create a chunk with optional properties
    chunk = SAMPLE_CHUNKS[0]

    # Add optional properties
    chunk.sentences = ["This is a sentence."]
    chunk.words = ["This", "is", "a", "sentence"]
    chunk.language = "en"

    # Generate properties
    properties = handshake._generate_properties(chunk)

    # Verify basic properties
    assert properties["text"] == chunk.text
    assert properties["start_index"] == chunk.start_index
    assert properties["end_index"] == chunk.end_index
    assert properties["token_count"] == chunk.token_count
    assert properties["chunk_type"] == type(chunk).__name__

    # Verify optional properties
    assert properties["sentence_count"] == len(chunk.sentences)
    assert properties["word_count"] == len(chunk.words)
    assert properties["language"] == chunk.language


def test_weaviate_handshake_close(mock_weaviate_client) -> None:
    """Test the close method."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)
    handshake.close()
    mock_weaviate_client.close.assert_called_once()


def test_weaviate_handshake_call(mock_weaviate_client) -> None:
    """Test the __call__ method."""
    handshake = WeaviateHandshake(client=mock_weaviate_client)

    # Mock the write method
    from unittest.mock import Mock

    handshake.write = Mock(return_value=["id1", "id2", "id3"])

    # Call the __call__ method
    result = handshake(SAMPLE_CHUNKS)

    # Verify write method was called
    handshake.write.assert_called_once_with(SAMPLE_CHUNKS)
    assert result == ["id1", "id2", "id3"]

    # Test with a single Chunk
    handshake.write.reset_mock()
    result = handshake(SAMPLE_CHUNKS[0])
    handshake.write.assert_called_once_with(SAMPLE_CHUNKS[0])

    # Test with invalid input
    with pytest.raises(TypeError):
        handshake(123)  # Not a Chunk or Sequence[Chunk]
