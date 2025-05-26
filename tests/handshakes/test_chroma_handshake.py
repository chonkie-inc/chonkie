"""Test the ChromaHandshake class."""
import uuid
from typing import List
from unittest.mock import Mock, patch

import pytest

# Try to import chromadb, skip tests if unavailable
try:
    import chromadb

    # Keep client types for reference but won't use for isinstance
    from chromadb import Client as ChromaClientType
    from chromadb import PersistentClient as ChromaPersistentClientType

    # Import the specific error type
    from chromadb.errors import NotFoundError
except ImportError:
    chromadb = None
    # Define dummy types/exceptions if import fails
    NotFoundError = type(None) # type: ignore
    ChromaClientType = type(None)
    ChromaPersistentClientType = type(None)

import chromadb

from chonkie import ChromaHandshake
from chonkie.friends.handshakes.chroma import ChromaEmbeddingFunction
from chonkie.types import Chunk

# Mark all tests in this module to be skipped if chromadb is not installed
pytestmark = pytest.mark.skipif(chromadb is None, reason="chromadb not installed")


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models in CI."""
    with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get_embeddings:
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
        mock_get_embeddings.return_value = mock_embedding
        yield mock_get_embeddings


# Sample Chunks for testing
SAMPLE_CHUNKS: List[Chunk] = [
    Chunk(text="This is the first chunk.", start_index=0, end_index=25, token_count=6),
    Chunk(text="This is the second chunk.", start_index=26, end_index=52, token_count=6),
    Chunk(text="Another chunk follows.", start_index=53, end_index=75, token_count=4),
]

def test_chroma_handshake_init_default() -> None:
    """Test ChromaHandshake initialization with default settings (random collection name)."""
    handshake = ChromaHandshake()
    assert handshake.client is not None
    # Check type name instead of using isinstance
    assert type(handshake.client).__name__ == "Client"
    assert isinstance(handshake.collection_name, str)
    assert len(handshake.collection_name) > 0
    # Check if the collection exists
    collection = handshake.client.get_collection(handshake.collection_name)
    assert collection.name == handshake.collection_name
    assert isinstance(handshake.embedding_function, ChromaEmbeddingFunction)
    # Clean up the collection
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_init_specific_collection() -> None:
    """Test ChromaHandshake initialization with a specific collection name."""
    collection_name = "test-collection-chonkie"
    # Ensure collection doesn't exist initially
    # Need to instantiate a client locally for cleanup check
    client = chromadb.Client() # type: ignore[name-defined]
    try:
        client.delete_collection(collection_name)
    except NotFoundError: # Use the correct exception
        pass # It's fine if it doesn't exist

    handshake = ChromaHandshake(collection_name=collection_name)
    assert handshake.collection_name == collection_name
    # Check type name
    assert type(handshake.client).__name__ == "Client"
    collection = client.get_collection(collection_name) # Use local client to check
    assert collection.name == collection_name

    # Test get_or_create_collection logic (initialize again with same name)
    handshake_again = ChromaHandshake(collection_name=collection_name)
    assert handshake_again.collection_name == collection_name
    # Check type name
    assert type(handshake_again.client).__name__ == "Client"
    collection_again = client.get_collection(collection_name) # Use local client to check
    assert collection_again.name == collection_name

    # Clean up using the local client instance
    client.delete_collection(collection_name)

# def test_chroma_handshake_init_persistent_client(tmp_path: Path) -> None:
#     """Test ChromaHandshake initialization with a persistent client path."""
#     persist_dir = tmp_path / "chroma_test_persist"
#     persist_dir.mkdir()
#     collection_name = "persistent-test-collection"

#     # Ensure collection doesn't exist initially in the target path
#     try:
#         # Use the specific type for the temporary client for cleanup check
#         # Need to reference chromadb directly here as ChromaPersistentClientType might be None
#         temp_client = chromadb.PersistentClient(path=str(persist_dir)) # type: ignore[name-defined]
#         try:
#             temp_client.delete_collection(collection_name)
#         except NotFoundError: # Use the correct exception
#              pass # It's fine if it doesn't exist
#         # Explicitly reset the temp client to release resources if needed
#         temp_client.reset() # Good practice for persistent clients
#     except Exception as e:
#          # Handle potential errors during temp client creation/deletion
#          print(f"Warning: Could not ensure clean state for persistent test: {e}")


#     handshake = ChromaHandshake(path=str(persist_dir), collection_name=collection_name)
#     assert handshake.collection_name == collection_name
#     # Check type name instead of using isinstance
#     assert type(handshake.client).__name__ == "PersistentClient"

#     # Verify collection exists by creating a *new* persistent client instance
#     verify_client = chromadb.PersistentClient(path=str(persist_dir)) # type: ignore[name-defined]
#     collection = verify_client.get_collection(collection_name)
#     assert collection.name == collection_name

#     # Clean up using the verification client
#     verify_client.delete_collection(collection_name)
#     verify_client.reset() # Also reset the verification client

def test_chroma_handshake_is_available() -> None:
    """Test the _is_available check."""
    handshake = ChromaHandshake()
    assert handshake._is_available() is True
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_generate_id() -> None:
    """Test the _generate_id method."""
    handshake = ChromaHandshake()
    chunk = SAMPLE_CHUNKS[0]
    index = 0
    expected_id_str = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{handshake.collection_name}::chunk-{index}:{chunk.text}"))
    generated_id = handshake._generate_id(index, chunk)
    assert generated_id == expected_id_str
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_generate_metadata() -> None:
    """Test the _generate_metadata method."""
    handshake = ChromaHandshake()
    chunk = SAMPLE_CHUNKS[0]
    expected_metadata = {
        "start_index": chunk.start_index,
        "end_index": chunk.end_index,
        "token_count": chunk.token_count,
    }
    generated_metadata = handshake._generate_metadata(chunk)
    assert generated_metadata == expected_metadata
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_write_single_chunk() -> None:
    """Test writing a single Chunk."""
    handshake = ChromaHandshake()
    chunk = SAMPLE_CHUNKS[0]
    
    handshake.write(chunk)
    
    # Verify the chunk exists in the collection
    collection = handshake.client.get_collection(handshake.collection_name)
    results = collection.get(include=["documents", "metadatas"])
    
    assert len(results["ids"]) == 1
    assert results["documents"][0] == chunk.text
    assert results["metadatas"][0]["start_index"] == chunk.start_index
    assert results["metadatas"][0]["end_index"] == chunk.end_index
    assert results["metadatas"][0]["token_count"] == chunk.token_count
    
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_write_multiple_chunks() -> None:
    """Test writing multiple Chunks."""
    handshake = ChromaHandshake()
    
    handshake.write(SAMPLE_CHUNKS)
    
    # Verify the chunks exist
    collection = handshake.client.get_collection(handshake.collection_name)
    results = collection.get(include=["documents", "metadatas"])
    
    assert len(results["ids"]) == len(SAMPLE_CHUNKS)
    
    # Check if documents match (order might not be guaranteed by get)
    retrieved_docs = sorted(results["documents"])
    expected_docs = sorted([c.text for c in SAMPLE_CHUNKS])
    assert retrieved_docs == expected_docs

    # Verify metadata integrity by checking one chunk
    # Find the index corresponding to the first sample chunk
    try:
        idx = results["documents"].index(SAMPLE_CHUNKS[0].text)
        assert results["metadatas"][idx]["start_index"] == SAMPLE_CHUNKS[0].start_index
        assert results["metadatas"][idx]["end_index"] == SAMPLE_CHUNKS[0].end_index
        assert results["metadatas"][idx]["token_count"] == SAMPLE_CHUNKS[0].token_count
    except ValueError:
        pytest.fail(f"Document '{SAMPLE_CHUNKS[0].text}' not found in results")

    # Clean up
    handshake.client.delete_collection(handshake.collection_name)


def test_chroma_handshake_write_upsert() -> None:
    """Test the upsert behavior of the write method."""
    handshake = ChromaHandshake()
    chunk = SAMPLE_CHUNKS[0]
    
    # Write the chunk once
    handshake.write(chunk)
    collection = handshake.client.get_collection(handshake.collection_name)
    results_before = collection.get()
    assert len(results_before["ids"]) == 1
    
    # Write the exact same chunk again
    handshake.write(chunk)
    results_after = collection.get()
    
    # Count should remain the same due to upsert
    assert len(results_after["ids"]) == 1
    assert results_before["ids"] == results_after["ids"] # IDs should be identical
    
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

def test_chroma_handshake_repr() -> None:
    """Test the __repr__ method."""
    handshake = ChromaHandshake()
    expected_repr = f"ChromaHandshake(collection_name={handshake.collection_name})"
    assert repr(handshake) == expected_repr
    # Clean up
    handshake.client.delete_collection(handshake.collection_name)

# Note: Testing the embedding function directly might require significant setup 
# or mocking, which was explicitly excluded. The write tests implicitly cover 
# that the embedding function is called during collection creation and potentially during upsert.
# Also, testing _import_dependencies failure case without mocking is difficult.
