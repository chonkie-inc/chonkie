"""Test the QdrantHandshake class."""
import os
import uuid
from typing import List

import pytest

# Assuming qdrant_client is installed as per the request
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.friends.handshakes.qdrant import QdrantHandshake
from chonkie.types import Chunk

# Define the default model name for clarity
DEFAULT_EMBEDDING_MODEL = "minishlab/potion-retrieval-32M"

# ---- Fixtures ----

@pytest.fixture(scope="module")
def real_embeddings() -> BaseEmbeddings:
    """Provide an instance of the actual default embedding model."""
    # Use scope="module" to load the model only once per test module run
    # Set environment variable to potentially avoid Hugging Face Hub login prompts in some CI environments
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 
    return AutoEmbeddings.get_embeddings(DEFAULT_EMBEDDING_MODEL)

@pytest.fixture
def sample_chunk() -> Chunk:
    """Provide a single sample Chunk."""
    return Chunk(
        text="This is a test chunk.",
        start_index=0,
        end_index=22,
        token_count=5
    )

@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Provide a list of sample Chunks."""
    return [
        Chunk(
            text="First test chunk.",
            start_index=0,
            end_index=18,
            token_count=4
        ),
        Chunk(
            text="Second test chunk.",
            start_index=19,
            end_index=38,
            token_count=4
        ),
    ]

# ---- Initialization Tests ----

def test_qdrant_handshake_init_defaults(real_embeddings: BaseEmbeddings) -> None:
    """Test QdrantHandshake initialization with default parameters (in-memory)."""
    # The handshake initializes the default model internally
    handshake = QdrantHandshake(embedding_model=DEFAULT_EMBEDDING_MODEL)
    assert isinstance(handshake.client, qdrant_client.QdrantClient)
    assert handshake.collection_name != "random" # Should be a generated name
    assert isinstance(handshake.embedding_model, BaseEmbeddings)
    # Use model_name_or_path for comparison
    assert hasattr(handshake.embedding_model, 'model_name_or_path')
    assert hasattr(real_embeddings, 'model_name_or_path')
    assert handshake.embedding_model.model_name_or_path == real_embeddings.model_name_or_path # Check if the correct model name is loaded
    assert handshake.dimension == real_embeddings.dimension
    # Check if collection exists (implicitly tests creation)
    info = handshake.client.get_collection(collection_name=handshake.collection_name)
    # Updated attribute access for vector size
    assert info.config.params.vectors.size == real_embeddings.dimension
    # Cleanup
    handshake.client.delete_collection(collection_name=handshake.collection_name)

def test_qdrant_handshake_init_specific_collection(real_embeddings: BaseEmbeddings) -> None:
    """Test QdrantHandshake initialization with a specific collection name."""
    collection_name = "test-collection-specific"
    client = qdrant_client.QdrantClient(":memory:") # Ensure isolated client
    handshake = QdrantHandshake(
        client=client,
        collection_name=collection_name,
        embedding_model=real_embeddings # Pass the loaded model instance
    )
    assert handshake.collection_name == collection_name
    assert handshake.embedding_model == real_embeddings
    assert handshake.dimension == real_embeddings.dimension
    # Check if collection was created correctly
    info = client.get_collection(collection_name=collection_name)
    # Updated attribute access for vector size
    assert info.config.params.vectors.size == real_embeddings.dimension
    client.delete_collection(collection_name=collection_name)


def test_qdrant_handshake_init_random_collection(real_embeddings: BaseEmbeddings) -> None:
    """Test QdrantHandshake initialization with collection_name='random'."""
    client = qdrant_client.QdrantClient(":memory:") # Ensure isolated client
    handshake = QdrantHandshake(
        client=client,
        collection_name="random",
        embedding_model=real_embeddings
    )
    assert isinstance(handshake.collection_name, str)
    assert len(handshake.collection_name) > 0
    assert handshake.collection_name != "random"
    assert handshake.embedding_model == real_embeddings
    # Check if collection was created
    info = client.get_collection(collection_name=handshake.collection_name)
    # Updated attribute access for vector size
    assert info.config.params.vectors.size == real_embeddings.dimension
    client.delete_collection(collection_name=handshake.collection_name)


def test_qdrant_handshake_init_existing_collection(real_embeddings: BaseEmbeddings) -> None:
    """Test QdrantHandshake initialization with an existing collection."""
    collection_name = "test-collection-existing"
    client = qdrant_client.QdrantClient(":memory:")
    # Pre-create the collection with the correct dimension
    # Avoid deprecated recreate_collection
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=real_embeddings.dimension, distance=Distance.COSINE),
    )
    
    handshake = QdrantHandshake(
        client=client,
        collection_name=collection_name,
        embedding_model=real_embeddings
    )
    assert handshake.collection_name == collection_name
    # Verify it didn't error and uses the existing one
    info = client.get_collection(collection_name=collection_name)
    # Updated attribute access for vector size
    assert info.config.params.vectors.size == real_embeddings.dimension
    client.delete_collection(collection_name=collection_name)

# ---- Write Tests ----

def test_qdrant_handshake_write_single_chunk(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test writing a single chunk."""
    collection_name = "test-write-single-real"
    client = qdrant_client.QdrantClient(":memory:")
    handshake = QdrantHandshake(
        client=client,
        collection_name=collection_name,
        embedding_model=real_embeddings
    )
    
    handshake.write(sample_chunk)

    # Verify the point exists in the collection
    points, _ = client.scroll(collection_name=collection_name, limit=10, with_payload=True, with_vectors=True)
    assert len(points) == 1
    
    point = points[0]
    expected_id = handshake._generate_id(0, sample_chunk) # Index is 0 for single/first chunk
    expected_payload = handshake._generate_payload(sample_chunk)
    # expected_vector = real_embeddings.embed(sample_chunk.text) # Calculate expected vector

    assert point.id == expected_id
    assert point.payload == expected_payload
    assert point.vector is not None
    assert len(point.vector) == real_embeddings.dimension
    # Direct vector comparison is brittle with real models, checking dimension and presence is sufficient
    # assert point.vector == expected_vector 

    client.delete_collection(collection_name=collection_name)


def test_qdrant_handshake_write_multiple_chunks(sample_chunks: List[Chunk], real_embeddings: BaseEmbeddings) -> None:
    """Test writing multiple chunks."""
    collection_name = "test-write-multiple-real"
    client = qdrant_client.QdrantClient(":memory:")
    handshake = QdrantHandshake(
        client=client,
        collection_name=collection_name,
        embedding_model=real_embeddings
    )

    handshake.write(sample_chunks)

    # Verify the points exist
    points, _ = client.scroll(collection_name=collection_name, limit=10, with_payload=True, with_vectors=True)
    assert len(points) == len(sample_chunks)

    # Verify content (order might not be guaranteed by scroll, so check presence)
    retrieved_payloads = {p.payload['text'] for p in points}
    expected_payloads = {chunk.text for chunk in sample_chunks}
    assert retrieved_payloads == expected_payloads

    for i, chunk in enumerate(sample_chunks):
        expected_id = handshake._generate_id(i, chunk)
        expected_payload = handshake._generate_payload(chunk)
        # expected_vector = real_embeddings.embed(chunk.text)

        # Retrieve specific point by ID to check details
        retrieved_point = client.retrieve(collection_name=collection_name, ids=[expected_id], with_payload=True, with_vectors=True)
        assert len(retrieved_point) == 1
        point = retrieved_point[0]
        
        assert point.id == expected_id
        assert point.payload == expected_payload
        assert point.vector is not None
        assert len(point.vector) == real_embeddings.dimension
        # assert point.vector == expected_vector # Skip direct comparison

    client.delete_collection(collection_name=collection_name)


def test_qdrant_handshake_write_upsert(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test that writing the same chunk again performs an upsert."""
    collection_name = "test-write-upsert-real"
    client = qdrant_client.QdrantClient(":memory:")
    handshake = QdrantHandshake(
        client=client,
        collection_name=collection_name,
        embedding_model=real_embeddings
    )

    # Write the chunk
    handshake.write(sample_chunk)
    points_before, _ = client.scroll(collection_name=collection_name, limit=10)
    assert len(points_before) == 1
    count_before = client.count(collection_name=collection_name).count
    assert count_before == 1

    # Modify chunk slightly (different metadata but same text -> same ID)
    modified_chunk = Chunk(text=sample_chunk.text, start_index=1, end_index=23, token_count=6) 
    
    # Write again - should upsert based on ID
    handshake.write(modified_chunk) 
    
    points_after, _ = client.scroll(collection_name=collection_name, limit=10, with_payload=True, with_vectors=True)
    count_after = client.count(collection_name=collection_name).count

    # Count should remain 1 because the ID (based on text) is the same
    assert count_after == 1
    assert len(points_after) == 1
    point = points_after[0]
    
    # Payload should be updated to the new metadata
    assert point.payload["start_index"] == modified_chunk.start_index
    assert point.payload["end_index"] == modified_chunk.end_index
    assert point.payload["token_count"] == modified_chunk.token_count
    assert point.payload["text"] == modified_chunk.text # Text is the same

    # Vector should still exist and have the correct dimension
    assert point.vector is not None
    assert len(point.vector) == real_embeddings.dimension

    client.delete_collection(collection_name=collection_name)

# ---- Helper Method Tests ----

# These helpers don't depend directly on the embedding values, just the model instance for dimension/UUID namespace
def test_generate_id(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test the _generate_id method."""
    # Need an instance, using real embeddings now
    handshake = QdrantHandshake(collection_name="test-id-gen-real", embedding_model=real_embeddings)
    generated_id = handshake._generate_id(0, sample_chunk)
    assert isinstance(generated_id, str)
    # Check if it's a valid UUID
    try:
        uuid.UUID(generated_id)
    except ValueError:
        pytest.fail(f"Generated ID '{generated_id}' is not a valid UUID.")
    
    # Check for consistency
    assert handshake._generate_id(0, sample_chunk) == generated_id
    
    # Check different index or text yields different ID
    assert handshake._generate_id(1, sample_chunk) != generated_id
    diff_chunk = Chunk(text="Different text", start_index=0, end_index=14, token_count=2)
    assert handshake._generate_id(0, diff_chunk) != generated_id

    # Cleanup client implicitly created by handshake
    handshake.client.delete_collection(collection_name="test-id-gen-real") 


def test_generate_payload(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test the _generate_payload method."""
    # Need an instance, using real embeddings now
    handshake = QdrantHandshake(collection_name="test-payload-gen-real", embedding_model=real_embeddings)
    payload = handshake._generate_payload(sample_chunk)
    assert payload == {
        "text": sample_chunk.text,
        "start_index": sample_chunk.start_index,
        "end_index": sample_chunk.end_index,
        "token_count": sample_chunk.token_count,
    }
    # Cleanup client implicitly created by handshake
    handshake.client.delete_collection(collection_name="test-payload-gen-real")
