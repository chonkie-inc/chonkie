"""Test the PgvectorHandshake class."""
import uuid
from typing import List
from unittest.mock import Mock, patch

import pytest

# Try to import vecs, skip tests if unavailable
try:
    import vecs
    vecs_available = True
except ImportError:
    vecs = None
    vecs_available = False

from chonkie.types import Chunk

if vecs_available:
    from chonkie.friends.handshakes.pgvector import PgvectorHandshake

# Mark all tests in this module to be skipped if vecs is not installed
pytestmark = pytest.mark.skipif(not vecs_available, reason="vecs not installed")


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock dependencies to avoid real connections and downloads in CI."""
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
        
        # Create a mock vecs client and collection
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.dimension = 128
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None
        mock_collection.upsert.return_value = None
        mock_collection.query.return_value = []
        mock_collection.create_index.return_value = None
        
        # Create a mock create_client function
        mock_create_client = Mock(return_value=mock_client)
        
        yield {
            'embeddings': mock_get_embeddings,
            'client': mock_client,
            'collection': mock_collection,
            'create_client': mock_create_client
        }


# Remove the old fixture since we're using autouse mock_dependencies now


# Sample Chunks for testing
SAMPLE_CHUNKS: List[Chunk] = [
    Chunk(text="This is the first chunk.", start_index=0, end_index=25, token_count=6),
    Chunk(text="This is the second chunk.", start_index=26, end_index=52, token_count=6),
    Chunk(text="Another chunk follows.", start_index=53, end_index=75, token_count=4),
]


def test_pgvector_handshake_is_available():
    """Test the _is_available check."""
    with patch('chonkie.friends.handshakes.pgvector.importutil.find_spec') as mock_find_spec:
        # Mock vecs as available
        mock_find_spec.side_effect = lambda name: Mock() if name == 'vecs' else None
        
        handshake = PgvectorHandshake.__new__(PgvectorHandshake)  # Create without calling __init__
        assert handshake._is_available() is True
        
        # Test when vecs is not available
        mock_find_spec.side_effect = lambda name: None
        assert handshake._is_available() is False


def test_pgvector_handshake_init_with_individual_params(mock_dependencies):
    """Test PgvectorHandshake initialization with individual connection parameters."""
    mock_create_client = mock_dependencies['create_client']
    mock_client = mock_dependencies['client']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password",
            collection_name="test_collection"
        )
        
        # Verify client was created with correct connection string
        mock_create_client.assert_called_once_with("postgresql://test_user:test_password@localhost:5432/test_db")
        assert handshake.client == mock_client
        assert handshake.collection_name == "test_collection"
        assert handshake.vector_dimensions == 128  # From mocked embedding


def test_pgvector_handshake_init_with_connection_string(mock_dependencies):
    """Test PgvectorHandshake initialization with connection string."""
    mock_client = mock_dependencies['client']
    mock_create_client = mock_dependencies['create_client']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        connection_string = "postgresql://user:pass@host:5432/db"
        handshake = PgvectorHandshake(connection_string=connection_string)
        
        # Verify client was created with provided connection string
        mock_create_client.assert_called_with(connection_string)
        assert handshake.client == mock_client


def test_pgvector_handshake_init_with_existing_client(mock_dependencies):
    """Test PgvectorHandshake initialization with existing vecs client."""
    mock_client = mock_dependencies['client']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'):
        handshake = PgvectorHandshake(client=mock_client)
        
        # Verify provided client was used directly
        assert handshake.client == mock_client


def test_pgvector_handshake_init_custom_vector_dimensions(mock_dependencies):
    """Test PgvectorHandshake initialization with custom vector dimensions."""
    mock_create_client = mock_dependencies['create_client']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(
            host="localhost",
            vector_dimensions=256
        )
        
        assert handshake.vector_dimensions == 256


def test_pgvector_handshake_dimension_detection_from_property(mock_dependencies):
    """Test that dimension is detected from embedding model's dimension property when available."""
    mock_embeddings = mock_dependencies['embeddings']
    mock_create_client = mock_dependencies['create_client']
    
    # Create a mock embedding with dimension property
    mock_embedding_with_dim = Mock()
    mock_embedding_with_dim.dimension = 384
    mock_embedding_with_dim.embed.return_value = [0.1] * 384
    mock_embeddings.return_value = mock_embedding_with_dim
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Should use dimension property, not call embed()
        assert handshake.vector_dimensions == 384
        mock_embedding_with_dim.embed.assert_not_called()


def test_pgvector_handshake_dimension_detection_from_test_embedding(mock_dependencies):
    """Test that dimension is detected from test embedding when dimension property is not available."""
    mock_embeddings = mock_dependencies['embeddings']
    mock_create_client = mock_dependencies['create_client']
    
    # Create a mock embedding without dimension property
    mock_embedding_no_dim = Mock()
    # Remove dimension property to simulate embeddings that don't have it
    if hasattr(mock_embedding_no_dim, 'dimension'):
        delattr(mock_embedding_no_dim, 'dimension')
    mock_embedding_no_dim.embed.return_value = [0.1] * 512
    mock_embeddings.return_value = mock_embedding_no_dim
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Should call embed() to determine dimension
        assert handshake.vector_dimensions == 512
        mock_embedding_no_dim.embed.assert_called_once_with("test")


def test_pgvector_handshake_dimension_detection_from_test_embedding_when_none(mock_dependencies):
    """Test that dimension is detected from test embedding when dimension property is None."""
    mock_embeddings = mock_dependencies['embeddings']
    mock_create_client = mock_dependencies['create_client']
    
    # Create a mock embedding with dimension property set to None
    mock_embedding_none_dim = Mock()
    mock_embedding_none_dim.dimension = None
    mock_embedding_none_dim.embed.return_value = [0.1] * 768
    mock_embeddings.return_value = mock_embedding_none_dim
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Should call embed() because dimension is None
        assert handshake.vector_dimensions == 768
        mock_embedding_none_dim.embed.assert_called_once_with("test")


def test_pgvector_handshake_generate_id(mock_dependencies):
    """Test the _generate_id method."""
    mock_client = mock_dependencies['client']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        chunk = SAMPLE_CHUNKS[0]
        index = 0
        
        expected_id_str = str(uuid.uuid5(
            uuid.NAMESPACE_OID, 
            f"{handshake.collection_name}::chunk-{index}:{chunk.text}"
        ))
        generated_id = handshake._generate_id(index, chunk)
        assert generated_id == expected_id_str


def test_pgvector_handshake_generate_metadata(mock_dependencies):
    """Test the _generate_metadata method."""
    mock_client = mock_dependencies['client']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        chunk = SAMPLE_CHUNKS[0]
        
        metadata = handshake._generate_metadata(chunk)
        
        assert metadata["text"] == chunk.text
        assert metadata["start_index"] == chunk.start_index
        assert metadata["end_index"] == chunk.end_index
        assert metadata["token_count"] == chunk.token_count
        assert metadata["chunk_type"] == "Chunk"


def test_pgvector_handshake_generate_metadata_with_attributes(mock_dependencies):
    """Test the _generate_metadata method with chunk attributes."""
    mock_client = mock_dependencies['client']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Create a mock chunk with additional attributes
        chunk = Mock()
        chunk.text = "Sample text"
        chunk.start_index = 0
        chunk.end_index = 11
        chunk.token_count = 2
        chunk.sentences = ["Sentence 1", "Sentence 2"]
        chunk.words = ["Sample", "text"]
        chunk.language = "en"
        
        metadata = handshake._generate_metadata(chunk)
        
        assert metadata["text"] == "Sample text"
        assert metadata["chunk_type"] == "Mock"
        assert metadata["sentence_count"] == 2
        assert metadata["word_count"] == 2
        assert metadata["language"] == "en"


def test_pgvector_handshake_write_single_chunk(mock_dependencies):
    """Test writing a single Chunk."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        chunk = SAMPLE_CHUNKS[0]
        
        result = handshake.write(chunk)
        
        # Should return list of IDs
        assert isinstance(result, list)
        assert len(result) == 1
        
        # Verify collection upsert was called
        mock_collection.upsert.assert_called_once()
        
        # Check the upsert call arguments
        call_args = mock_collection.upsert.call_args
        records = call_args[1]['records']  # records is passed as keyword argument
        assert len(records) == 1
        
        # Verify record structure: (id, vector, metadata)
        record = records[0]
        assert len(record) == 3
        assert isinstance(record[0], str)  # ID
        assert isinstance(record[1], list)  # vector
        assert isinstance(record[2], dict)  # metadata


def test_pgvector_handshake_write_multiple_chunks(mock_dependencies):
    """Test writing multiple Chunks."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        
        result = handshake.write(SAMPLE_CHUNKS)
        
        # Should return list of IDs equal to number of chunks
        assert isinstance(result, list)
        assert len(result) == len(SAMPLE_CHUNKS)
        
        # Verify collection upsert was called once with all records
        mock_collection.upsert.assert_called_once()
        
        # Check the upsert call arguments
        call_args = mock_collection.upsert.call_args
        records = call_args[1]['records']
        assert len(records) == len(SAMPLE_CHUNKS)


def test_pgvector_handshake_search(mock_dependencies):
    """Test similarity search functionality."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Mock return data for search
        mock_collection.query.return_value = [
            ("id1", 0.5, {"text": "Similar text", "chunk_type": "Chunk"}),
            ("id2", 0.7, {"text": "Another similar text", "chunk_type": "Chunk"})
        ]
        
        results = handshake.search("test query", limit=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["similarity"] == 0.5
        assert results[0]["text"] == "Similar text"
        
        # Verify search query was executed
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args[1]['limit'] == 2


def test_pgvector_handshake_search_with_filters(mock_dependencies):
    """Test similarity search with metadata filters."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Mock return data for search
        mock_collection.query.return_value = [
            ("id1", 0.5, {"text": "Filtered text", "chunk_type": "SentenceChunk"})
        ]
        
        filters = {"chunk_type": {"$eq": "SentenceChunk"}}
        results = handshake.search("test query", limit=1, filters=filters)
        
        assert len(results) == 1
        assert results[0]["chunk_type"] == "SentenceChunk"
        
        # Verify filters were passed to query
        call_args = mock_collection.query.call_args
        assert call_args[1]['filters'] == filters


def test_pgvector_handshake_create_index(mock_dependencies):
    """Test creating vector index."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client.return_value = mock_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost")
        
        handshake.create_index(method="hnsw", m=32, ef_construction=128)
        
        # Verify index creation was called
        mock_collection.create_index.assert_called_once()
        call_args = mock_collection.create_index.call_args
        assert call_args[1]['method'] == "hnsw"
        assert call_args[1]['m'] == 32
        assert call_args[1]['ef_construction'] == 128


def test_pgvector_handshake_delete_collection(mock_dependencies):
    """Test deleting collection."""
    mock_client = mock_dependencies['client']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client.return_value = mock_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost", collection_name="test_collection")
        
        handshake.delete_collection()
        
        # Verify delete collection was called
        mock_client.delete_collection.assert_called_once_with("test_collection")


def test_pgvector_handshake_get_collection_info(mock_dependencies):
    """Test getting collection information."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client.return_value = mock_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost")
        
        info = handshake.get_collection_info()
        
        assert info["name"] == mock_collection.name
        assert info["dimension"] == mock_collection.dimension


def test_pgvector_handshake_repr(mock_dependencies):
    """Test the __repr__ method."""
    mock_client = mock_dependencies['client']
    
    # Create a mock vecs module
    mock_vecs = Mock()
    mock_vecs.create_client.return_value = mock_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(host="localhost", collection_name="test_collection")
        expected_repr = f"PgvectorHandshake(collection_name=test_collection, vector_dimensions={handshake.vector_dimensions})"
        assert repr(handshake) == expected_repr


def test_pgvector_handshake_import_dependencies_error():
    """Test import dependencies failure."""
    with patch.object(PgvectorHandshake, '_is_available', return_value=False):
        handshake = PgvectorHandshake.__new__(PgvectorHandshake)  # Create without calling __init__
        
        with pytest.raises(ImportError, match="vecs is not installed"):
            handshake._import_dependencies()


def test_pgvector_handshake_import_dependencies_success():
    """Test successful import of dependencies."""
    with patch('chonkie.friends.handshakes.pgvector.importutil.find_spec') as mock_find_spec:
        # Mock vecs as available
        mock_find_spec.side_effect = lambda name: Mock() if name == 'vecs' else None
        
        handshake = PgvectorHandshake.__new__(PgvectorHandshake)  # Create without calling __init__
        
        # Mock the imports to avoid actual import
        with patch('builtins.__import__') as mock_import:
            handshake._import_dependencies()
            
            # Verify the imports were attempted
            assert mock_import.call_count >= 1


def test_pgvector_handshake_search_without_metadata_and_similarity(mock_dependencies):
    """Test search with include_metadata=False and include_value=False."""
    mock_client = mock_dependencies['client']
    mock_collection = mock_dependencies['collection']
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch('chonkie.friends.handshakes.pgvector.vecs', Mock(create_client=Mock(return_value=mock_client))):
        
        handshake = PgvectorHandshake(host="localhost")
        
        # Mock return data for search (only ID when metadata and value are False)
        mock_collection.query.return_value = [("id1",), ("id2",)]
        
        results = handshake.search(
            "test query", 
            limit=2, 
            include_metadata=False, 
            include_value=False
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert "similarity" not in results[0]
        assert "text" not in results[0]


def test_pgvector_handshake_connection_priority():
    """Test that connection parameters are used in correct priority order."""
    mock_client = Mock()
    
    # Test 1: Client parameter takes priority
    with patch.object(PgvectorHandshake, '_import_dependencies'):
        handshake = PgvectorHandshake(
            client=mock_client,
            connection_string="postgresql://should_not_be_used",
            host="should_not_be_used"
        )
        assert handshake.client == mock_client
    
    # Test 2: Connection string takes priority over individual params
    mock_create_client = Mock(return_value=mock_client)
    mock_vecs = Mock()
    mock_vecs.create_client = mock_create_client
    
    with patch.object(PgvectorHandshake, '_import_dependencies'), \
         patch.dict('chonkie.friends.handshakes.pgvector.__dict__', {'vecs': mock_vecs}):
        
        handshake = PgvectorHandshake(
            connection_string="postgresql://user:pass@host:5432/db",
            host="should_not_be_used",
            user="should_not_be_used"
        )
        
        mock_create_client.assert_called_once_with("postgresql://user:pass@host:5432/db")