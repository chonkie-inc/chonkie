"""Test the MongoDBHandshake class."""

from unittest.mock import MagicMock, patch

import pytest

from chonkie.embeddings import BaseEmbeddings
from chonkie.friends.handshakes.mongodb import MongoDBHandshake
from chonkie.types import Chunk

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-retrieval-32M"

# Skip all tests in this module if pymongo is not installed
pytestmark = pytest.mark.skipif(
    not MongoDBHandshake()._is_available(), reason="pymongo not installed"
)


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models in CI."""
    with patch(
        "chonkie.embeddings.AutoEmbeddings.get_embeddings"
    ) as mock_get_embeddings:

        class MockEmbeddings(BaseEmbeddings):
            def __init__(self):
                self._dimension = 8
                self.model_name_or_path = "mock-model"

            @property
            def dimension(self):
                return self._dimension

            def embed(self, text):
                return [0.1] * self._dimension

            def embed_batch(self, texts):
                return [[0.1] * self._dimension for _ in texts]

            def get_tokenizer_or_token_counter(self):
                return lambda x: len(x.split())

            def _is_available(self):
                return True

        mock_embedding = MockEmbeddings()
        mock_get_embeddings.return_value = mock_embedding
        yield mock_get_embeddings


@pytest.fixture
def sample_chunk():
    """Fixture for a sample chunk."""
    return Chunk(
        text="This is a test chunk.", start_index=0, end_index=22, token_count=5
    )


@pytest.fixture
def sample_chunks():
    """Fixture for sample chunks."""
    return [
        Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
        Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
    ]


# ---- Initialization Tests ----


def test_mongodb_handshake_init_defaults():
    """Test MongoDBHandshake initialization with default parameters."""
    with patch("pymongo.MongoClient"):
        handshake = MongoDBHandshake()
        assert handshake.db_name != "random"
        assert handshake.collection_name != "random"
        assert isinstance(handshake.embedding_model, BaseEmbeddings)
        assert handshake.dimension == handshake.embedding_model.dimension
        assert handshake.collection is not None


def test_mongodb_handshake_init_specific_db_collection():
    """Test MongoDBHandshake initialization with specific db and collection name."""
    with patch("pymongo.MongoClient"):
        handshake = MongoDBHandshake(db_name="testdb", collection_name="testcol")
        assert handshake.db_name == "testdb"
        assert handshake.collection_name == "testcol"
        assert handshake.collection is not None


# ---- Write Tests ----


def test_mongodb_handshake_write_single_chunk(sample_chunk):
    """Test writing a single chunk."""
    with patch("pymongo.MongoClient") as mock_client:
        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        handshake = MongoDBHandshake(db_name="testdb", collection_name="testcol")
        handshake.collection = mock_collection
        handshake.write(sample_chunk)
        assert mock_collection.insert_many.called
        args, kwargs = mock_collection.insert_many.call_args
        assert len(args[0]) == 1


def test_mongodb_handshake_write_multiple_chunks(sample_chunks):
    """Test writing multiple chunks."""
    with patch("pymongo.MongoClient") as mock_client:
        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        handshake = MongoDBHandshake(db_name="testdb", collection_name="testcol")
        handshake.collection = mock_collection
        handshake.write(sample_chunks)
        assert mock_collection.insert_many.called
        args, kwargs = mock_collection.insert_many.call_args
        assert len(args[0]) == len(sample_chunks)


# ---- Helper Method Tests ----


def test_generate_id(sample_chunk):
    """Test the _generate_id method."""
    with patch("pymongo.MongoClient"):
        handshake = MongoDBHandshake(db_name="testdb", collection_name="testcol")
        generated_id = handshake._generate_id(0, sample_chunk)
        import uuid

        assert isinstance(generated_id, str)
        try:
            uuid.UUID(generated_id)
        except ValueError:
            pytest.fail(f"Generated ID '{generated_id}' is not a valid UUID.")
        assert handshake._generate_id(0, sample_chunk) == generated_id
        assert handshake._generate_id(1, sample_chunk) != generated_id
        diff_chunk = Chunk(
            text="Different text", start_index=0, end_index=14, token_count=2
        )
        assert handshake._generate_id(0, diff_chunk) != generated_id


def test_generate_document(sample_chunk):
    """Test the _generate_document method."""
    with patch("pymongo.MongoClient"):
        handshake = MongoDBHandshake(db_name="testdb", collection_name="testcol")
        embedding = [0.1] * handshake.dimension
        doc = handshake._generate_document(0, sample_chunk, embedding)
        assert doc["_id"] == handshake._generate_id(0, sample_chunk)
        assert doc["text"] == sample_chunk.text
        assert doc["start_index"] == sample_chunk.start_index
        assert doc["end_index"] == sample_chunk.end_index
        assert doc["token_count"] == sample_chunk.token_count
        assert doc["embedding"] == embedding
