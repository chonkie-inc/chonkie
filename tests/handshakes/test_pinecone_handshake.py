"""Test the PineconeHandshake class."""
import os
from unittest.mock import MagicMock, patch

import pytest

from chonkie.embeddings import BaseEmbeddings
from chonkie.friends.handshakes.pinecone import PineconeHandshake
from chonkie.types import Chunk

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-retrieval-32M"

# Skip all tests in this module if PINECONE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    os.getenv("PINECONE_API_KEY") is None,
    reason="PINECONE_API_KEY not set"
)


@pytest.fixture(autouse=True)
def mock_embeddings():
	"""Mock AutoEmbeddings to avoid downloading models in CI."""
	with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get_embeddings:
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
		text="This is a test chunk.",
		start_index=0,
		end_index=22,
		token_count=5
	)

@pytest.fixture
def sample_chunks():
    """Fixture for sample chunks."""
    return [
		Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
		Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
	]

# ---- Initialization Tests ----

def test_pinecone_handshake_init_defaults():
	"""Test PineconeHandshake initialization with default parameters."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = MagicMock()
		handshake = PineconeHandshake(api_key="fake-key")
		assert handshake.index_name != "random"
		assert isinstance(handshake.embedding_model, BaseEmbeddings)
		assert handshake.dimension == handshake.embedding_model.dimension
		assert handshake.index is not None

def test_pinecone_handshake_init_specific_index():
	"""Test PineconeHandshake initialization with a specific index name."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = MagicMock()
		handshake = PineconeHandshake(api_key="fake-key", index_name="test-index")
		assert handshake.index_name == "test-index"
		assert handshake.index is not None

def test_pinecone_handshake_init_existing_index():
	"""Test PineconeHandshake initialization with an existing index."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_client.return_value.has_index.return_value = True
		mock_client.return_value.Index.return_value = MagicMock()
		handshake = PineconeHandshake(api_key="fake-key", index_name="existing-index")
		assert handshake.index_name == "existing-index"
		assert handshake.index is not None

# ---- Write Tests ----

def test_pinecone_handshake_write_single_chunk(sample_chunk):
	"""Test writing a single chunk."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_index = MagicMock()
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = mock_index
		handshake = PineconeHandshake(api_key="fake-key", index_name="test-write-single")
		handshake.write(sample_chunk)
		assert mock_index.upsert.called
		args, kwargs = mock_index.upsert.call_args
		assert len(args[0]) == 1

def test_pinecone_handshake_write_multiple_chunks(sample_chunks):
	"""Test writing multiple chunks."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_index = MagicMock()
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = mock_index
		handshake = PineconeHandshake(api_key="fake-key", index_name="test-write-multiple")
		handshake.write(sample_chunks)
		assert mock_index.upsert.called
		args, kwargs = mock_index.upsert.call_args
		assert len(args[0]) == len(sample_chunks)

# ---- Helper Method Tests ----

def test_generate_id(sample_chunk):
	"""Test the _generate_id method."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = MagicMock()
		handshake = PineconeHandshake(api_key="fake-key", index_name="test-id-gen")
		generated_id = handshake._generate_id(0, sample_chunk)
		import uuid
		assert isinstance(generated_id, str)
		try:
			uuid.UUID(generated_id)
		except ValueError:
			pytest.fail(f"Generated ID '{generated_id}' is not a valid UUID.")
		assert handshake._generate_id(0, sample_chunk) == generated_id
		assert handshake._generate_id(1, sample_chunk) != generated_id
		diff_chunk = Chunk(text="Different text", start_index=0, end_index=14, token_count=2)
		assert handshake._generate_id(0, diff_chunk) != generated_id

def test_generate_metadata(sample_chunk):
	"""Test the _generate_metadata method."""
	with patch("pinecone.Pinecone") as mock_client:
		mock_client.return_value.has_index.return_value = False
		mock_client.return_value.create_index.return_value = None
		mock_client.return_value.Index.return_value = MagicMock()
		handshake = PineconeHandshake(api_key="fake-key", index_name="test-metadata-gen")
		metadata = handshake._generate_metadata(sample_chunk)
		assert metadata == {
			"text": sample_chunk.text,
			"start_index": sample_chunk.start_index,
			"end_index": sample_chunk.end_index,
			"token_count": sample_chunk.token_count,
		}
