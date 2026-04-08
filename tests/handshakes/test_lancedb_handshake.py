"""Test the LanceDBHandshake class."""

import uuid
from unittest.mock import Mock, patch

import lancedb
import pytest

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.handshakes.lancedb import LanceDBHandshake
from chonkie.types import Chunk

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-retrieval-32M"


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models in CI."""
    with patch("chonkie.embeddings.AutoEmbeddings.get_embeddings") as mock_get_embeddings:
        from chonkie.embeddings import BaseEmbeddings

        class MockEmbeddings(BaseEmbeddings):
            def __init__(self):
                super().__init__()
                self._dimension = 512
                self.model_name_or_path = DEFAULT_EMBEDDING_MODEL

            @property
            def dimension(self):
                return self._dimension

            def embed(self, text):
                import numpy as np

                return np.array([0.1] * 512, dtype=np.float32)

            def embed_batch(self, texts):
                import numpy as np

                return [np.array([0.1] * 512, dtype=np.float32) for _ in texts]

            def get_tokenizer(self):
                return Mock()

            @classmethod
            def _is_available(cls):
                return True

        mock_get_embeddings.return_value = MockEmbeddings()
        yield mock_get_embeddings


@pytest.fixture(scope="module")
def real_embeddings() -> BaseEmbeddings:
    """Load the actual default embedding model (skipped if unavailable)."""
    import os

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        return AutoEmbeddings.get_embeddings(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load embedding model: {e}")


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(text="This is a test chunk.", start_index=0, end_index=22, token_count=5)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(text="First test chunk.", start_index=0, end_index=18, token_count=4),
        Chunk(text="Second test chunk.", start_index=19, end_index=38, token_count=4),
    ]


# ---- Initialization Tests ----


def test_lancedb_handshake_init_defaults(real_embeddings: BaseEmbeddings) -> None:
    """Test initialization with default parameters (in-memory)."""
    handshake = LanceDBHandshake(embedding_model=DEFAULT_EMBEDDING_MODEL)
    assert handshake.table_name != "random"
    assert isinstance(handshake.embedding_model, BaseEmbeddings)
    assert handshake.dimension == real_embeddings.dimension
    assert handshake.table_name in handshake.connection.table_names()


def test_lancedb_handshake_init_specific_table(real_embeddings: BaseEmbeddings) -> None:
    """Test initialization with a specific table name."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-table-specific",
        embedding_model=real_embeddings,
    )
    assert handshake.table_name == "test-table-specific"
    assert "test-table-specific" in conn.table_names()


def test_lancedb_handshake_init_random_table(real_embeddings: BaseEmbeddings) -> None:
    """Test initialization with table_name='random'."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="random",
        embedding_model=real_embeddings,
    )
    assert isinstance(handshake.table_name, str)
    assert len(handshake.table_name) > 0
    assert handshake.table_name != "random"
    assert handshake.table_name in conn.table_names()


def test_lancedb_handshake_init_existing_table(real_embeddings: BaseEmbeddings) -> None:
    """Test initialization when the table already exists."""
    import pyarrow as pa

    conn = lancedb.connect("memory://")
    schema = pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("text", pa.utf8()),
        pa.field("start_index", pa.int32()),
        pa.field("end_index", pa.int32()),
        pa.field("token_count", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), real_embeddings.dimension)),
    ])
    conn.create_table("existing-table", schema=schema)

    handshake = LanceDBHandshake(
        connection=conn,
        table_name="existing-table",
        embedding_model=real_embeddings,
    )
    assert handshake.table_name == "existing-table"


# ---- Write Tests ----


def test_lancedb_handshake_write_single_chunk(
    sample_chunk: Chunk, real_embeddings: BaseEmbeddings
) -> None:
    """Test writing a single chunk."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-write-single",
        embedding_model=real_embeddings,
    )
    handshake.write(sample_chunk)

    rows = handshake.table.to_pandas()
    assert len(rows) == 1
    assert rows.iloc[0]["text"] == sample_chunk.text
    assert rows.iloc[0]["start_index"] == sample_chunk.start_index
    assert rows.iloc[0]["end_index"] == sample_chunk.end_index
    assert rows.iloc[0]["token_count"] == sample_chunk.token_count


def test_lancedb_handshake_write_multiple_chunks(
    sample_chunks: list[Chunk], real_embeddings: BaseEmbeddings
) -> None:
    """Test writing multiple chunks."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-write-multiple",
        embedding_model=real_embeddings,
    )
    handshake.write(sample_chunks)

    rows = handshake.table.to_pandas()
    assert len(rows) == len(sample_chunks)
    texts = set(rows["text"].tolist())
    assert texts == {chunk.text for chunk in sample_chunks}


def test_lancedb_handshake_write_upsert(
    sample_chunk: Chunk, real_embeddings: BaseEmbeddings
) -> None:
    """Test that writing the same chunk again performs an upsert (no duplicates)."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-write-upsert",
        embedding_model=real_embeddings,
    )
    handshake.write(sample_chunk)
    assert len(handshake.table.to_pandas()) == 1

    # Write again — same text → same ID → upsert
    modified = Chunk(text=sample_chunk.text, start_index=1, end_index=23, token_count=6)
    handshake.write(modified)

    rows = handshake.table.to_pandas()
    assert len(rows) == 1
    assert rows.iloc[0]["start_index"] == modified.start_index
    assert rows.iloc[0]["end_index"] == modified.end_index
    assert rows.iloc[0]["token_count"] == modified.token_count


# ---- Helper Method Tests ----


def test_generate_id(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test _generate_id returns a valid, deterministic UUID."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-id-gen",
        embedding_model=real_embeddings,
    )
    generated_id = handshake._generate_id(0, sample_chunk)
    assert isinstance(generated_id, str)
    uuid.UUID(generated_id)  # raises ValueError if not valid UUID

    assert handshake._generate_id(0, sample_chunk) == generated_id
    assert handshake._generate_id(1, sample_chunk) != generated_id

    diff_chunk = Chunk(text="Different text", start_index=0, end_index=14, token_count=2)
    assert handshake._generate_id(0, diff_chunk) != generated_id


def test_generate_row(sample_chunk: Chunk, real_embeddings: BaseEmbeddings) -> None:
    """Test _generate_row returns the expected dict structure."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-row-gen",
        embedding_model=real_embeddings,
    )
    embedding = [0.1] * real_embeddings.dimension
    row = handshake._generate_row(0, sample_chunk, embedding)

    assert row["id"] == handshake._generate_id(0, sample_chunk)
    assert row["text"] == sample_chunk.text
    assert row["start_index"] == sample_chunk.start_index
    assert row["end_index"] == sample_chunk.end_index
    assert row["token_count"] == sample_chunk.token_count
    assert row["vector"] == embedding


# ---- Search Tests ----


def test_lancedb_handshake_search(
    sample_chunks: list[Chunk], real_embeddings: BaseEmbeddings
) -> None:
    """Test semantic search returns results with expected keys."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="test-search",
        embedding_model=real_embeddings,
    )
    handshake.write(sample_chunks)

    results = handshake.search(query="test chunk", limit=2)
    assert len(results) <= 2
    for r in results:
        assert "id" in r
        assert "score" in r
        assert "text" in r
        assert "start_index" in r
        assert "end_index" in r
        assert "token_count" in r


def test_lancedb_handshake_search_no_query_raises() -> None:
    """Test that search raises ValueError when neither query nor embedding is provided."""
    with pytest.raises(ValueError, match="Either query or embedding must be provided"):
        conn = lancedb.connect("memory://")
        h = LanceDBHandshake(
            connection=conn,
            table_name="test-search-err",
            embedding_model="minishlab/potion-retrieval-32M",
        )
        h.search()


# ---- Repr Test ----


def test_repr(real_embeddings: BaseEmbeddings) -> None:
    """Test __repr__ returns the expected string."""
    conn = lancedb.connect("memory://")
    handshake = LanceDBHandshake(
        connection=conn,
        table_name="my-table",
        embedding_model=real_embeddings,
    )
    assert repr(handshake) == "LanceDBHandshake(table_name=my-table)"
