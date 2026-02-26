"""Unit tests for handshake payload/metadata methods — no DB connections required."""

from __future__ import annotations

from chonkie.types import Chunk


def _make_chunk(
    text: str = "hello world",
    start_index: int = 0,
    end_index: int = 11,
    token_count: int = 2,
    context: str | None = None,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        text=text,
        start_index=start_index,
        end_index=end_index,
        token_count=token_count,
        context=context,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# QdrantHandshake._generate_payload
# ---------------------------------------------------------------------------


def _qdrant_payload(chunk: Chunk) -> dict:
    """Duplicate of QdrantHandshake._generate_payload for isolation."""
    from chonkie.handshakes.qdrant import QdrantHandshake

    instance = object.__new__(QdrantHandshake)
    instance.collection_name = "test"
    return instance._generate_payload(chunk)


def test_qdrant_payload_includes_chunk_id():
    chunk = _make_chunk()
    payload = _qdrant_payload(chunk)
    assert payload.get("chunk_id") == chunk.id


def test_qdrant_payload_includes_context_when_set():
    chunk = _make_chunk(context="some context")
    payload = _qdrant_payload(chunk)
    assert payload.get("context") == "some context"


def test_qdrant_payload_excludes_context_when_none():
    chunk = _make_chunk(context=None)
    payload = _qdrant_payload(chunk)
    assert "context" not in payload


def test_qdrant_payload_flattens_metadata():
    chunk = _make_chunk(metadata={"title": "My Doc", "source": "wiki"})
    payload = _qdrant_payload(chunk)
    assert payload.get("title") == "My Doc"
    assert payload.get("source") == "wiki"


def test_qdrant_payload_empty_metadata():
    chunk = _make_chunk(metadata={})
    payload = _qdrant_payload(chunk)
    assert "title" not in payload


# ---------------------------------------------------------------------------
# ChromaHandshake._generate_metadata
# ---------------------------------------------------------------------------


def _chroma_metadata(chunk: Chunk) -> dict:
    from chonkie.handshakes.chroma import ChromaHandshake

    instance = object.__new__(ChromaHandshake)
    instance.collection_name = "test"
    return instance._generate_metadata(chunk)


def test_chroma_metadata_includes_chunk_id():
    chunk = _make_chunk()
    meta = _chroma_metadata(chunk)
    assert meta.get("chunk_id") == chunk.id


def test_chroma_metadata_includes_context_when_set():
    chunk = _make_chunk(context="ctx")
    meta = _chroma_metadata(chunk)
    assert meta.get("context") == "ctx"


def test_chroma_metadata_excludes_context_when_none():
    chunk = _make_chunk()
    meta = _chroma_metadata(chunk)
    assert "context" not in meta


def test_chroma_metadata_flattens_metadata():
    chunk = _make_chunk(metadata={"title": "Doc"})
    meta = _chroma_metadata(chunk)
    assert meta.get("title") == "Doc"


# ---------------------------------------------------------------------------
# PineconeHandshake._generate_metadata
# ---------------------------------------------------------------------------


def _pinecone_metadata(chunk: Chunk) -> dict:
    from chonkie.handshakes.pinecone import PineconeHandshake

    instance = object.__new__(PineconeHandshake)
    instance.index_name = "test"
    return instance._generate_metadata(chunk)


def test_pinecone_metadata_includes_chunk_id():
    chunk = _make_chunk()
    meta = _pinecone_metadata(chunk)
    assert meta.get("chunk_id") == chunk.id


def test_pinecone_metadata_includes_context_when_set():
    chunk = _make_chunk(context="ctx")
    meta = _pinecone_metadata(chunk)
    assert meta.get("context") == "ctx"


def test_pinecone_metadata_excludes_context_when_none():
    chunk = _make_chunk()
    meta = _pinecone_metadata(chunk)
    assert "context" not in meta


def test_pinecone_metadata_flattens_metadata():
    chunk = _make_chunk(metadata={"title": "Doc", "page": 5})
    meta = _pinecone_metadata(chunk)
    assert meta.get("title") == "Doc"
    assert meta.get("page") == 5


# ---------------------------------------------------------------------------
# pgvectorHandshake._generate_metadata
# ---------------------------------------------------------------------------


def _pgvector_metadata(chunk: Chunk) -> dict:
    from chonkie.handshakes.pgvector import PgvectorHandshake

    instance = object.__new__(PgvectorHandshake)
    instance.collection_name = "test"
    return instance._generate_metadata(chunk)


def test_pgvector_metadata_includes_chunk_id():
    chunk = _make_chunk()
    meta = _pgvector_metadata(chunk)
    assert meta.get("chunk_id") == chunk.id


def test_pgvector_metadata_includes_context_when_set():
    chunk = _make_chunk(context="ctx")
    meta = _pgvector_metadata(chunk)
    assert meta.get("context") == "ctx"


def test_pgvector_metadata_excludes_context_when_none():
    chunk = _make_chunk()
    meta = _pgvector_metadata(chunk)
    assert "context" not in meta


def test_pgvector_metadata_flattens_metadata():
    chunk = _make_chunk(metadata={"title": "Doc"})
    meta = _pgvector_metadata(chunk)
    assert meta.get("title") == "Doc"


# ---------------------------------------------------------------------------
# MongoDBHandshake._generate_document
# (MongoDB uses _generate_document instead of _generate_metadata)
# ---------------------------------------------------------------------------


def _mongodb_document(chunk: Chunk) -> dict:
    from chonkie.handshakes.mongodb import MongoDBHandshake

    instance = object.__new__(MongoDBHandshake)
    instance.collection_name = "test"
    return instance._generate_document(0, chunk, [0.1, 0.2, 0.3])


def test_mongodb_document_includes_chunk_id():
    chunk = _make_chunk()
    doc = _mongodb_document(chunk)
    assert doc.get("chunk_id") == chunk.id


def test_mongodb_document_includes_context_when_set():
    chunk = _make_chunk(context="ctx")
    doc = _mongodb_document(chunk)
    assert doc.get("context") == "ctx"


def test_mongodb_document_excludes_context_when_none():
    chunk = _make_chunk()
    doc = _mongodb_document(chunk)
    assert "context" not in doc


def test_mongodb_document_flattens_metadata():
    chunk = _make_chunk(metadata={"title": "Doc"})
    doc = _mongodb_document(chunk)
    assert doc.get("title") == "Doc"
