"""Tests for Document type."""

from chonkie import Chunk
from chonkie.types import Document


def test_document_default_initialization():
    """Document created with no args has empty content, chunks, and metadata."""
    doc = Document()
    assert doc.content == ""
    assert doc.chunks == []
    assert doc.metadata == {}


def test_document_id_has_doc_prefix():
    """Auto-generated document id starts with 'doc_'."""
    doc = Document()
    assert doc.id.startswith("doc_")


def test_document_id_is_unique():
    """Two Document instances get different auto-generated ids."""
    doc1 = Document()
    doc2 = Document()
    assert doc1.id != doc2.id


def test_document_content_stored():
    """Document stores the provided content string."""
    doc = Document(content="Hello, world!")
    assert doc.content == "Hello, world!"


def test_document_chunks_stored():
    """Document stores and retrieves a list of Chunk objects."""
    chunks = [
        Chunk(text="first", start_index=0, end_index=5, token_count=1),
        Chunk(text="second", start_index=6, end_index=12, token_count=1),
    ]
    doc = Document(content="first second", chunks=chunks)
    assert len(doc.chunks) == 2
    assert doc.chunks[0].text == "first"
    assert doc.chunks[1].text == "second"


def test_document_chunks_are_mutable():
    """Chunks list on a Document can be appended to after creation."""
    doc = Document(content="hello world")
    chunk = Chunk(text="hello", start_index=0, end_index=5, token_count=1)
    doc.chunks.append(chunk)
    assert len(doc.chunks) == 1


def test_document_metadata_stored():
    """Document stores arbitrary metadata key-value pairs."""
    doc = Document(content="text", metadata={"source": "web", "page": 3})
    assert doc.metadata["source"] == "web"
    assert doc.metadata["page"] == 3


def test_document_metadata_is_mutable():
    """Metadata dict on a Document can be updated after creation."""
    doc = Document()
    doc.metadata["key"] = "value"
    assert doc.metadata["key"] == "value"


def test_document_explicit_id():
    """Document accepts an explicit id."""
    doc = Document(id="my-custom-id")
    assert doc.id == "my-custom-id"


def test_document_default_chunks_are_independent():
    """Two Documents created with defaults have independent chunk lists."""
    doc1 = Document()
    doc2 = Document()
    doc1.chunks.append(Chunk(text="x", start_index=0, end_index=1, token_count=1))
    assert doc2.chunks == []


def test_document_default_metadata_are_independent():
    """Two Documents created with defaults have independent metadata dicts."""
    doc1 = Document()
    doc2 = Document()
    doc1.metadata["k"] = "v"
    assert "k" not in doc2.metadata
