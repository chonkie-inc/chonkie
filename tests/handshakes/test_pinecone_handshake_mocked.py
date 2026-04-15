"""PineconeHandshake tests using a mocked Pinecone client (no API key required)."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest

from chonkie.embeddings import BaseEmbeddings
from chonkie.handshakes.pinecone import PineconeHandshake
from chonkie.types import Chunk

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pinecone") is None,
    reason="pinecone not installed",
)


class _TinyEmbeddings(BaseEmbeddings):
    """Fixed-dimension embeddings for tests."""

    def __init__(self) -> None:
        self._dimension = 4

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return np.array([0.25] * self._dimension, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.25] * self._dimension for _ in texts], dtype=np.float32)

    def get_tokenizer(self):  # noqa: ANN201
        return lambda x: len(x.split())

    @classmethod
    def _is_available(cls) -> bool:
        return True


def _mock_pinecone_client(index_mock: MagicMock) -> MagicMock:
    client = MagicMock()
    client.has_index.return_value = True
    client.Index.return_value = index_mock
    return client


def test_pinecone_write_single_chunk() -> None:
    index = MagicMock()
    index.upsert = MagicMock()
    client = _mock_pinecone_client(index)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-write",
        embedding_model=_TinyEmbeddings(),
    )
    chunk = Chunk(text="only one", start_index=0, end_index=8, token_count=2)
    hs.write(chunk)
    index.upsert.assert_called_once()
    vectors = index.upsert.call_args[0][0]
    assert len(vectors) == 1
    assert np.allclose(vectors[0][1], [0.25] * 4)


def test_pinecone_search_uses_query_embedding() -> None:
    index = MagicMock()
    index.query.return_value = {
        "matches": [
            {"id": "m1", "score": 0.9, "metadata": {"text": "hit", "start_index": 0}},
        ],
    }
    client = _mock_pinecone_client(index)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-search",
        embedding_model=_TinyEmbeddings(),
    )
    out = hs.search(query="hello", limit=2)
    assert len(out) == 1
    assert out[0]["text"] == "hit"
    index.query.assert_called_once()


def test_pinecone_search_prefers_query_when_both_provided() -> None:
    """If both query and embedding are passed, query wins and drives the embed call."""
    index = MagicMock()
    index.query.return_value = {"matches": []}
    client = _mock_pinecone_client(index)
    emb = _TinyEmbeddings()
    seen: list[str] = []

    def track_embed(text: str) -> np.ndarray:
        seen.append(text)
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    emb.embed = track_embed  # type: ignore[method-assign]
    hs = PineconeHandshake(
        client=client,
        index_name="idx-both",
        embedding_model=emb,
    )
    hs.search(query="use this", embedding=[0.9, 0.9, 0.9, 0.9])
    assert seen == ["use this"]


def test_pinecone_search_requires_vector_input() -> None:
    index = MagicMock()
    client = _mock_pinecone_client(index)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-vec",
        embedding_model=_TinyEmbeddings(),
    )
    with pytest.raises(ValueError, match="Query string or embedding"):
        hs.search()


def test_pinecone_search_rejects_bad_embedding_shape() -> None:
    index = MagicMock()
    client = _mock_pinecone_client(index)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-bad",
        embedding_model=_TinyEmbeddings(),
    )
    with pytest.raises(ValueError, match="list of floats"):
        hs.search(embedding="not-a-vector")  # type: ignore[arg-type]


def test_pinecone_get_vectors_numpy_embedding() -> None:
    index = MagicMock()
    client = _mock_pinecone_client(index)
    emb = _TinyEmbeddings()

    def embed_side_effect(text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    emb.embed = MagicMock(side_effect=embed_side_effect)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-np",
        embedding_model=emb,
    )
    chunk = Chunk(text="x", start_index=0, end_index=1, token_count=1)
    vectors = hs._get_vectors(chunk)
    assert np.allclose(vectors[0][1], [0.1, 0.2, 0.3, 0.4])


def test_pinecone_index_without_upsert_raises() -> None:
    class BadIndex:
        """Index object missing ``upsert`` (invalid for PineconeHandshake)."""

        pass

    client = MagicMock()
    client.has_index.return_value = True
    client.Index.return_value = BadIndex()
    with pytest.raises(TypeError, match="Failed to initialize"):
        PineconeHandshake(
            client=client,
            index_name="bad-index",
            embedding_model=_TinyEmbeddings(),
        )


def test_pinecone_search_unexpected_response_type() -> None:
    index = MagicMock()
    index.query.return_value = object()
    client = _mock_pinecone_client(index)
    hs = PineconeHandshake(
        client=client,
        index_name="idx-bad-resp",
        embedding_model=_TinyEmbeddings(),
    )
    with pytest.raises(ValueError, match="Unexpected response"):
        hs.search(embedding=[0.1, 0.2, 0.3, 0.4])
