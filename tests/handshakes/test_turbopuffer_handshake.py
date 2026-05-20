"""Tests for TurbopufferHandshake with a stub ``turbopuffer`` package."""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonkie.types import Chunk, Document


class _FakeNamespace:
    """Minimal stand-in for ``turbopuffer.Namespace``."""

    def __init__(self, ns_id: str = "ns-test") -> None:
        self.id = ns_id

    def write(self, **kwargs: Any) -> None:
        self.last_write = kwargs

    def query(self, **kwargs: Any) -> Any:
        class _Rows:
            rows = [
                {
                    "id": "row-1",
                    "$dist": 0.25,
                    "text": "hello",
                    "start_index": 0,
                    "end_index": 5,
                    "token_count": 2,
                    "chunk_metadata": json.dumps({"extra": 1}),
                },
            ]

        return _Rows()


class _FakeTurbopuffer:
    def __init__(self, api_key: str | None = None, region: str | None = None) -> None:
        self._api_key = api_key
        self._region = region

    def namespaces(self) -> list[Any]:
        return []

    def namespace(self, name: str) -> _FakeNamespace:
        return _FakeNamespace(name)


@pytest.fixture
def fake_turbopuffer_module() -> Any:
    """Install a minimal ``turbopuffer`` module for the duration of the test."""
    mod = types.ModuleType("turbopuffer")
    mod.Turbopuffer = _FakeTurbopuffer
    mod.Namespace = _FakeNamespace
    old = sys.modules.get("turbopuffer")
    sys.modules["turbopuffer"] = mod
    yield mod
    if old is not None:
        sys.modules["turbopuffer"] = old
    else:
        sys.modules.pop("turbopuffer", None)


@pytest.fixture
def mock_embeddings_tpuf():
    with patch("chonkie.embeddings.AutoEmbeddings.get_embeddings") as m:
        emb = MagicMock()
        emb.embed.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb.embed_batch.return_value = np.array(
            [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32
        )
        m.return_value = emb
        yield emb


def test_turbopuffer_handshake_write_and_search(
    fake_turbopuffer_module,
    mock_embeddings_tpuf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write and search against a stub namespace."""
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "test-key-for-handshake-unit")

    from chonkie.handshakes.turbopuffer import TurbopufferHandshake

    ns = _FakeNamespace("fixed-ns")
    hs = TurbopufferHandshake(namespace=ns, api_key="test-key-for-handshake-unit")
    chunks = [
        Chunk(text="a", start_index=0, end_index=1, token_count=1, metadata={"k": "v"}),
        Chunk(text="b", start_index=1, end_index=2, token_count=1),
    ]
    hs.write(chunks)
    assert hasattr(ns, "last_write")
    assert "upsert_columns" in ns.last_write

    rows = hs.search(query="find", limit=3)
    assert len(rows) == 1
    assert rows[0]["text"] == "hello"
    assert rows[0]["extra"] == 1
    mock_embeddings_tpuf.embed.assert_called_once_with("find")


def test_turbopuffer_search_requires_query_or_embedding(
    fake_turbopuffer_module,
    mock_embeddings_tpuf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "k")
    from chonkie.handshakes.turbopuffer import TurbopufferHandshake

    hs = TurbopufferHandshake(namespace=_FakeNamespace(), api_key="k")
    with pytest.raises(ValueError, match="Either"):
        hs.search()


def test_turbopuffer_invalid_namespace_type_raises(
    fake_turbopuffer_module,
    mock_embeddings_tpuf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "k")
    from chonkie.handshakes.turbopuffer import TurbopufferHandshake

    with pytest.raises(ValueError, match="valid Turbopuffer Namespace"):
        TurbopufferHandshake(namespace="not-a-namespace", api_key="k")  # type: ignore[arg-type]


def test_turbopuffer_repr(
    fake_turbopuffer_module, mock_embeddings_tpuf, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "k")
    from chonkie.handshakes.turbopuffer import TurbopufferHandshake

    ns = _FakeNamespace("my-id")
    hs = TurbopufferHandshake(namespace=ns, api_key="k")
    assert "my-id" in repr(hs)


def test_turbopuffer_write_documents_uses_document_embeddings(
    fake_turbopuffer_module,
    mock_embeddings_tpuf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document-aware writes preserve chunk groups for contextual embedders."""
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "k")
    from chonkie.handshakes.turbopuffer import TurbopufferHandshake

    mock_embeddings_tpuf.embed_documents.return_value = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    ns = _FakeNamespace("fixed-ns")
    hs = TurbopufferHandshake(namespace=ns, api_key="k")
    documents = [
        Document(
            content="doc one",
            chunks=[
                Chunk(text="a", start_index=0, end_index=1, token_count=1),
                Chunk(text="b", start_index=1, end_index=2, token_count=1),
            ],
        ),
        Document(
            content="doc two",
            chunks=[Chunk(text="c", start_index=0, end_index=1, token_count=1)],
        ),
    ]

    hs.write_documents(documents)

    mock_embeddings_tpuf.embed_documents.assert_called_once_with([["a", "b"], ["c"]])
    assert ns.last_write["upsert_columns"]["text"] == ["a", "b", "c"]
    np.testing.assert_allclose(
        ns.last_write["upsert_columns"]["vector"],
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
    )
