"""Integration tests for the Chonkie OSS FastAPI application."""

import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from chonkie.types import Chunk


def test_health(api_client: TestClient) -> None:
    """GET /health returns status ok."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root(api_client: TestClient) -> None:
    """GET / returns API metadata."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Chonkie OSS API"
    assert "version" in data
    assert "/v1/chunk/token" in data["chunkers"]


def test_chunk_token_character_tokenizer(api_client: TestClient) -> None:
    """POST /v1/chunk/token chunks short text with character tokenizer."""
    payload = {
        "text": "hello world",
        "tokenizer": "character",
        "chunk_size": 5,
        "chunk_overlap": 0,
    }
    response = api_client.post("/v1/chunk/token", json=payload)
    assert response.status_code == 200
    chunks = response.json()
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all("text" in c and "start_index" in c for c in chunks)


def test_refine_overlap(api_client: TestClient) -> None:
    """POST /v1/refine/overlap enriches chunk dicts."""
    chunks = [
        {"text": "aa", "start_index": 0, "end_index": 2, "token_count": 2},
        {"text": "bb", "start_index": 3, "end_index": 5, "token_count": 2},
    ]
    payload = {
        "chunks": chunks,
        "tokenizer": "character",
        "context_size": 1,
        "mode": "token",
        "method": "suffix",
        "merge": True,
    }
    response = api_client.post("/v1/refine/overlap", json=payload)
    assert response.status_code == 200
    out = response.json()
    assert len(out) == 2


def test_refine_overlap_missing_body_field(api_client: TestClient) -> None:
    """Request missing required ``chunks`` is rejected by FastAPI (422)."""
    response = api_client.post("/v1/refine/overlap", json={"tokenizer": "character"})
    assert response.status_code == 422


def test_refine_embeddings_mocked(api_client: TestClient) -> None:
    """Embeddings refinery route succeeds with EmbeddingsRefinery mocked."""
    chunks_in = [
        {"text": "hi", "start_index": 0, "end_index": 2, "token_count": 2},
    ]

    mock_instance = MagicMock()
    mock_instance.refine.return_value = [
        Chunk(text="hi", start_index=0, end_index=2, token_count=2, embedding=[0.1, 0.2]),
    ]

    with patch("chonkie.api.routes.refineries.EmbeddingsRefinery") as mock_cls:
        mock_cls.return_value = mock_instance
        response = api_client.post(
            "/v1/refine/embeddings",
            json={"chunks": chunks_in, "embedding_model": "dummy"},
        )
        assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0].get("embedding") == [0.1, 0.2]


def test_pipelines_crud_and_execute(api_client: TestClient) -> None:
    """Create, list, get, update, execute, and delete a pipeline."""
    unique = uuid.uuid4().hex[:12]
    pipeline_name = f"test-pipeline-api-{unique}"
    create_body = {
        "name": pipeline_name,
        "description": "integration test",
        "steps": [
            {
                "type": "chunk",
                "chunker": "token",
                "config": {"tokenizer": "character", "chunk_size": 100, "chunk_overlap": 0},
            },
        ],
    }
    r_create = api_client.post("/v1/pipelines", json=create_body)
    assert r_create.status_code == 201, r_create.text
    created = r_create.json()
    pipeline_id = created["id"]
    assert created["name"] == pipeline_name

    r_dup = api_client.post("/v1/pipelines", json=create_body)
    assert r_dup.status_code == 400

    r_list = api_client.get("/v1/pipelines")
    assert r_list.status_code == 200
    names = {p["name"] for p in r_list.json()}
    assert pipeline_name in names

    r_get = api_client.get(f"/v1/pipelines/{pipeline_id}")
    assert r_get.status_code == 200
    assert r_get.json()["id"] == pipeline_id

    r_patch = api_client.put(
        f"/v1/pipelines/{pipeline_id}",
        json={"description": "updated"},
    )
    assert r_patch.status_code == 200
    assert r_patch.json()["description"] == "updated"

    r_exec = api_client.post(
        f"/v1/pipelines/{pipeline_id}/execute",
        json={"text": "one two three four five"},
    )
    assert r_exec.status_code == 200
    exec_chunks = r_exec.json()
    assert isinstance(exec_chunks, list)
    assert len(exec_chunks) >= 1

    r_del = api_client.delete(f"/v1/pipelines/{pipeline_id}")
    assert r_del.status_code == 204

    r_gone = api_client.get(f"/v1/pipelines/{pipeline_id}")
    assert r_gone.status_code == 404
