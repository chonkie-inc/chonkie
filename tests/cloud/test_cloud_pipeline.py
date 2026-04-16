"""Unit tests for ``chonkie.cloud.pipeline`` with HTTP mocked."""

import os
from unittest.mock import MagicMock, patch

import pytest

from chonkie.cloud.pipeline import Pipeline, PipelineStep
from chonkie.types import Chunk


def test_pipeline_step_to_dict_and_from_dict_roundtrip() -> None:
    """PipelineStep serialisation matches cloud API shape."""
    step = PipelineStep(type="chunk", component="recursive", params={"chunk_size": 128})
    data = step.to_dict()
    assert data["type"] == "chunk"
    assert data["component"] == "recursive"
    assert data["chunk_size"] == 128
    restored = PipelineStep.from_dict(data)
    assert restored.type == step.type
    assert restored.component == step.component
    assert restored.params == step.params


def test_pipeline_invalid_slug_raises() -> None:
    """Invalid slug format is rejected before any HTTP."""
    with pytest.raises(ValueError, match="Invalid slug"):
        Pipeline(slug="Bad Slug", api_key="test-key")


def test_pipeline_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor requires an API key or CHONKIE_API_KEY."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        Pipeline(slug="valid-slug", api_key=None)


def test_pipeline_describe_and_reset() -> None:
    """Describe, reset, and to_config reflect step state."""
    p = Pipeline(slug="a-pipeline", api_key="k")
    assert p.describe() == "Empty pipeline"
    p.chunk_with("token", chunk_size=64).refine_with("overlap", context_size=8)
    assert "chunk(token)" in p.describe()
    assert "refine(overlap)" in p.describe()
    assert len(p.to_config()) == 2
    p.reset()
    assert p.describe() == "Empty pipeline"
    assert p.to_config() == []


def test_pipeline_get_success() -> None:
    """Pipeline.get parses a successful JSON payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "slug": "my-pipe",
        "description": "test",
        "id": "id-1",
        "created_at": "2020-01-01T00:00:00",
        "updated_at": "2020-01-02T00:00:00",
        "steps": [
            {"type": "chunk", "component": "recursive", "chunk_size": 256},
        ],
    }
    with patch.dict(os.environ, {"CHONKIE_API_KEY": "secret"}, clear=False):
        with patch("chonkie.cloud.pipeline.httpx.get", return_value=mock_resp):
            p = Pipeline.get("my-pipe", api_key="secret")
    assert p.slug == "my-pipe"
    assert p.description == "test"
    assert p.is_saved is True
    assert len(p.steps) == 1
    assert p.steps[0].component == "recursive"


def test_pipeline_get_not_found() -> None:
    """404 from API becomes ValueError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("chonkie.cloud.pipeline.httpx.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="not found"):
            Pipeline.get("missing", api_key="k")


def test_pipeline_list_success() -> None:
    """Pipeline.list builds instances from API JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "pipelines": [
            {
                "slug": "p1",
                "description": None,
                "id": "1",
                "created_at": "t",
                "updated_at": "t",
                "steps": [],
            },
        ],
    }
    with patch("chonkie.cloud.pipeline.httpx.get", return_value=mock_resp):
        items = Pipeline.list(api_key="k")
    assert len(items) == 1
    assert items[0].slug == "p1"


def test_pipeline_validate_success() -> None:
    """Validate posts steps and returns API tuple."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"valid": True, "errors": None}
    with patch("chonkie.cloud.pipeline.httpx.post", return_value=mock_resp) as post:
        ok, errors = Pipeline.validate(
            [{"type": "chunk", "component": "token", "chunk_size": 10}],
            api_key="k",
        )
    post.assert_called_once()
    assert ok is True
    assert errors is None


def test_pipeline_run_with_text_mocks_save_and_execute() -> None:
    """run(text=...) saves if needed then POSTs execute; parses Chunk list."""
    save_resp = MagicMock(status_code=200)
    save_resp.json.return_value = {
        "id": "pid",
        "created_at": "c",
        "updated_at": "u",
    }
    exec_resp = MagicMock(status_code=200)
    exec_resp.json.return_value = {
        "chunks": [
            {
                "text": "hello",
                "start_index": 0,
                "end_index": 5,
                "token_count": 5,
            },
        ],
    }

    def post_side_effect(url: str, **kwargs: object) -> MagicMock:  # noqa: ARG001
        if url.endswith("/pipeline"):
            return save_resp
        return exec_resp

    with patch("chonkie.cloud.pipeline.httpx.post", side_effect=post_side_effect):
        p = Pipeline(slug="run-me", api_key="k")
        p.chunk_with("token", chunk_size=32, tokenizer="character")
        chunks = p.run(text="hello world")

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "hello"


def test_pipeline_run_requires_text_or_file() -> None:
    """run() validates inputs."""
    p = Pipeline(slug="x", api_key="k")
    p.chunk_with("token", chunk_size=8, tokenizer="character")
    with patch("chonkie.cloud.pipeline.httpx.post") as post:
        with pytest.raises(ValueError, match="Either 'text' or 'file'"):
            p.run()
    post.assert_not_called()


def test_pipeline_run_rejects_both_text_and_file() -> None:
    p = Pipeline(slug="x", api_key="k")
    p.chunk_with("token", chunk_size=8, tokenizer="character")
    with pytest.raises(ValueError, match="Cannot provide both"):
        p.run(text="a", file="dummy.txt")
