"""Tests for cloud refinery classes (BaseRefinery, OverlapRefinery, EmbeddingsRefinery)."""

from unittest.mock import MagicMock, patch

import pytest

from chonkie import Chunk
from chonkie.cloud.refineries.base import BaseRefinery
from chonkie.cloud.refineries.embeddings import EmbeddingsRefinery
from chonkie.cloud.refineries.overlap import OverlapRefinery

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunks(n: int = 3) -> list[Chunk]:
    """Return a list of n simple Chunk objects."""
    text = "The hippo chunked the river."
    return [
        Chunk(text=text, start_index=i, end_index=i + len(text), token_count=5)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BaseRefinery tests
# ---------------------------------------------------------------------------


def test_base_refinery_is_abstract():
    """Instantiating BaseRefinery directly raises TypeError."""
    with pytest.raises(TypeError):
        BaseRefinery()  # type: ignore[abstract]


def test_base_refinery_subclass_must_implement_refine():
    """A concrete subclass without refine() raises TypeError on instantiation."""

    class Incomplete(BaseRefinery):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_base_refinery_call_delegates_to_refine():
    """__call__ on a concrete subclass invokes refine() with the same args."""

    class Concrete(BaseRefinery):
        def refine(self, chunks):
            return [c for c in chunks]  # identity

    obj = Concrete()
    chunks = make_chunks(2)
    result = obj(chunks)
    assert result == chunks


def test_base_refinery_constants():
    """BaseRefinery exposes BASE_URL and VERSION class constants."""
    assert BaseRefinery.BASE_URL == "https://api.chonkie.ai"
    assert BaseRefinery.VERSION == "v1"


# ---------------------------------------------------------------------------
# OverlapRefinery tests
# ---------------------------------------------------------------------------


def test_overlap_refinery_raises_without_api_key(monkeypatch):
    """OverlapRefinery raises ValueError when no API key is available."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        OverlapRefinery()


def test_overlap_refinery_loads_api_key_from_env(monkeypatch):
    """OverlapRefinery reads the API key from CHONKIE_API_KEY env var."""
    monkeypatch.setenv("CHONKIE_API_KEY", "env-key")
    refinery = OverlapRefinery()
    assert refinery.api_key == "env-key"


def test_overlap_refinery_explicit_api_key_takes_priority(monkeypatch):
    """Explicit api_key arg takes priority over the env var."""
    monkeypatch.setenv("CHONKIE_API_KEY", "env-key")
    refinery = OverlapRefinery(api_key="explicit-key")
    assert refinery.api_key == "explicit-key"


def test_overlap_refinery_stores_all_parameters(monkeypatch):
    """All constructor parameters are stored as instance attributes."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = OverlapRefinery(
        tokenizer="gpt2",
        context_size=0.5,
        mode="token",
        method="prefix",
        recipe="default",
        lang="fr",
        merge=False,
    )
    assert refinery.tokenizer == "gpt2"
    assert refinery.context_size == 0.5
    assert refinery.mode == "token"
    assert refinery.method == "prefix"
    assert refinery.recipe == "default"
    assert refinery.lang == "fr"
    assert refinery.merge is False


def test_overlap_refinery_raises_on_mixed_chunk_types(monkeypatch):
    """OverlapRefinery.refine raises ValueError for heterogeneous chunk types."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = OverlapRefinery()

    class OtherChunk(Chunk):
        pass

    chunks = [
        Chunk(text="a", start_index=0, end_index=1, token_count=1),
        OtherChunk(text="b", start_index=1, end_index=2, token_count=1),
    ]
    with pytest.raises(ValueError, match="same type"):
        result = refinery.refine(chunks)  # noqa: F841


def test_overlap_refinery_refine_posts_correct_payload(monkeypatch):
    """refine() sends a POST with the expected JSON payload."""
    monkeypatch.setenv("CHONKIE_API_KEY", "test-key")
    refinery = OverlapRefinery(tokenizer="gpt2", context_size=0.25, mode="token", method="suffix")
    chunks = make_chunks(2)

    response_data = [c.to_dict() for c in chunks]
    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch("chonkie.cloud.refineries.overlap.httpx.post", return_value=mock_response) as mock_post:
        result = refinery.refine(chunks)  # noqa: F841

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    # Verify URL
    assert "/refine/overlap" in call_kwargs[0][0]
    # Verify Authorization header
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"
    # Verify payload includes chunks
    payload = call_kwargs[1]["json"]
    assert len(payload["chunks"]) == 2
    assert payload["tokenizer_or_token_counter"] == "gpt2"
    assert payload["context_size"] == 0.25
    assert payload["mode"] == "token"
    assert payload["method"] == "suffix"


def test_overlap_refinery_refine_returns_deserialized_chunks(monkeypatch):
    """refine() deserializes the API response back into Chunk objects."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = OverlapRefinery()
    chunks = make_chunks(2)
    response_data = [c.to_dict() for c in chunks]

    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch("chonkie.cloud.refineries.overlap.httpx.post", return_value=mock_response):
        result = refinery.refine(chunks)  # noqa: F841

    assert len(result) == 2
    assert all(isinstance(c, Chunk) for c in result)


def test_overlap_refinery_call_invokes_refine(monkeypatch):
    """__call__ invokes refine() with the same arguments."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = OverlapRefinery()
    chunks = make_chunks(1)
    response_data = [c.to_dict() for c in chunks]

    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch("chonkie.cloud.refineries.overlap.httpx.post", return_value=mock_response):
        result = refinery(chunks)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# EmbeddingsRefinery tests
# ---------------------------------------------------------------------------


def test_embeddings_refinery_raises_without_api_key(monkeypatch):
    """EmbeddingsRefinery raises ValueError when no API key is available."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        EmbeddingsRefinery()


def test_embeddings_refinery_loads_api_key_from_env(monkeypatch):
    """EmbeddingsRefinery reads the API key from CHONKIE_API_KEY env var."""
    monkeypatch.setenv("CHONKIE_API_KEY", "env-key")
    refinery = EmbeddingsRefinery()
    assert refinery.api_key == "env-key"


def test_embeddings_refinery_explicit_api_key(monkeypatch):
    """Explicit api_key arg is stored on the instance."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    refinery = EmbeddingsRefinery(api_key="my-key")
    assert refinery.api_key == "my-key"


def test_embeddings_refinery_stores_embedding_model(monkeypatch):
    """EmbeddingsRefinery stores the embedding_model parameter."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = EmbeddingsRefinery(embedding_model="custom/model")
    assert refinery.embedding_model == "custom/model"


def test_embeddings_refinery_raises_on_mixed_chunk_types(monkeypatch):
    """EmbeddingsRefinery.refine raises ValueError for heterogeneous chunk types."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = EmbeddingsRefinery()

    class OtherChunk(Chunk):
        pass

    chunks = [
        Chunk(text="a", start_index=0, end_index=1, token_count=1),
        OtherChunk(text="b", start_index=1, end_index=2, token_count=1),
    ]
    with pytest.raises(ValueError, match="same type"):
        result = refinery.refine(chunks)  # noqa: F841


def test_embeddings_refinery_refine_posts_correct_payload(monkeypatch):
    """refine() sends a POST with chunks and embedding_model in the payload."""
    monkeypatch.setenv("CHONKIE_API_KEY", "test-key")
    refinery = EmbeddingsRefinery(embedding_model="minishlab/potion-retrieval-32M")
    chunks = make_chunks(2)

    # Build fake response: chunk dicts with embeddings attached
    response_data = [c.to_dict() for c in chunks]
    for item in response_data:
        item["embedding"] = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch(
        "chonkie.cloud.refineries.embeddings.httpx.post", return_value=mock_response
    ) as mock_post:
        result = refinery.refine(chunks)  # noqa: F841

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/refine/embeddings" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"
    payload = call_kwargs[1]["json"]
    assert payload["embedding_model"] == "minishlab/potion-retrieval-32M"
    assert len(payload["chunks"]) == 2


def test_embeddings_refinery_attaches_numpy_embeddings(monkeypatch):
    """refine() converts response embeddings to numpy arrays on returned chunks."""
    import numpy as np

    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = EmbeddingsRefinery()
    chunks = make_chunks(2)

    response_data = [c.to_dict() for c in chunks]
    for item in response_data:
        item["embedding"] = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch("chonkie.cloud.refineries.embeddings.httpx.post", return_value=mock_response):
        result = refinery.refine(chunks)  # noqa: F841

    for chunk in result:
        assert chunk.embedding is not None
        assert isinstance(chunk.embedding, np.ndarray)
        assert list(chunk.embedding) == pytest.approx([0.1, 0.2, 0.3])


def test_embeddings_refinery_call_invokes_refine(monkeypatch):
    """__call__ invokes refine() with the same arguments."""
    monkeypatch.setenv("CHONKIE_API_KEY", "k")
    refinery = EmbeddingsRefinery()
    chunks = make_chunks(1)

    response_data = [c.to_dict() for c in chunks]
    response_data[0]["embedding"] = [0.5]

    mock_response = MagicMock()
    mock_response.json.return_value = response_data

    with patch("chonkie.cloud.refineries.embeddings.httpx.post", return_value=mock_response):
        result = refinery(chunks)

    assert len(result) == 1
