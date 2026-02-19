"""Refinery endpoints for the Chonkie OSS API.

All endpoints live under the ``/v1/refine`` prefix.  They accept a list of
chunk dicts (as returned by the chunking endpoints) and return an enriched list
of chunk dicts.
"""

from typing import Any, Dict, List

from chonkie import EmbeddingsRefinery, OverlapRefinery
from chonkie.embeddings import OpenAIEmbeddings
from chonkie.types import Chunk
from fastapi import APIRouter, HTTPException

from api.schemas import EmbeddingsRefineryRequest, OverlapRefineryRequest
from api.utils import Timer, get_logger

router = APIRouter(prefix="/refine", tags=["Refineries"])
log = get_logger("api.routes.refineries")

# OpenAI models that are supported by this endpoint
_ALLOWED_OPENAI_MODELS = ("text-embedding-3-small", "text-embedding-3-large")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dicts_to_chunks(chunk_dicts: List[Dict[str, Any]]) -> List[Chunk]:
    """Convert a list of chunk dicts to :class:`~chonkie.types.Chunk` objects.

    Args:
        chunk_dicts: Raw dicts from the request body.

    Returns:
        List of :class:`Chunk` instances.

    Raises:
        HTTPException: 400 if any dict is missing required fields.
    """
    try:
        return [Chunk(**d) for d in chunk_dicts]
    except TypeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chunk format.  Each chunk must contain "
                "'text', 'start_index', 'end_index', and 'token_count'.  "
                f"Error: {exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Embeddings refinery
# ---------------------------------------------------------------------------

@router.post(
    "/embeddings",
    response_model=None,
    summary="Add OpenAI embeddings to chunks",
)
async def embeddings_refine(
    request: EmbeddingsRefineryRequest,
) -> List[Dict[str, Any]]:
    """Compute and attach OpenAI embeddings to a list of chunks.

    Each chunk in the response will include an ``embedding`` field containing a
    list of floats.

    Supported models:
    - ``text-embedding-3-small``
    - ``text-embedding-3-large``

    An ``OPENAI_API_KEY`` environment variable must be set.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/refine/embeddings",
        chunk_count=len(request.chunks),
        embedding_model=request.embedding_model,
    )

    if not request.chunks:
        return []

    # Normalise model name (strip provider prefix if present)
    actual_model = (
        request.embedding_model
        .replace("openai/", "")
        .replace("OpenAI/", "")
    )

    if actual_model not in _ALLOWED_OPENAI_MODELS:
        log.error(
            "Unsupported embedding model",
            embedding_model=request.embedding_model,
            allowed=_ALLOWED_OPENAI_MODELS,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only OpenAI models are supported: "
                f"{', '.join(_ALLOWED_OPENAI_MODELS)}.  "
                f"Got: {actual_model!r}"
            ),
        )

    try:
        timer.start("embedding_init")
        embedding = OpenAIEmbeddings(model=actual_model)
        log.info(
            "Embedding model ready",
            endpoint="POST /v1/refine/embeddings",
            model=actual_model,
            duration_ms=round(timer.end("embedding_init"), 2),
        )

        timer.start("convert_to_chunks")
        chunk_objects = _dicts_to_chunks(request.chunks)
        log.info(
            "Converted dicts to Chunk objects",
            endpoint="POST /v1/refine/embeddings",
            duration_ms=round(timer.end("convert_to_chunks"), 2),
        )

        timer.start("refinery")
        refinery = EmbeddingsRefinery(embedding_model=embedding)
        refined = refinery.refine(chunk_objects)
        log.info(
            "Refinery processing completed",
            endpoint="POST /v1/refine/embeddings",
            duration_ms=round(timer.end("refinery"), 2),
        )

        result = [chunk.to_dict() for chunk in refined]
        log.info(
            "Refining completed",
            endpoint="POST /v1/refine/embeddings",
            refined_count=len(result),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except HTTPException:
        raise
    except ImportError as exc:
        msg = (
            "EmbeddingsRefinery requires the 'openai' extra.  "
            "Install it with: pip install 'chonkie[openai]'"
        )
        log.error(msg, error=str(exc))
        raise HTTPException(status_code=500, detail=msg) from exc
    except Exception as exc:
        log.error(
            "Embeddings refinery failed",
            endpoint="POST /v1/refine/embeddings",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500, detail=f"Embeddings refinery failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Overlap refinery
# ---------------------------------------------------------------------------

@router.post(
    "/overlap",
    response_model=None,
    summary="Add overlap context to chunks",
)
async def overlap_refine(
    request: OverlapRefineryRequest,
) -> List[Dict[str, Any]]:
    """Append or prepend overlapping context from neighbouring chunks.

    This is useful when downstream consumers (e.g. RAG pipelines) need
    continuity between adjacent chunks.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/refine/overlap",
        chunk_count=len(request.chunks),
        context_size=request.context_size,
        mode=request.mode,
        method=request.method,
    )

    if not request.chunks:
        return []

    try:
        timer.start("convert_to_chunks")
        chunk_objects = _dicts_to_chunks(request.chunks)
        log.info(
            "Converted dicts to Chunk objects",
            endpoint="POST /v1/refine/overlap",
            duration_ms=round(timer.end("convert_to_chunks"), 2),
        )

        timer.start("refinery")
        refinery = OverlapRefinery(
            tokenizer=request.tokenizer,
            context_size=request.context_size,
            mode=request.mode,
            method=request.method,
            merge=request.merge,
            inplace=False,  # never mutate the input objects
        )
        refined = refinery.refine(chunk_objects)
        log.info(
            "Refinery processing completed",
            endpoint="POST /v1/refine/overlap",
            duration_ms=round(timer.end("refinery"), 2),
        )

        result = [chunk.to_dict() for chunk in refined]
        log.info(
            "Refining completed",
            endpoint="POST /v1/refine/overlap",
            refined_count=len(result),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "Overlap refinery failed",
            endpoint="POST /v1/refine/overlap",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500, detail=f"Overlap refinery failed: {exc}"
        ) from exc
