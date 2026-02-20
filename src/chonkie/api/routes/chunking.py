"""Chunking endpoints for the Chonkie OSS API.

All endpoints live under the ``/v1/chunk`` prefix and return a list of chunk
dicts (or a list-of-lists when the caller submits a list of texts).
"""

from typing import Any, Dict, List, Union

from fastapi import APIRouter, HTTPException

from chonkie import (
    CodeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)
from chonkie.api.schemas import (
    ChunkingResponse,
    CodeChunkerRequest,
    RecursiveChunkerRequest,
    SemanticChunkerRequest,
    SentenceChunkerRequest,
    TokenChunkerRequest,
)
from chonkie.api.utils import Timer, fix_escaped_text, get_logger, sanitize_text_encoding

router = APIRouter(prefix="/chunk", tags=["Chunking"])
log = get_logger("api.routes.chunking")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_to_response(chunks: Any) -> ChunkingResponse:
    """Convert chunker output to a JSON-serialisable response.

    Handles both a flat list (single text input) and a list-of-lists
    (batch / multiple-text input).

    Args:
        chunks: Output from ``chunker(text)``.

    Returns:
        List of dicts or list-of-lists of dicts.

    """
    if not chunks:
        return []
    if isinstance(chunks[0], list):
        return [[c.to_dict() for c in chunk_list] for chunk_list in chunks]
    return [c.to_dict() for c in chunks]


def _chunk_count(chunks: Any) -> int:
    """Return total number of chunks across a flat or nested list."""
    if not chunks:
        return 0
    if isinstance(chunks[0], list):
        return sum(len(cl) for cl in chunks)
    return len(chunks)


# ---------------------------------------------------------------------------
# Token chunker
# ---------------------------------------------------------------------------


@router.post("/token", response_model=None, summary="Chunk text by token count")
async def token_chunk(request: TokenChunkerRequest) -> ChunkingResponse:
    """Chunk text using a fixed token-window strategy.

    Splits ``text`` into non-overlapping (or overlapping) windows of
    ``chunk_size`` tokens, measured by ``tokenizer``.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/chunk/token",
        has_list=isinstance(request.text, list),
        tokenizer=request.tokenizer,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    text = fix_escaped_text(request.text)

    try:
        timer.start("init")
        chunker = TokenChunker(
            tokenizer=request.tokenizer,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        log.info(
            "Chunker ready",
            endpoint="POST /v1/chunk/token",
            chunker="TokenChunker",
            duration_ms=round(timer.end("init"), 2),
        )

        timer.start("chunk")
        chunks = chunker(text)
        result = _chunks_to_response(chunks)
        chunk_count = _chunk_count(chunks)
        log.info(
            "Chunking completed",
            endpoint="POST /v1/chunk/token",
            chunk_count=chunk_count,
            duration_ms=round(timer.end("chunk"), 2),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except ValueError as exc:
        log.error("Invalid request parameters", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "Token chunking failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Token chunking failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------


@router.post("/sentence", response_model=None, summary="Chunk text at sentence boundaries")
async def sentence_chunk(request: SentenceChunkerRequest) -> ChunkingResponse:
    """Chunk text at sentence boundaries while respecting a token-size limit.

    Sentences are detected using the ``delim`` separators and grouped into
    chunks that do not exceed ``chunk_size`` tokens.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/chunk/sentence",
        has_list=isinstance(request.text, list),
        chunk_size=request.chunk_size,
    )

    text = fix_escaped_text(request.text)

    try:
        timer.start("init")
        chunker = SentenceChunker(
            tokenizer=request.tokenizer,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            min_sentences_per_chunk=request.min_sentences_per_chunk,
            min_characters_per_sentence=request.min_characters_per_sentence,
            approximate=request.approximate,
            delim=request.delim,
            include_delim=request.include_delim,
        )
        log.info(
            "Chunker ready",
            endpoint="POST /v1/chunk/sentence",
            chunker="SentenceChunker",
            duration_ms=round(timer.end("init"), 2),
        )

        timer.start("chunk")
        chunks = chunker(text)
        result = _chunks_to_response(chunks)
        chunk_count = _chunk_count(chunks)
        log.info(
            "Chunking completed",
            endpoint="POST /v1/chunk/sentence",
            chunk_count=chunk_count,
            duration_ms=round(timer.end("chunk"), 2),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except ValueError as exc:
        log.error("Invalid request parameters", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "Sentence chunking failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Sentence chunking failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------

# In-process cache so that the same (recipe, lang, tokenizer) combination
# reuses the already-initialised chunker across requests.
_recursive_cache: Dict[str, RecursiveChunker] = {}


def _get_recursive_chunker(recipe: str, lang: str, tokenizer: str) -> RecursiveChunker:
    """Return a cached :class:`RecursiveChunker` for the given configuration.

    Args:
        recipe: Named recipe (e.g. ``"default"``, ``"markdown"``).
        lang: Language hint.
        tokenizer: Tokenizer name.

    Returns:
        A :class:`RecursiveChunker` instance (potentially from cache).

    """
    key = f"{recipe}:{lang}:{tokenizer}"
    if key not in _recursive_cache:
        _recursive_cache[key] = RecursiveChunker.from_recipe(
            name=recipe, lang=lang, tokenizer=tokenizer
        )
    return _recursive_cache[key]


@router.post(
    "/recursive", response_model=None, summary="Recursively chunk text using structural separators"
)
async def recursive_chunk(request: RecursiveChunkerRequest) -> ChunkingResponse:
    """Chunk text recursively using a hierarchy of separators defined by ``recipe``.

    Common recipes include ``"default"`` (paragraph → sentence → word) and
    ``"markdown"`` (heading → paragraph → sentence → word).
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/chunk/recursive",
        has_list=isinstance(request.text, list),
        chunk_size=request.chunk_size,
        recipe=request.recipe,
        lang=request.lang,
    )

    # Sanitise then fix escapes
    if isinstance(request.text, list):
        text: Union[str, List[str]] = [sanitize_text_encoding(t) for t in request.text]
    else:
        text = sanitize_text_encoding(request.text)
    text = fix_escaped_text(text)

    try:
        timer.start("init")
        chunker = _get_recursive_chunker(
            recipe=request.recipe,
            lang=request.lang,
            tokenizer=request.tokenizer,
        )
        # Update mutable parameters per-request
        chunker.chunk_size = request.chunk_size
        chunker.min_characters_per_chunk = request.min_characters_per_chunk
        log.info(
            "Chunker ready",
            endpoint="POST /v1/chunk/recursive",
            chunker="RecursiveChunker",
            duration_ms=round(timer.end("init"), 2),
        )

        timer.start("chunk")
        chunks = chunker(text)
        result = _chunks_to_response(chunks)
        chunk_count = _chunk_count(chunks)
        log.info(
            "Chunking completed",
            endpoint="POST /v1/chunk/recursive",
            chunk_count=chunk_count,
            duration_ms=round(timer.end("chunk"), 2),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except ValueError as exc:
        log.error("Invalid request parameters", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "Recursive chunking failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Recursive chunking failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------


@router.post("/semantic", response_model=None, summary="Chunk text by semantic similarity")
async def semantic_chunk(request: SemanticChunkerRequest) -> ChunkingResponse:
    """Chunk text using sentence-embedding cosine similarity.

    Adjacent sentences whose similarity drops below ``threshold`` are placed
    into separate chunks.  Requires the ``chonkie[semantic]`` extra.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/chunk/semantic",
        has_list=isinstance(request.text, list),
        embedding_model=request.embedding_model,
        threshold=request.threshold,
        chunk_size=request.chunk_size,
    )

    text = fix_escaped_text(request.text)

    try:
        # SemanticChunker is intentionally not cached because the embedding
        # model is heavyweight and request parameters vary widely.
        timer.start("init")
        chunker = SemanticChunker(
            embedding_model=request.embedding_model,
            threshold=request.threshold,
            chunk_size=request.chunk_size,
            similarity_window=request.similarity_window,
            min_sentences_per_chunk=request.min_sentences_per_chunk,
            min_characters_per_sentence=request.min_characters_per_sentence,
            delim=request.delim,
            include_delim=request.include_delim,
            skip_window=request.skip_window,
            filter_window=request.filter_window,
            filter_polyorder=request.filter_polyorder,
            filter_tolerance=request.filter_tolerance,
        )
        log.info(
            "Chunker ready",
            endpoint="POST /v1/chunk/semantic",
            chunker="SemanticChunker",
            duration_ms=round(timer.end("init"), 2),
        )

        timer.start("chunk")
        chunks = chunker(text)
        result = _chunks_to_response(chunks)
        chunk_count = _chunk_count(chunks)
        log.info(
            "Chunking completed",
            endpoint="POST /v1/chunk/semantic",
            chunk_count=chunk_count,
            duration_ms=round(timer.end("chunk"), 2),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except ImportError as exc:
        msg = (
            "SemanticChunker requires the 'semantic' extra.  "
            "Install it with: pip install 'chonkie[semantic]'"
        )
        log.error(msg, error=str(exc))
        raise HTTPException(status_code=500, detail=msg) from exc
    except ValueError as exc:
        log.error("Invalid request parameters", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "Semantic chunking failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Semantic chunking failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Code chunker
# ---------------------------------------------------------------------------

_code_cache: Dict[str, CodeChunker] = {}


def _get_code_chunker(tokenizer: str, language: str) -> CodeChunker:
    """Return a cached :class:`CodeChunker` for the given tokenizer/language pair.

    Args:
        tokenizer: Tokenizer name.
        language: Programming language.

    Returns:
        A :class:`CodeChunker` instance (potentially from cache).

    """
    key = f"{tokenizer}:{language}"
    if key not in _code_cache:
        _code_cache[key] = CodeChunker(tokenizer=tokenizer, language=language)
    return _code_cache[key]


@router.post("/code", response_model=None, summary="Chunk source code using AST-based splitting")
async def code_chunk(request: CodeChunkerRequest) -> ChunkingResponse:
    """Chunk source code by respecting syntactic boundaries.

    Uses tree-sitter to parse the code into an AST and splits at logical
    boundaries (functions, classes, etc.) without breaking inside nodes.
    Requires the ``chonkie[code]`` extra.
    """
    timer = Timer()
    timer.start()

    log.info(
        "Request received",
        endpoint="POST /v1/chunk/code",
        has_list=isinstance(request.text, list),
        language=request.language,
        chunk_size=request.chunk_size,
    )

    text = fix_escaped_text(request.text)

    try:
        timer.start("init")
        chunker = _get_code_chunker(
            tokenizer=request.tokenizer,
            language=request.language,
        )
        # Update mutable parameters per-request
        chunker.chunk_size = request.chunk_size
        chunker.include_nodes = request.include_nodes
        log.info(
            "Chunker ready",
            endpoint="POST /v1/chunk/code",
            chunker="CodeChunker",
            duration_ms=round(timer.end("init"), 2),
        )

        timer.start("chunk")
        chunks = chunker(text)
        result = _chunks_to_response(chunks)
        chunk_count = _chunk_count(chunks)
        log.info(
            "Chunking completed",
            endpoint="POST /v1/chunk/code",
            chunk_count=chunk_count,
            duration_ms=round(timer.end("chunk"), 2),
            total_ms=round(timer.elapsed(), 2),
        )
        return result

    except ImportError as exc:
        msg = (
            "CodeChunker requires the 'code' extra.  Install it with: pip install 'chonkie[code]'"
        )
        log.error(msg, error=str(exc))
        raise HTTPException(status_code=500, detail=msg) from exc
    except ValueError as exc:
        log.error("Invalid request parameters", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "Code chunking failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Code chunking failed: {exc}") from exc
