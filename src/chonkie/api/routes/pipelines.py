"""Pipeline management endpoints.

All endpoints live under the ``/v1/pipelines`` prefix.
"""

import asyncio
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chonkie.api.database import get_db
from chonkie.api.models import Pipeline
from chonkie.api.schemas import (
    ChunkingResponse,
    PipelineCreateRequest,
    PipelineExecuteRequest,
    PipelineResponse,
    PipelineUpdateRequest,
)
from chonkie.api.utils import Timer, fix_escaped_text, get_logger

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
log = get_logger(__name__)


def _run_chunk_step(chunker_name: str, config: dict, text: Any) -> Any:
    """Instantiate a chunker and run it on *text*.

    Args:
        chunker_name: Lowercase chunker name (e.g. ``"token"``).
        config: Keyword arguments forwarded to the chunker constructor.
        text: A string or list of strings to chunk.

    Returns:
        ``List[Chunk]`` for a single text input, ``List[List[Chunk]]`` for a
        list of texts.

    Raises:
        ValueError: For unknown chunker names or bad config.

    """
    from chonkie import (
        CodeChunker,
        FastChunker,
        LateChunker,
        NeuralChunker,
        RecursiveChunker,
        SemanticChunker,
        SentenceChunker,
        SlumberChunker,
        TableChunker,
        TokenChunker,
    )

    chunker_cls_map = {
        "token": TokenChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "code": CodeChunker,
        "late": LateChunker,
        "neural": NeuralChunker,
        "slumber": SlumberChunker,
        "table": TableChunker,
        "fast": FastChunker,
    }

    key = chunker_name.lower()
    chunker_cls = chunker_cls_map.get(key)
    if chunker_cls is None:
        raise ValueError(
            f"Unknown chunker '{chunker_name}'. "
            f"Valid options: {sorted(chunker_cls_map.keys())}"
        )

    cfg = dict(config)

    # RecursiveChunker supports a named recipe via a class-method constructor.
    if chunker_cls is RecursiveChunker and "recipe" in cfg:
        recipe = cfg.pop("recipe")
        lang = cfg.pop("lang", "en")
        tokenizer = cfg.pop("tokenizer", "character")
        chunk_size = cfg.pop("chunk_size", 512)
        min_chars = cfg.pop("min_characters_per_chunk", 24)
        chunker = RecursiveChunker.from_recipe(name=recipe, lang=lang, tokenizer=tokenizer)
        chunker.chunk_size = chunk_size
        chunker.min_characters_per_chunk = min_chars
    else:
        try:
            chunker = chunker_cls(**cfg)
        except TypeError as exc:
            raise ValueError(f"Invalid config for {chunker_name}: {exc}") from exc

    return chunker(text)


def _run_refine_step(refinery_name: str, config: dict, chunks: list) -> list:
    """Instantiate a refinery and run it on *chunks*.

    Args:
        refinery_name: Lowercase refinery name (e.g. ``"overlap"``).
        config: Keyword arguments forwarded to the refinery constructor.
        chunks: ``List[Chunk]`` to refine.

    Returns:
        Refined ``List[Chunk]``.

    Raises:
        ValueError: For unknown refinery names or bad config.

    """
    from typing import Union

    from chonkie import EmbeddingsRefinery, OverlapRefinery

    refinery_cls_map = {
        "overlap": OverlapRefinery,
        "embeddings": EmbeddingsRefinery,
    }

    key = refinery_name.lower()
    refinery_cls = refinery_cls_map.get(key)
    if refinery_cls is None:
        raise ValueError(
            f"Unknown refinery '{refinery_name}'. "
            f"Valid options: {sorted(refinery_cls_map.keys())}"
        )

    cfg = dict(config)

    ref: Union[EmbeddingsRefinery, OverlapRefinery]
    if refinery_cls is EmbeddingsRefinery:
        embedding_model = cfg.pop("embedding_model", "text-embedding-3-small")
        try:
            from catsu import Embeddings

            embedding = Embeddings(model=embedding_model)
            ref = EmbeddingsRefinery(embedding_model=embedding)
        except ImportError:
            # Fall back to AutoEmbeddings (e.g. model2vec for local models)
            ref = EmbeddingsRefinery(embedding_model=embedding_model)
    elif refinery_cls is OverlapRefinery:
        # Never mutate the chunks that might be reused by later steps
        cfg.setdefault("inplace", False)
        try:
            ref = OverlapRefinery(**cfg)
        except TypeError as exc:
            raise ValueError(f"Invalid config for {refinery_name}: {exc}") from exc
    else:
        try:
            ref = refinery_cls(**cfg)
        except TypeError as exc:
            raise ValueError(f"Invalid config for {refinery_name}: {exc}") from exc

    return ref.refine(chunks)


@router.post("", response_model=PipelineResponse, status_code=201)
async def create_pipeline(
    request: PipelineCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new pipeline configuration."""
    result = await db.execute(select(Pipeline).where(Pipeline.name == request.name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Pipeline '{request.name}' already exists")

    pipeline = Pipeline(
        name=request.name,
        description=request.description,
        config={"steps": [step.model_dump() for step in request.steps]},
    )
    db.add(pipeline)
    await db.commit()
    await db.refresh(pipeline)

    log.info("Pipeline created", pipeline_id=pipeline.id, name=pipeline.name)
    return pipeline.to_dict()


@router.get("", response_model=List[PipelineResponse])
async def list_pipelines(db: AsyncSession = Depends(get_db)) -> list:
    """List all pipelines, ordered by most recently created."""
    result = await db.execute(select(Pipeline).order_by(Pipeline.created_at.desc()))
    pipelines = result.scalars().all()
    return [p.to_dict() for p in pipelines]


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Fetch a single pipeline by ID."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return pipeline.to_dict()


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    request: PipelineUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a pipeline's name, description, or steps."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    if request.name and request.name != pipeline.name:
        conflict = await db.execute(select(Pipeline).where(Pipeline.name == request.name))
        if conflict.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline name '{request.name}' already exists",
            )
        pipeline.name = request.name  # type: ignore[assignment]

    if request.description is not None:
        pipeline.description = request.description  # type: ignore[assignment]

    if request.steps is not None:
        pipeline.config = {"steps": [step.model_dump() for step in request.steps]}  # type: ignore[assignment]

    await db.commit()
    await db.refresh(pipeline)
    log.info("Pipeline updated", pipeline_id=pipeline.id)
    return pipeline.to_dict()


@router.delete("/{pipeline_id}", status_code=204)
async def delete_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a pipeline by ID."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    await db.delete(pipeline)
    await db.commit()
    log.info("Pipeline deleted", pipeline_id=pipeline_id)


@router.post("/{pipeline_id}/execute", response_model=ChunkingResponse)
async def execute_pipeline(
    pipeline_id: str,
    request: PipelineExecuteRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Execute a pipeline on input text.

    Steps run sequentially.  Each ``chunk`` step produces
    :class:`~chonkie.types.Chunk` objects; each ``refine`` step enriches them.
    The final state must contain chunks, which are returned as a list of dicts
    (or a list-of-lists for batch input).

    **Step types**

    * ``chunk`` – requires a ``chunker`` field (e.g. ``"token"``) and optional
      ``config`` keyword arguments for the chunker constructor.
    * ``refine`` – requires a ``refinery`` field (e.g. ``"overlap"``) and
      optional ``config`` keyword arguments for the refinery constructor.

    **Chunkers**: ``token``, ``sentence``, ``recursive``, ``semantic``,
    ``code``, ``late``, ``neural``, ``slumber``, ``table``, ``fast``.

    **Refineries**: ``overlap``, ``embeddings``.
    """
    timer = Timer()
    timer.start()

    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    steps = pipeline.config.get("steps", [])
    if not steps:
        raise HTTPException(status_code=400, detail="Pipeline has no steps defined")

    log.info(
        "Pipeline execution started",
        pipeline_id=pipeline_id,
        pipeline_name=pipeline.name,
        step_count=len(steps),
        is_batch=isinstance(request.text, list),
    )

    text = fix_escaped_text(request.text)
    is_batch = isinstance(text, list)

    # ``state`` holds the current data flowing through the pipeline.
    # It starts as text and becomes chunks after the first chunk step.
    state: Any = text
    has_chunks = False

    for i, step in enumerate(steps):
        step_type = step.get("type")
        config = dict(step.get("config") or {})

        log.info(
            "Executing step",
            pipeline_id=pipeline_id,
            step_index=i,
            step_type=step_type,
        )
        timer.start(f"step_{i}")

        if step_type == "chunk":
            chunker_name = step.get("chunker")
            if not chunker_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Step {i}: chunk step requires a 'chunker' field",
                )

            # If we already have chunks from a previous step, extract their
            # text so we can re-chunk it (e.g. chunk → refine → chunk).
            if has_chunks:
                if is_batch:
                    state = [" ".join(c.text for c in chunk_list) for chunk_list in state]
                else:
                    state = " ".join(c.text for c in state)
                has_chunks = False

            try:
                state = await asyncio.to_thread(_run_chunk_step, chunker_name, config, state)
                has_chunks = True
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Step {i} (chunk/{chunker_name}): {exc}",
                ) from exc
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Step {i} (chunk/{chunker_name}): missing dependency – {exc}",
                ) from exc
            except Exception as exc:
                log.error(
                    "Chunk step failed",
                    pipeline_id=pipeline_id,
                    step_index=i,
                    chunker=chunker_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Step {i} (chunk/{chunker_name}) execution failed: {exc}",
                ) from exc

        elif step_type == "refine":
            if not has_chunks:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Step {i}: refine step requires chunks as input. "
                        "Add a chunk step before this refine step."
                    ),
                )

            refinery_name = step.get("refinery")
            if not refinery_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Step {i}: refine step requires a 'refinery' field",
                )

            try:
                if is_batch:
                    refined: list = []
                    for chunk_list in state:
                        r = await asyncio.to_thread(
                            _run_refine_step, refinery_name, config, chunk_list
                        )
                        refined.append(r)
                    state = refined
                else:
                    state = await asyncio.to_thread(
                        _run_refine_step, refinery_name, config, state
                    )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Step {i} (refine/{refinery_name}): {exc}",
                ) from exc
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Step {i} (refine/{refinery_name}): missing dependency – {exc}",
                ) from exc
            except Exception as exc:
                log.error(
                    "Refine step failed",
                    pipeline_id=pipeline_id,
                    step_index=i,
                    refinery=refinery_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Step {i} (refine/{refinery_name}) execution failed: {exc}",
                ) from exc

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Step {i}: unknown step type '{step_type}'. "
                    "Valid types: 'chunk', 'refine'."
                ),
            )

        log.info(
            "Step completed",
            pipeline_id=pipeline_id,
            step_index=i,
            step_type=step_type,
            duration_ms=round(timer.end(f"step_{i}"), 2),
        )

    if not has_chunks:
        raise HTTPException(
            status_code=400,
            detail="Pipeline produced no chunks. Ensure it contains at least one chunk step.",
        )

    if is_batch:
        response = [[c.to_dict() for c in chunk_list] for chunk_list in state]
    else:
        response = [c.to_dict() for c in state]

    chunk_count = sum(len(cl) for cl in response) if is_batch else len(response)
    log.info(
        "Pipeline execution completed",
        pipeline_id=pipeline_id,
        chunk_count=chunk_count,
        total_ms=round(timer.elapsed(), 2),
    )

    return response
