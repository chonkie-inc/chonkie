"""Pipeline management endpoints.

All endpoints live under the ``/v1/pipelines`` prefix.
"""

from typing import List

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
from chonkie.api.utils import get_logger

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
log = get_logger(__name__)


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
    """List all pipelines."""
    result = await db.execute(select(Pipeline).order_by(Pipeline.created_at.desc()))
    pipelines = result.scalars().all()
    return [p.to_dict() for p in pipelines]


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a pipeline by ID."""
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
    """Update a pipeline."""
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
        pipeline.name = request.name

    if request.description is not None:
        pipeline.description = request.description

    if request.steps is not None:
        pipeline.config = {"steps": [step.model_dump() for step in request.steps]}

    await db.commit()
    await db.refresh(pipeline)
    log.info("Pipeline updated", pipeline_id=pipeline.id)
    return pipeline.to_dict()


@router.delete("/{pipeline_id}", status_code=204)
async def delete_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a pipeline."""
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
) -> None:
    """Execute a pipeline on input text.

    .. note::
        Pipeline execution is not yet implemented. This endpoint will be
        available in a future release.
    """
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    raise HTTPException(
        status_code=501,
        detail="Pipeline execution not yet implemented. Coming in next iteration.",
    )
