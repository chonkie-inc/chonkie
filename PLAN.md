# Implementation Plan: SQLite Pipelines + Catsu Embeddings

## Overview
Add local SQLite database for storing pipeline configurations and switch to Catsu for multi-provider embeddings support.

---

## Phase 1: Database Foundation

### 1.1 Add Dependencies to pyproject.toml
**File:** `pyproject.toml`

Add to `[project.optional-dependencies].api`:
```toml
"sqlalchemy>=2.0.0",      # ORM with async support
"alembic>=1.13.0",        # Database migrations
"aiosqlite>=0.20.0",      # Async SQLite driver
```

### 1.2 Create Database Module
**File:** `src/chonkie/api/database.py`

```python
"""Database configuration and session management."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/chonkie.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    """Dependency for FastAPI endpoints."""
    async with async_session_maker() as session:
        yield session

async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### 1.3 Create Pipeline Model
**File:** `src/chonkie/api/models/__init__.py`

```python
"""Database models."""
from .pipeline import Pipeline

__all__ = ["Pipeline"]
```

**File:** `src/chonkie/api/models/pipeline.py`

```python
"""Pipeline model for storing reusable chunking workflows."""
from sqlalchemy import Column, String, JSON, DateTime, Text
from datetime import datetime, timezone
import uuid

from chonkie.api.database import Base

class Pipeline(Base):
    """A reusable pipeline configuration."""
    
    __tablename__ = "pipelines"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text)
    config = Column(JSON, nullable=False)  # {"steps": [...]}
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
```

### 1.4 Create Alembic Migrations
**File:** `src/chonkie/api/alembic.ini`

Standard Alembic config pointing to `src/chonkie/api/migrations/`.

**File:** `src/chonkie/api/migrations/env.py`

Configure Alembic to use our Base and async engine.

**File:** `src/chonkie/api/migrations/versions/001_initial.py`

Initial migration creating the `pipelines` table.

### 1.5 Update main.py to Initialize Database
**File:** `src/chonkie/api/main.py`

Add startup event:
```python
from chonkie.api.database import init_db

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    await init_db()
```

---

## Phase 2: Pipeline Endpoints

### 2.1 Create Pipeline Schemas
**File:** `src/chonkie/api/schemas.py`

Add new schemas:
```python
class PipelineStepRequest(BaseModel):
    """A single step in a pipeline."""
    type: Literal["chunk", "refine"]
    chunker: Optional[str] = None  # For chunk steps
    refinery: Optional[str] = None  # For refine steps
    config: Dict[str, Any] = Field(default_factory=dict)

class PipelineCreateRequest(BaseModel):
    """Request to create a pipeline."""
    name: str = Field(..., description="Unique pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    steps: List[PipelineStepRequest] = Field(..., description="Pipeline steps")

class PipelineUpdateRequest(BaseModel):
    """Request to update a pipeline."""
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[PipelineStepRequest]] = None

class PipelineExecuteRequest(BaseModel):
    """Request to execute a pipeline."""
    text: Union[str, List[str]] = Field(..., description="Text to process")

class PipelineResponse(BaseModel):
    """Pipeline metadata response."""
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    created_at: str
    updated_at: str
```

### 2.2 Create Pipeline Routes
**File:** `src/chonkie/api/routes/pipelines.py`

```python
"""Pipeline management and execution endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from chonkie.api.database import get_db
from chonkie.api.models import Pipeline
from chonkie.api.schemas import (
    PipelineCreateRequest,
    PipelineUpdateRequest,
    PipelineExecuteRequest,
    PipelineResponse,
    ChunkingResponse,
)
from chonkie.api.utils import get_logger

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
log = get_logger(__name__)

@router.post("", response_model=PipelineResponse, status_code=201)
async def create_pipeline(
    request: PipelineCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new pipeline configuration."""
    # Check if name already exists
    result = await db.execute(select(Pipeline).where(Pipeline.name == request.name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Pipeline '{request.name}' already exists")
    
    pipeline = Pipeline(
        name=request.name,
        description=request.description,
        config={"steps": [step.model_dump() for step in request.steps]}
    )
    db.add(pipeline)
    await db.commit()
    await db.refresh(pipeline)
    
    return pipeline.to_dict()

@router.get("", response_model=List[PipelineResponse])
async def list_pipelines(db: AsyncSession = Depends(get_db)):
    """List all pipelines."""
    result = await db.execute(select(Pipeline).order_by(Pipeline.created_at.desc()))
    pipelines = result.scalars().all()
    return [p.to_dict() for p in pipelines]

@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get pipeline by ID."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return pipeline.to_dict()

@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    request: PipelineUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update a pipeline."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    if request.name and request.name != pipeline.name:
        # Check if new name conflicts
        result = await db.execute(select(Pipeline).where(Pipeline.name == request.name))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail=f"Pipeline name '{request.name}' already exists")
        pipeline.name = request.name
    
    if request.description is not None:
        pipeline.description = request.description
    
    if request.steps is not None:
        pipeline.config = {"steps": [step.model_dump() for step in request.steps]}
    
    await db.commit()
    await db.refresh(pipeline)
    return pipeline.to_dict()

@router.delete("/{pipeline_id}", status_code=204)
async def delete_pipeline(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a pipeline."""
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    await db.delete(pipeline)
    await db.commit()

@router.post("/{pipeline_id}", response_model=ChunkingResponse)
async def execute_pipeline(
    pipeline_id: str,
    request: PipelineExecuteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Execute a pipeline on input text."""
    # Get pipeline
    result = await db.execute(select(Pipeline).where(Pipeline.id == pipeline_id))
    pipeline = result.scalar_one_or_none()
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    # Execute pipeline steps
    # (Implementation to follow - execute each step in order)
    # This will import chunkers/refineries and run them
    
    # For now, return error
    raise HTTPException(
        status_code=501,
        detail="Pipeline execution not yet implemented. Coming in next iteration."
    )
```

### 2.3 Register Pipeline Router
**File:** `src/chonkie/api/main.py`

```python
from chonkie.api.routes.pipelines import router as pipelines_router

app.include_router(pipelines_router, prefix="/v1")
```

---

## Phase 3: Catsu Integration

### 3.1 Update EmbeddingsRefinery Endpoint
**File:** `src/chonkie/api/routes/refineries.py`

Replace OpenAI-only logic:

**Before:**
```python
_ALLOWED_OPENAI_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
}

actual_model = request.embedding_model.replace("openai/", "").replace("OpenAI/", "")

if actual_model not in _ALLOWED_OPENAI_MODELS:
    raise HTTPException(...)

embedding_model = OpenAIEmbeddings(model=actual_model)
```

**After:**
```python
try:
    from catsu import Embeddings
except ImportError:
    raise HTTPException(
        status_code=500,
        detail="Embeddings require catsu. Install with: pip install catsu"
    )

# Catsu auto-detects provider from model name or env vars
embedding_model = Embeddings(model=request.embedding_model)
```

### 3.2 Update Documentation
**File:** `docs/oss/api/endpoints.mdx`

Update embeddings refinery section to show multiple providers:

```markdown
## Embeddings Providers

The API supports multiple embedding providers via [Catsu](https://github.com/chonkie-inc/catsu).

### Supported Providers

- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`
- **Cohere**: `embed-english-v3.0`, `embed-multilingual-v3.0`
- **Voyage AI**: `voyage-large-2`, `voyage-code-2`
- **Mistral**: `mistral-embed`
- **Azure OpenAI**: Use Azure endpoint + API key
- **HuggingFace**: Any sentence-transformers model
- **Local**: Run models locally

### Configuration

Set the API key for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Cohere
export COHERE_API_KEY=...

# Voyage AI
export VOYAGE_API_KEY=...

# Mistral
export MISTRAL_API_KEY=...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
```

### Example Request

```bash
curl -X POST http://localhost:8000/v1/refine/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"text": "First chunk", "start_index": 0, "end_index": 11},
      {"text": "Second chunk", "start_index": 12, "end_index": 24}
    ],
    "embedding_model": "voyage-large-2"
  }'
```
```

**File:** `docs/oss/api/docker.mdx`

Add environment variables section for embeddings providers.

---

## Phase 4: Docker & Persistence

### 4.1 Update Dockerfile
**File:** `Dockerfile`

Add data directory creation:
```dockerfile
# In runtime stage, after WORKDIR /app
RUN mkdir -p /app/data

# Before USER chonkie
RUN chown -R chonkie:chonkie /app/data
```

### 4.2 Update docker-compose.yml
**File:** `docker-compose.yml`

Add volume for database persistence:
```yaml
services:
  api:
    # ... existing config
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=sqlite+aiosqlite:///app/data/chonkie.db
      # Optional: Add embeddings API keys
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - COHERE_API_KEY=${COHERE_API_KEY}
      # - VOYAGE_API_KEY=${VOYAGE_API_KEY}
```

### 4.3 Create .gitignore Entry
**File:** `.gitignore`

Add:
```
data/
*.db
*.db-shm
*.db-wal
```

---

## Phase 5: Documentation

### 5.1 Create Pipelines Documentation
**File:** `docs/oss/api/pipelines.mdx`

New page documenting:
- What pipelines are
- How to create them
- How to execute them
- Example multi-step pipelines
- CRUD operations

### 5.2 Update Navigation
**File:** `docs/docs.json`

Add pipelines page to API Server group:
```json
{
  "group": "API Server",
  "pages": [
    "oss/api/overview",
    "oss/api/quickstart",
    "oss/api/endpoints",
    "oss/api/pipelines",  // NEW
    "oss/api/docker"
  ]
}
```

### 5.3 Update README.md
**File:** `README.md`

Add pipelines to features list and show quick example.

---

## Phase 6: Testing

### 6.1 Manual Testing Checklist

```bash
# Start server with DB
chonkie serve

# Create a pipeline
curl -X POST http://localhost:8000/v1/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag-chunker",
    "description": "Semantic chunking + embeddings",
    "steps": [
      {
        "type": "chunk",
        "chunker": "semantic",
        "config": {"chunk_size": 512}
      },
      {
        "type": "refine",
        "refinery": "embeddings",
        "config": {"embedding_model": "text-embedding-3-small"}
      }
    ]
  }'

# List pipelines
curl http://localhost:8000/v1/pipelines

# Get specific pipeline
curl http://localhost:8000/v1/pipelines/{id}

# Update pipeline
curl -X PUT http://localhost:8000/v1/pipelines/{id} \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'

# Delete pipeline
curl -X DELETE http://localhost:8000/v1/pipelines/{id}

# Test Catsu embeddings with different providers
curl -X POST http://localhost:8000/v1/refine/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [...],
    "embedding_model": "voyage-large-2"
  }'
```

---

## Implementation Order

1. ✅ Database setup (database.py, models, init in main.py)
2. ✅ Alembic migrations
3. ✅ Pipeline schemas
4. ✅ Pipeline CRUD endpoints (create, list, get, update, delete)
5. ⏸️ Pipeline execution (stub for now - "not yet implemented")
6. ✅ Catsu integration (replace OpenAI-only in refineries.py)
7. ✅ Docker updates (volumes, data directory)
8. ✅ Documentation (pipelines.mdx, update other docs)
9. ✅ README updates

**Pipeline execution logic can be Phase 2 in a follow-up PR** - focus on CRUD + Catsu first.

---

## Commit Messages

1. "feat(api): add SQLite database support with SQLAlchemy and Alembic"
2. "feat(api): add pipeline CRUD endpoints for storing reusable workflows"
3. "feat(api): replace OpenAI-only embeddings with Catsu multi-provider support"
4. "docs(api): add pipelines documentation and update embeddings provider info"
5. "chore(docker): add persistent volume for SQLite database"

---

## Notes

- Keep existing chunking/refinery endpoints unchanged (backward compatible)
- Pipeline execution (actually running the steps) can be implemented later
- Focus on: CRUD operations + Catsu embeddings first
- Database file will be created automatically on first run
- Alembic migrations for future schema changes
