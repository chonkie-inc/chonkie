"""Chonkie OSS API – FastAPI application entry point.

Run locally::

    chonkie serve

Or with uvicorn directly::

    uvicorn chonkie.api.main:app --reload --port 8000

Or via Docker::

    docker compose up
"""

import os
from importlib.metadata import version, PackageNotFoundError

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chonkie.api.routes.chunking import router as chunking_router
from chonkie.api.routes.refineries import router as refineries_router
from chonkie.api.utils import configure_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    _chonkie_version = version("chonkie")
except PackageNotFoundError:
    _chonkie_version = "unknown"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Chonkie OSS API",
    description=(
        "A lightweight, self-hostable REST API that exposes the "
        "[Chonkie](https://github.com/chonkie-ai/chonkie) chunking library "
        "over HTTP.\n\n"
        "No authentication or billing required – just run it and chunk away.\n\n"
        "**Source:** https://github.com/chonkie-ai/chonkie"
    ),
    version=_chonkie_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS – permissive by default for local development.
# Override with the CORS_ORIGINS env var (comma-separated list of origins).
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("CORS_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(chunking_router, prefix="/v1")
app.include_router(refineries_router, prefix="/v1")


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"], summary="Health check")
async def health() -> dict:
    """Return a simple alive signal.

    Useful for container health-checks and load-balancer probes.
    """
    return {"status": "ok"}


@app.get("/", tags=["Meta"], summary="API information")
async def root() -> dict:
    """Return basic information about this API instance."""
    return {
        "name": "Chonkie OSS API",
        "version": _chonkie_version,
        "docs": "/docs",
        "health": "/health",
        "chunkers": [
            "/v1/chunk/token",
            "/v1/chunk/sentence",
            "/v1/chunk/recursive",
            "/v1/chunk/semantic",
            "/v1/chunk/code",
        ],
        "refineries": [
            "/v1/refine/embeddings",
            "/v1/refine/overlap",
        ],
    }
