"""Module for pipeline.

This module provides infrastructure for building and managing Chonkie pipelines.
Components can be registered using decorators and then composed into pipelines
that follow the CHOMP (CHOnkie's Multi-step Pipeline) architecture.

The CHOMP pipeline stages are:
1. Fetcher - Retrieve raw data
2. Vision - Extract text from visual documents (OCR)
3. Chef - Preprocess and transform
4. Chunker - Split into chunks
5. Refinery - Post-process chunks
6. Porter - Export to storage formats
7. Handshake - Ingest into vector databases
"""

from .component import Component, ComponentType
from .pipeline import Pipeline
from .registry import (
    ComponentRegistry,
    chef,
    chunker,
    fetcher,
    handshake,
    pipeline_component,
    porter,
    refinery,
    vision,
)

__all__ = [
    # Core types
    "Component",
    "ComponentType",
    "ComponentRegistry",
    "Pipeline",
    # Decorators
    "pipeline_component",
    "fetcher",
    "vision",
    "chef",
    "chunker",
    "refinery",
    "porter",
    "handshake",
]
