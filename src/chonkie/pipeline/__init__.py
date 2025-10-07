"""Module for pipeline.

This module provides infrastructure for building and managing Chonkie pipelines.
Components can be registered using decorators and then composed into pipelines
that follow the CHOMP (CHOnkie's Multi-step Pipeline) architecture.

The CHOMP pipeline stages are:
1. Fetcher - Retrieve raw data
2. Chef - Preprocess and transform
3. Chunker - Split into chunks
4. Refinery - Post-process chunks
5. Porter - Export to storage formats
6. Handshake - Ingest into vector databases
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
    "chef",
    "chunker",
    "refinery",
    "porter",
    "handshake",
]