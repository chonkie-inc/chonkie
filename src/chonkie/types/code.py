"""Module containing CodeChunker types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from chonkie.types.base import Chunk

if TYPE_CHECKING:
    try:
        from tree_sitter import Node
    except ImportError:
        Node = Any  # type: ignore


@dataclass
class CodeChunk(Chunk):
    """Code chunk with metadata."""

    lang: Optional[str] = None
    nodes: Optional[List["Node"]] = None