"""Module containing CodeChunker types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from chonkie.types.base import Chunk

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class CodeChunk(Chunk):
    """Code chunk with metadata."""

    lang: Optional[str] = None
    nodes: Optional[List["Node"]] = None