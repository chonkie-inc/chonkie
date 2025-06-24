"""Module containing CodeChunker types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .base import Chunk


@dataclass
class MergeRule:
  """Configuration for merging adjacent nodes of specific types."""

  name: str
  node_types: List[str]
  text_pattern: Optional[str] = None
  bidirectional: bool = False

@dataclass
class SplitRule:
  """Configuration for splitting large nodes into smaller chunks.
  
  Args:
    name: Descriptive name for the rule
    node_type: The AST node type to apply this rule to
    body_child: Path to the body node to split. Can be:
      - str: Direct child name (e.g., "class_body")
      - List[str]: Path through nested children (e.g., ["class_declaration", "class_body"])
    exclude_nodes: Optional list of node types to exclude from splitting (e.g., structural punctuation)
    recursive: If True, recursively apply splitting to child nodes of body_child type that exceed chunk_size

  """

  name: str
  node_type: str
  body_child: Union[str, List[str]]
  exclude_nodes: Optional[List[str]] = None
  recursive: bool = False

@dataclass
class LanguageConfig:
  """Configuration for a specific programming language's chunking rules."""

  language: str
  merge_rules: List[MergeRule]
  split_rules: List[SplitRule]

@dataclass
class CodeChunk(Chunk):
  """A chunk of code with language-specific metadata."""

  language: Optional[str] = None
  nodes: Optional[List[Dict[str, Any]]] = None
  node_type: Optional[str] = None
  start_line: Optional[int] = None
  end_line: Optional[int] = None

  def to_dict(self) -> dict:
    """Return the Chunk as a dictionary."""
    result = super().to_dict()
    result["language"] = self.language
    result["nodes"] = self.nodes
    result["node_type"] = self.node_type
    result["start_line"] = self.start_line
    result["end_line"] = self.end_line
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "CodeChunk":
    """Create a Chunk object from a dictionary."""
    if "language" not in data and "lang" in data:
      data["language"] = data.pop("lang") # Compatibility with old versions
    return cls(**data)
