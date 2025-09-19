"""Module containing CodeChunker configuration types."""

from dataclasses import dataclass
from typing import List, Optional, Union


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
