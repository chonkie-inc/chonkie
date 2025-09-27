"""Markdown types for Chonkie."""

from dataclasses import dataclass, field
from typing import List, Optional

from .document import Document


@dataclass 
class MarkdownTable:
    """MarkdownTable is a table found in the middle of a markdown document."""

    content: str = field(default_factory=str)
    start_index: int = field(default_factory=int)
    end_index: int = field(default_factory=int)

@dataclass 
class MarkdownCode:
    """MarkdownCode is a code block found in the middle of a markdown document."""

    content: str = field(default_factory=str)
    language: Optional[str] = field(default=None)
    start_index: int = field(default_factory=int)
    end_index: int = field(default_factory=int)

@dataclass
class MarkdownImage:
    """MarkdownImage is an image found in the middle of a markdown document."""

    alias: str = field(default_factory=str)
    content: str = field(default_factory=str)
    start_index: int = field(default_factory=int)
    end_index: int = field(default_factory=int)
    link: Optional[str] = field(default=None)

@dataclass
class MarkdownDocument(Document):
    """MarkdownDocument is a document that contains markdown content."""

    tables: List[MarkdownTable] = field(default_factory=list)
    code: List[MarkdownCode] = field(default_factory=list)
    images: List[MarkdownImage] = field(default_factory=list)