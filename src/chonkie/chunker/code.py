"""Module containing CodeChunker class.

This module provides a CodeChunker class for splitting code into chunks of a specified size.

"""

import warnings
import logging  # <- added for logging
from bisect import bisect_left
from itertools import accumulate
from typing import TYPE_CHECKING, Any, List, Literal, Tuple, Union

from chonkie.chunker.base import BaseChunker
from chonkie.tokenizer import Tokenizer
from chonkie.types import Chunk

if TYPE_CHECKING:
    from typing import Any
    try:
        from tree_sitter import Node
    except ImportError:
        class Node:  # type: ignore
            """Stub class for tree_sitter Node when not available."""
            pass


class CodeChunker(BaseChunker):
    """Chunker that recursively splits the code based on code context."""

    # Logger for this class
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    def __init__(self,
                 tokenizer_or_token_counter: Union[str, List, Any] = "character",
                 chunk_size: int = 2048,
                 language: Union[Literal["auto"], Any] = "auto",
                 include_nodes: bool = False,
                 enable_logging: bool = True  # new enhancement
                 ) -> None:
        """Initialize a CodeChunker object with optional logging."""
        self.enable_logging = enable_logging

        # Lazy import dependencies
        self._import_dependencies()

        # Initialize all the values
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)
        self.chunk_size = chunk_size
        self.include_nodes = include_nodes

        # Language handling
        self.language = language
        if language == "auto":
            warnings.warn("The language is set to `auto`. This may affect performance.")
            self.magika = Magika()  # type: ignore
            self.parser = None
        else:
            self.parser = get_parser(language)  # type: ignore

        self._use_multiprocessing = False

        # Logging
        if self.enable_logging:
            self.logger.info(f"CodeChunker initialized with chunk_size={self.chunk_size}, language={self.language}")

    def _import_dependencies(self) -> None:
        """Import the dependencies for the CodeChunker."""
        try:
            global Node, Parser, Tree
            global get_parser, SupportedLanguage, Magika

            from magika import Magika
            from tree_sitter import Node, Parser, Tree
            from tree_sitter_language_pack import SupportedLanguage, get_parser
        except ImportError:
            raise ImportError(
                "One or more dependencies not installed: [tree-sitter, tree-sitter-language-pack, magika]. "
                "Install with `pip install chonkie[code]`."
            )

    # All other methods remain unchanged, except we add logging where useful

    def chunk(self, text: str) -> List[Chunk]:
        """Recursively chunks the code based on context from tree-sitter."""
        if not text.strip():
            return []

        original_text_bytes = text.encode("utf-8")

        if self.language == "auto":
            language = self._detect_language(original_text_bytes)
            self.parser = get_parser(language)  # type: ignore

        try:
            tree: Tree = self.parser.parse(original_text_bytes)  # type: ignore
            root_node: Node = tree.root_node  # type: ignore

            node_groups, token_counts = self._group_child_nodes(root_node)
            texts: List[str] = self._get_texts_from_node_groups(node_groups, original_text_bytes)
        finally:
            if not self.include_nodes:
                del tree, root_node
                node_groups = []

        chunks = self._create_chunks(texts, token_counts, node_groups)

        # Logging after chunks are created
        if self.enable_logging:
            self.logger.info(f"Created {len(chunks)} chunks for text of length {len(text)}")

        return chunks

    def __repr__(self) -> str:
        """Return the string representation of the CodeChunker."""
        return (f"CodeChunker(tokenizer_or_token_counter={self.tokenizer}, "
                f"chunk_size={self.chunk_size}, language={self.language})")
