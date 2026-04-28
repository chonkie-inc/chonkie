"""Module containing CodeChunker class.

This module provides a CodeChunker class for splitting code into chunks of a specified size.

"""

import asyncio
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk, Document

logger = get_logger(__name__)

if TYPE_CHECKING:
    from tree_sitter import Node


@chunker("code")
class CodeChunker(BaseChunker):
    """Chunker that recursively splits the code based on code context.

    Args:
        tokenizer: The tokenizer to use.
        chunk_size: The size of the chunks to create.
        chunk_overlap: Number of tokens to overlap between chunks.
        language: The language of the code to parse. Accepts any of the languages
            supported by tree-sitter-language-pack.
        include_nodes: Whether to include the nodes in the returned chunks.

    """

    def __init__(
        self,
        tokenizer: str | TokenizerProtocol = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        language: Literal["auto"] | str = "auto",
        include_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a CodeChunker object.

        Args:
            tokenizer: The tokenizer to use.
            chunk_size: The size of the chunks to create.
            chunk_overlap: Number of tokens to overlap between chunks.
            language: The language of the code to parse. Accepts any of the languages
                supported by tree-sitter-language-pack.
            include_nodes: Whether to include the nodes in the returned chunks.
            **kwargs: Additional overlap parameters passed to BaseChunker

        Raises:
            ImportError: If tree-sitter and tree-sitter-language-pack are not installed.
            ValueError: If the language is not supported.

        """
        # Initialize the base chunker
        super().__init__(tokenizer=tokenizer, chunk_overlap=chunk_overlap, **kwargs)

        # Initialize chunker-specific values
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_nodes = include_nodes
        self.language = language

        # NOTE: Magika is a language detection library made by Google, that uses a
        #       deep-learning model to detect the language of the code.

        # Initialize the Magika instance if the language is auto
        if language == "auto":
            logger.warning(
                "The language is set to `auto`. This would adversely affect the "
                "performance of the chunker. "
                "Consider setting the `language` parameter to a specific language "
                "to improve performance.",
            )
            from magika import Magika

            # Set the language to auto and initialize the Magika instance
            self.magika = Magika()
            self.parser = None
        else:
            from tree_sitter_language_pack import SupportedLanguage, get_parser

            try:
                self.parser = get_parser(cast(SupportedLanguage, language))
            except LookupError as e:
                raise ValueError(
                    f"Unsupported language '{language}'. "
                    f"Supported languages are: {list(get_args(SupportedLanguage))}. "
                    "Or set language='auto'."
                ) from e

        # Set the use_multiprocessing flag
        self._use_multiprocessing = False

    def _detect_language(self, bytes_text: bytes) -> Any:
        """Detect the language of the code."""
        response = self.magika.identify_bytes(bytes_text)
        return response.output.label

    def _merge_node_groups(self, node_groups: list[list["Node"]]) -> list["Node"]:
        """Merge node groups by joining sibling nodes."""
        merged = []
        for group in node_groups:
            merged_nodes = []
            for node in group:
                # Find all siblings
                siblings = [node]
                sibling = node.next_named_sibling
                while sibling:
                    siblings.append(sibling)
                    sibling = sibling.next_named_sibling
                # Join the merged text
                merged_text = "\n".join([
                    (s.text.decode("utf-8") if s.text else "") for s in siblings
                ])
                merged_nodes.append(node)
                merged.append(merged_text)
        return merged_nodes

    def _create_chunks(self, code_text: str, nodes: list["Node"]) -> list[Chunk]:
        """Create chunks from code text and nodes."""
        chunks = []
        for node in nodes:
            # Get the text of the node
            node_text = node.text.decode("utf-8") if node.text else str(node)
            # Get start and end indices
            start_index = node.start_byte
            end_index = node.end_byte
            # Create the chunk
            chunks.append(
                Chunk(
                    text=node_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=self.tokenizer.count_tokens(node_text),
                )
            )
        return chunks

    def chunk(self, text: str) -> list[Chunk]:
        """Split code text into chunks based on code structure.

        Args:
            text: Code text to be chunked.

        Returns:
            List of Chunk objects containing the chunked code and metadata.

        Raises:
            ImportError: If tree-sitter and tree-sitter-language-pack are not installed.

        """
        logger.debug(f"Chunking code of length {len(text)}")

        # Detect language if auto
        if self.language == "auto":
            from magika import Magika

            self.magika = Magika()
            lang = self._detect_language(text.encode("utf-8"))
            from tree_sitter_language_pack import SupportedLanguage, get_parser

            self.parser = get_parser(cast(SupportedLanguage, lang))
            self.language = lang

        # Parse the code
        assert self.parser is not None
        tree = self.parser.parse(bytes(text.encode("utf-8")))
        root_node = tree.root_node
        # Get all nodes to chunk
        nodes = list(root_node.children)
        # Merge nodes if needed
        merged_nodes = self._merge_node_groups(nodes)
        # Create chunks
        chunks = self._create_chunks(text, merged_nodes)
        logger.info(f"Created {len(chunks)} chunks from code")
        return chunks

    def chunk_batch(  # type: ignore[override]
        self, texts: list[str], batch_size: int = 1, show_progress_bar: bool = True
    ) -> list[list[Chunk]]:
        """Split a batch of code texts into their respective chunks.

        Args:
            texts: List of code texts to be chunked.
            batch_size: Number of texts to process in a single batch.
            show_progress_bar: Whether to show a progress bar.

        Returns:
            List of lists of Chunk objects containing the chunked code and metadata.

        """
        from tqdm import trange

        chunks: list[list[Chunk]] = []
        for i in trange(
            0,
            len(texts),
            batch_size,
            desc="🦛",
            disable=not show_progress_bar,
            unit="batch",
            bar_format=(
                "{desc} ch{bar:20}nk "
                "{percentage:3.0f}% • {n_fmt}/{total_fmt} batches chunked "
                "[{elapsed}<{remaining}, {rate_fmt}] 🌱"
            ),
            ascii=" o",
        ):
            batch_texts = texts[i : min(i + batch_size, len(texts))]
            for text in batch_texts:
                chunks.append(self.chunk(text))
        return chunks

    def __call__(  # type: ignore[override]
        self,
        text: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> list[Chunk] | list[list[Chunk]]:
        """Make the CodeChunker callable directly.

        Args:
            text: Code text or list of code texts to be chunked.
            batch_size: Number of texts to process in a single batch.
            show_progress_bar: Whether to show a progress bar (for batch chunking).

        Returns:
            List of Chunk objects or list of lists of Chunk.

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list) and isinstance(text[0], str):
            return self.chunk_batch(text, batch_size, show_progress_bar)
        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")

    def __repr__(self) -> str:
        """Return a string representation of the CodeChunker."""
        return (
            f"CodeChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"language={self.language}, "
            f"include_nodes={self.include_nodes})"
        )

    def chunk_document(self, document: Document) -> Document:
        """Chunk a document.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            chunk_results = [self.chunk(c.text) for c in document.chunks]
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = self.chunk(document.content)
        self._propagate_document_metadata(document)
        return document

    async def achunk_document(self, document: Document) -> Document:
        """Chunk a document asynchronously.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            tasks = [self.achunk(c.text) for c in document.chunks]
            chunk_results = await asyncio.gather(*tasks)
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = await self.achunk(document.content)
        self._propagate_document_metadata(document)
        return document

    @staticmethod
    def _merge_new_chunks(
        original_chunks: list[Chunk], new_chunk_batches: list[list[Chunk]]
    ) -> list[Chunk]:
        """Merge new chunks batches into a single list, shifting indices.

        Args:
            original_chunks: The original chunks from the document.
            new_chunk_batches: The new batches of chunks corresponding to each
                original chunk.

        Returns:
            list[Chunk]: The merged and shifted chunks.

        """
        from dataclasses import replace

        return [
            replace(
                c,
                start_index=c.start_index + old_chunk.start_index,
                end_index=c.end_index + old_chunk.start_index,
            )
            for old_chunk, new_chunks in zip(original_chunks, new_chunk_batches)
            for c in new_chunks
        ]

    async def achunk(self, text: str) -> list[Chunk]:
        """Chunk the given text asynchronously.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: A list of Chunks.

        """
        return await asyncio.to_thread(self.chunk, text)
