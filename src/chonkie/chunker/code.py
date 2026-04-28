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
from chonkie.types.markdown import MarkdownDocument

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

    def _flatten_nodes(self, nodes: list["Node"]) -> list["Node"]:
        """Recursively flatten nodes that exceed chunk_size into their children."""
        result: list["Node"] = []
        for node in nodes:
            node_text = node.text.decode("utf-8") if node.text else ""
            node_tokens = self.tokenizer.count_tokens(node_text)
            if node_tokens > self.chunk_size and node.child_count > 0:
                result.extend(self._flatten_nodes(list(node.children)))
            else:
                result.append(node)
        return result

    def chunk(self, text: str) -> list[Chunk]:
        """Split code text into chunks based on code structure.

        Args:
            text: Code text to be chunked.

        Returns:
            List of Chunk objects containing the chunked code and metadata.

        Raises:
            ImportError: If tree-sitter and tree-sitter-language-pack are not installed.

        """
        if not text or not text.strip():
            return []

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
        tree = self.parser.parse(text.encode("utf-8"))
        root_node = tree.root_node
        nodes = list(root_node.children)

        if not nodes:
            return []

        # Flatten oversized nodes into their children recursively
        nodes = self._flatten_nodes(nodes)

        # Build chunks by grouping consecutive nodes to fit within chunk_size
        text_bytes = text.encode("utf-8")
        chunks: list[Chunk] = []
        current_start = 0
        group_start_byte = 0
        group_token_count = 0

        for i, node in enumerate(nodes):
            node_end_byte = node.end_byte
            candidate_text = text_bytes[group_start_byte:node_end_byte].decode("utf-8")
            candidate_tokens = self.tokenizer.count_tokens(candidate_text)

            if group_token_count > 0 and candidate_tokens > self.chunk_size:
                # Flush current group up to start of this node
                flush_end_byte = node.start_byte
                chunk_text = text_bytes[group_start_byte:flush_end_byte].decode("utf-8")
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_index=current_start,
                        end_index=current_start + len(chunk_text),
                        token_count=self.tokenizer.count_tokens(chunk_text),
                    )
                )
                current_start += len(chunk_text)
                group_start_byte = node.start_byte
                group_token_count = 0

            # Recalculate after potential flush
            candidate_text = text_bytes[group_start_byte:node_end_byte].decode("utf-8")
            group_token_count = self.tokenizer.count_tokens(candidate_text)

        # Flush remaining (include any trailing text after last node)
        remaining_text = text[current_start:]
        if remaining_text:
            chunks.append(
                Chunk(
                    text=remaining_text,
                    start_index=current_start,
                    end_index=current_start + len(remaining_text),
                    token_count=self.tokenizer.count_tokens(remaining_text),
                )
            )

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

        For MarkdownDocument, chunks code blocks and merges with existing chunks.
        For plain Document, chunks the content or re-chunks existing chunks.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        if isinstance(document, MarkdownDocument) and document.code:
            existing_chunks = list(document.chunks) if document.chunks else []
            code_chunks: list[Chunk] = []
            for code_block in document.code:
                if not code_block.content or not code_block.content.strip():
                    continue
                # Set language for this code block
                orig_language = self.language
                orig_parser = self.parser
                try:
                    if code_block.language:
                        self.language = code_block.language
                        from tree_sitter_language_pack import SupportedLanguage, get_parser

                        self.parser = get_parser(cast(SupportedLanguage, code_block.language))

                    block_chunks = self.chunk(code_block.content)
                    # Offset indices by the code block's position in the document
                    for c in block_chunks:
                        code_chunks.append(
                            Chunk(
                                text=c.text,
                                start_index=c.start_index + code_block.start_index,
                                end_index=c.end_index + code_block.start_index,
                                token_count=c.token_count,
                            )
                        )
                finally:
                    self.language = orig_language
                    self.parser = orig_parser
            document.chunks = sorted(existing_chunks + code_chunks, key=lambda c: c.start_index)
        elif document.chunks:
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
        if isinstance(document, MarkdownDocument) and document.code:
            existing_chunks = list(document.chunks) if document.chunks else []
            code_chunks: list[Chunk] = []
            for code_block in document.code:
                if not code_block.content or not code_block.content.strip():
                    continue
                # Set language for this code block
                orig_language = self.language
                orig_parser = self.parser
                try:
                    if code_block.language:
                        self.language = code_block.language
                        from tree_sitter_language_pack import SupportedLanguage, get_parser

                        self.parser = get_parser(cast(SupportedLanguage, code_block.language))

                    block_chunks = await self.achunk(code_block.content)
                    # Offset indices by the code block's position in the document
                    for c in block_chunks:
                        code_chunks.append(
                            Chunk(
                                text=c.text,
                                start_index=c.start_index + code_block.start_index,
                                end_index=c.end_index + code_block.start_index,
                                token_count=c.token_count,
                            )
                        )
                finally:
                    self.language = orig_language
                    self.parser = orig_parser
            document.chunks = sorted(existing_chunks + code_chunks, key=lambda c: c.start_index)
        elif document.chunks:
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
