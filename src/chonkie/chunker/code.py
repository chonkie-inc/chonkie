"""Module containing CodeChunker class.

This module provides a CodeChunker class for splitting code into chunks of a specified size.

"""

from typing import TYPE_CHECKING, Literal

from chonkie.chunker.base import BaseChunker
from chonkie.logger import LoggerAdapter, get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk, Document, MarkdownDocument

logger: LoggerAdapter = get_logger(__name__)

if TYPE_CHECKING:
    from tree_sitter_language_pack import CodeChunk


def _detect_language_by_parsing(text: str) -> str | None:
    """Detect language by trial-parsing with all downloaded grammars.

    Tries each available language grammar and scores by:
      1. Fewest parse errors (primary)
      2. Most structure items + imports found (secondary)
      3. Most AST nodes (tiebreaker)

    Returns the best-matching language or None if no languages are downloaded.
    """
    from tree_sitter_language_pack import ProcessConfig, downloaded_languages, process

    languages = downloaded_languages()
    if not languages:
        return None

    results: list[tuple[str, int, int, int]] = []
    for lang in languages:
        try:
            config = ProcessConfig(language=lang, structure=True, imports=True)
            result = process(text, config)  # ty: ignore[invalid-argument-type]
            structure_score = len(result.structure) + len(result.imports)
            results.append((
                lang,
                result.metrics.error_count,
                structure_score,
                result.metrics.node_count,
            ))
        except Exception as e:
            logger.debug(f"Failed to parse with language '{lang}': {e}")
            continue

    if not results:
        return None

    results.sort(key=lambda x: (x[1], -x[2], -x[3]))
    return results[0][0]


@chunker("code")
class CodeChunker(BaseChunker):
    """Chunker that recursively splits the code based on code context.

    Args:
        tokenizer: The tokenizer to use.
        chunk_size: The size of the chunks to create.
        language: The language of the code to parse. Accepts any of the languages supported by tree-sitter-language-pack.
        include_nodes: Whether to include the nodes in the returned chunks.

    """

    def __init__(
        self,
        tokenizer: str | TokenizerProtocol = "character",
        chunk_size: int = 2048,
        language: Literal["auto"] | str = "auto",
        include_nodes: bool = False,
    ) -> None:
        """Initialize a CodeChunker object.

        Args:
            tokenizer: The tokenizer to use.
            chunk_size: The size of the chunks to create.
            language: The language of the code to parse. Accepts any of the languages supported by tree-sitter-language-pack.
            include_nodes: Whether to include the nodes in the returned chunks.

        Raises:
            ImportError: If tree-sitter-language-pack is not installed.
            ValueError: If the language is not supported.

        """
        super().__init__(tokenizer=tokenizer)

        self.chunk_size = chunk_size
        self.include_nodes = include_nodes

        self.language = language

        # download tree_sitter_languages for first time when we use the chunker
        from tree_sitter_language_pack import download_all, downloaded_languages

        num_languages = len(downloaded_languages())
        # in version 1.6.6, when we ran the tests we downloaded 19 languages
        # this will download the rest of them for all onward versions
        # but it won't redownload for users who already have 19 or more languages downloaded
        if num_languages <= 19:
            logger.info("Downloading tree-sitter languages for the first time...")
            download_all()

        # load the language
        if language == "auto":
            # Set a debug message to the user that the language is auto and this might
            # affect the performance of the chunker.
            logger.debug(
                "The language is set to `auto`. This would adversely affect the performance of the chunker. "
                "Consider setting the `language` parameter to a specific language to improve performance.",
            )
        else:
            from tree_sitter_language_pack import download, has_language

            # backward compatibility: if the language is not downloaded yet
            # since this is recently made we can get rid of this block in the future
            if not has_language(language):
                download([language])

            if not has_language(language):
                raise ValueError(
                    f"Unsupported language '{language}'. "
                    "Use `tree_sitter_language_pack.available_languages()` for a full list, "
                    "or set language='auto'."
                )

        self._use_multiprocessing = False

    def _detect_language(self, text: str) -> str:
        """Detect the language of the code.

        Tries in order:
          1. Shebang detection (instant, exact)
          2. Trial-parsing with all downloaded grammars (picks fewest errors + most structure)
        """
        from tree_sitter_language_pack import detect_language_from_content

        language = detect_language_from_content(text)
        if language is not None:
            return language

        language = _detect_language_by_parsing(text)
        if language is not None:
            return language

        raise ValueError(
            "Could not auto-detect the language. Please specify a language explicitly."
        )

    def _estimate_chunk_max_bytes(self, text: str) -> int:
        """Estimate the byte size corresponding to chunk_size tokens."""
        text_bytes = len(text.encode("utf-8"))
        text_tokens = self.tokenizer.count_tokens(text)
        if text_tokens == 0:
            return text_bytes
        bytes_per_token = text_bytes / text_tokens
        return max(1, int(self.chunk_size * bytes_per_token))

    def _process_code(self, text: str, language: str) -> list["CodeChunk"]:
        """Process code using tree-sitter-language-pack's process() API."""
        from tree_sitter_language_pack import ProcessConfig, process

        chunk_max_bytes = self._estimate_chunk_max_bytes(text)
        config = ProcessConfig(language=language, chunk_max_size=chunk_max_bytes)
        result = process(text, config)  # ty: ignore[invalid-argument-type]
        return result.chunks  # ty: ignore[invalid-return-type]

    def _create_chunks_from_code_chunks(
        self, code_chunks: list["CodeChunk"], offset: int = 0
    ) -> list[Chunk]:
        """Convert tree-sitter-language-pack CodeChunks into chonkie Chunks."""
        chunks = []
        for code_chunk in code_chunks:
            text = code_chunk.content
            token_count = self.tokenizer.count_tokens(text)
            chunks.append(
                Chunk(
                    text=text,
                    start_index=offset + code_chunk.start_byte,
                    end_index=offset + code_chunk.end_byte,
                    token_count=token_count,
                ),
            )
        return chunks

    def chunk(self, text: str) -> list[Chunk]:
        """Recursively chunks the code based on context from tree-sitter."""
        if not text.strip():
            logger.debug("Empty or whitespace-only code provided")
            return []

        logger.debug(f"Starting code chunking for text of length {len(text)}")

        if self.language == "auto":
            language = self._detect_language(text)
            logger.info(f"Auto-detected code language: {language}")
        else:
            language = self.language
            logger.debug(f"Using configured language: {language}")

        code_chunks = self._process_code(text, language)

        if not code_chunks:
            token_count = self.tokenizer.count_tokens(text)
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text.encode("utf-8")),
                    token_count=token_count,
                )
            ]

        chunks = self._create_chunks_from_code_chunks(code_chunks)
        logger.info(f"Created {len(chunks)} code chunks from parsed syntax tree")
        return chunks

    def _chunk_code_block(self, content: str, language: str | None) -> list[Chunk]:
        """Chunk a single code block, using the block's language hint when available."""
        if language and self.language == "auto":
            from tree_sitter_language_pack import has_language

            if has_language(language):
                try:
                    code_chunks = self._process_code(content, language)
                    if code_chunks:
                        return self._create_chunks_from_code_chunks(code_chunks)
                except Exception as e:
                    logger.debug(f"Language hint '{language}' failed, falling back to auto: {e}")
        return self.chunk(content)

    def chunk_document(self, document: Document) -> Document:
        """Chunk a document, with special handling for MarkdownDocument code blocks."""
        if isinstance(document, MarkdownDocument):
            if document.code:
                logger.debug(f"Processing MarkdownDocument with {len(document.code)} code blocks")
                for code_block in document.code:
                    if not code_block.content.strip():
                        continue

                    fenced_block = document.content[code_block.start_index : code_block.end_index]
                    first_newline = fenced_block.find("\n")
                    content_start = (
                        code_block.start_index + first_newline + 1
                        if first_newline != -1
                        else code_block.start_index
                    )

                    try:
                        chunks = self._chunk_code_block(code_block.content, code_block.language)
                    except Exception as e:
                        logger.warning(
                            f"CodeChunker failed for code block at index {code_block.start_index}: {e}"
                        )
                        chunks = [
                            Chunk(
                                text=code_block.content,
                                start_index=0,
                                end_index=len(code_block.content),
                                token_count=self.tokenizer.count_tokens(code_block.content),
                            )
                        ]
                    for chunk in chunks:
                        chunk.start_index = content_start + chunk.start_index
                        chunk.end_index = content_start + chunk.end_index
                    document.chunks.extend(chunks)
                document.chunks.sort(key=lambda x: x.start_index)
            BaseChunker._propagate_document_metadata(document)
            return document
        return super().chunk_document(document)

    def __repr__(self) -> str:
        """Return the string representation of the CodeChunker."""
        return (
            f"CodeChunker(tokenizer={self.tokenizer},"
            f"chunk_size={self.chunk_size},"
            f"language={self.language})"
        )
