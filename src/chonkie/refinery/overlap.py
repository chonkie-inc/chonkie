"""Refinery for adding overlap to chunks."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from chonkie.types import Document

from chonkie.logger import get_logger
from chonkie.pipeline import refinery
from chonkie.tokenizer import AutoTokenizer, TokenizerProtocol
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

logger = get_logger(__name__)


class OverlapRefinery:
    """Mixin that adds chunk overlap capabilities to any class.

    When inherited, provides chunk overlap logic for maintaining contextual
    continuity between adjacent chunks. Useful for tasks like question answering
    or summarization over long documents.

    Usage in chunkers::

        class TokenChunker(OverlapRefinery, BaseChunker):
            def __init__(self, chunk_size=512, chunk_overlap=50, ...):
                BaseChunker.__init__(self, ...)
                OverlapRefinery.__init__(self, chunk_overlap=chunk_overlap, ...)
    """

    def __init__(
        self,
        chunk_overlap: int = 0,
        overlap_context_size: Union[int, float] = 0.25,
        overlap_mode: Literal["token", "recursive"] = "token",
        overlap_method: Literal["suffix", "prefix"] = "suffix",
        overlap_merge: bool = True,
        overlap_inplace: bool = True,
        overlap_rules: Optional[RecursiveRules] = None,
        overlap_tokenizer: Union[str, TokenizerProtocol, None] = None,
        # Backward-compatible aliases
        context_size: Union[int, float, None] = None,
        mode: Union[str, None] = None,
        method: Union[str, None] = None,
        merge: Union[bool, None] = None,
        inplace: Union[bool, None] = None,
        rules: Optional[RecursiveRules] = None,
        tokenizer: Union[str, TokenizerProtocol, None] = None,
    ) -> None:
        """Initialize the overlap mixin.

        Args:
            chunk_overlap: Number of tokens to overlap between chunks. If 0, no overlap is applied.
            overlap_context_size: The size of the context to add to chunks.
                A float between 0 and 1 is treated as a fraction of chunk size;
                an integer is an absolute token count.
            overlap_mode: The mode for overlap calculation. Could be 'token' or 'recursive'.
            overlap_method: The method for overlap. Could be 'suffix' (append context from
                previous chunk) or 'prefix' (prepend context from next chunk).
            overlap_merge: Whether to merge the context with the chunk text.
            overlap_inplace: Whether to modify chunks in place or make copies.
            overlap_rules: Rules for recursive overlap. Defaults to RecursiveRules().
            overlap_tokenizer: Tokenizer to use for overlap calculation.
                Falls back to self.tokenizer if not provided.
            context_size: Deprecated. Use overlap_context_size.
            mode: Deprecated. Use overlap_mode.
            method: Deprecated. Use overlap_method.
            merge: Deprecated. Use overlap_merge.
            inplace: Deprecated. Use overlap_inplace.
            rules: Deprecated. Use overlap_rules.
            tokenizer: Deprecated. Use overlap_tokenizer.

        """
        # Resolve backward-compatible aliases
        overlap_context_size = context_size if context_size is not None else overlap_context_size
        _mode: str = mode if mode is not None else overlap_mode
        _method: str = method if method is not None else overlap_method
        overlap_merge = merge if merge is not None else overlap_merge
        overlap_inplace = inplace if inplace is not None else overlap_inplace
        overlap_rules = rules if rules is not None else overlap_rules
        overlap_tokenizer = tokenizer if tokenizer is not None else overlap_tokenizer

        # Backward compat: context_size > 0 (int or float) also enables chunk_overlap
        # If the old API is used (context_size=N where N > 0), set chunk_overlap to enable overlap
        if (
            chunk_overlap == 0
            and isinstance(overlap_context_size, (int, float))
            and overlap_context_size > 0
        ):
            # For integer context_size, use it directly; for float or when other params set, use 1
            if isinstance(overlap_context_size, int):
                chunk_overlap = overlap_context_size
            elif (
                mode is not None or method is not None or inplace is not None or merge is not None
            ):
                chunk_overlap = 1
            else:
                # Enable overlap for any positive numeric context_size
                chunk_overlap = 1

        self.chunk_overlap = chunk_overlap
        self._overlap_context_size = overlap_context_size
        self._overlap_mode = _mode
        self._overlap_method = _method
        self._overlap_merge = overlap_merge
        self._overlap_inplace = overlap_inplace
        self._overlap_rules = overlap_rules or RecursiveRules()
        self._overlap_tokenizer = overlap_tokenizer

        # Expose old-style attribute names for backward compatibility
        self.context_size = overlap_context_size
        self.mode = _mode
        self.method = _method
        self.merge = overlap_merge
        self.inplace = overlap_inplace
        self.rules = overlap_rules

        # If chunk_overlap is provided, use it as context_size when overlap is enabled
        if (
            chunk_overlap > 0
            and isinstance(overlap_context_size, (int, float))
            and not isinstance(overlap_context_size, str)
        ):
            # chunk_overlap takes precedence for integer values
            if isinstance(chunk_overlap, int) and chunk_overlap > 0:
                self._effective_context_size = chunk_overlap
            else:
                self._effective_context_size = overlap_context_size
        elif isinstance(overlap_context_size, int):
            self._effective_context_size = overlap_context_size
        else:
            self._effective_context_size = overlap_context_size

        self._overlap_enabled = chunk_overlap > 0

        # Initialize the tokenizer for overlap calculations
        if overlap_tokenizer is not None:
            self._overlap_tokenizer_obj = AutoTokenizer(overlap_tokenizer)
        else:
            self._overlap_tokenizer_obj = None

        self._overlap_sep = "✄"

        # Performance optimization: Set cache size for LRU caches
        self._overlap_cache_size = 8192

        # Create LRU cached methods
        self._get_overlap_tokens_cached = lru_cache(maxsize=self._overlap_cache_size)(
            self._get_overlap_tokens_impl
        )
        self._count_overlap_tokens_cached = lru_cache(maxsize=self._overlap_cache_size)(
            self._count_overlap_tokens_impl
        )

    # ---- Internal overlap methods ----

    def _get_overlap_tokens_impl(self, text: str) -> list:
        """Get tokens from text using overlap tokenizer."""
        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            return list(text)  # fallback to character-level
        return list(tokenizer.encode(text))

    def _count_overlap_tokens_impl(self, text: str) -> int:
        """Count tokens in text using overlap tokenizer."""
        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            return len(text)  # fallback to character count
        return len(tokenizer.encode(text))

    def clear_overlap_cache(self) -> None:
        """Clear the LRU caches for overlap operations."""
        if hasattr(self, "_get_overlap_tokens_cached"):
            self._get_overlap_tokens_cached.cache_clear()
        if hasattr(self, "_count_overlap_tokens_cached"):
            self._count_overlap_tokens_cached.cache_clear()

    def _get_effective_context_size(self, chunks: list) -> int:
        """Get the effective context size for a set of chunks.

        Args:
            chunks: The chunks to compute context size for.

        Returns:
            The effective context size in tokens.

        """
        if isinstance(self._overlap_context_size, float):
            max_tokens = max((chunk.token_count for chunk in chunks), default=0)
            return int(self._overlap_context_size * max_tokens) if max_tokens > 0 else 0
        return int(self._overlap_context_size)

    # ---- Overlap context computation ----

    def _overlap_prefix_token(self, chunk: Chunk, context_size: int) -> str:
        """Calculate token-based overlap context (prefix mode).

        Takes text from the chunk's end as context for the next chunk.

        Args:
            chunk: The source chunk.
            context_size: Number of tokens for the context window.

        Returns:
            The overlap context text.

        """
        if context_size <= 0:
            return ""

        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            # Character-level fallback
            char_size = max(context_size, 1)
            return chunk.text[-char_size:] if len(chunk.text) >= char_size else chunk.text

        tokens = self._get_overlap_tokens_cached(chunk.text)
        if context_size > len(tokens):
            logger.debug(
                "Context size greater than chunk size. "
                "The entire chunk will be returned as context."
            )
            return chunk.text
        return tokenizer.decode(tokens[-context_size:])

    def _overlap_suffix_token(self, chunk: Chunk, context_size: int) -> str:
        """Calculate token-based overlap context (suffix mode).

        Takes text from the chunk's start as context for the previous chunk.

        Args:
            chunk: The source chunk.
            context_size: Number of tokens for the context window.

        Returns:
            The overlap context text.

        """
        if context_size <= 0:
            return ""

        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            char_size = max(context_size, 1)
            return chunk.text[:char_size] if len(chunk.text) >= char_size else chunk.text

        tokens = self._get_overlap_tokens_cached(chunk.text)
        if context_size > len(tokens):
            logger.debug(
                "Context size greater than chunk size. "
                "The entire chunk will be returned as context."
            )
            return chunk.text
        return tokenizer.decode(tokens[:context_size])

    def _get_overlap_prefix_context(self, chunk: Chunk, context_size: int) -> str:
        """Get prefix overlap context from a chunk."""
        if self._overlap_mode == "token":
            return self._overlap_prefix_token(chunk, context_size)
        elif self._overlap_mode == "recursive":
            return self._overlap_prefix_recursive(chunk, context_size)
        raise ValueError(f"Mode must be one of: token, recursive. Got: {self._overlap_mode}")

    def _get_overlap_suffix_context(self, chunk: Chunk, context_size: int) -> str:
        """Get suffix overlap context from a chunk."""
        if self._overlap_mode == "token":
            return self._overlap_suffix_token(chunk, context_size)
        elif self._overlap_mode == "recursive":
            return self._overlap_suffix_recursive(chunk, context_size)
        raise ValueError(f"Mode must be one of: token, recursive. Got: {self._overlap_mode}")

    def _overlap_prefix_recursive(self, chunk: Chunk, context_size: int) -> str:
        """Calculate recursive prefix overlap context."""
        return self._recursive_overlap(chunk.text, 0, "prefix", context_size)

    def _overlap_suffix_recursive(self, chunk: Chunk, context_size: int) -> str:
        """Calculate recursive suffix overlap context."""
        return self._recursive_overlap(chunk.text, 0, "suffix", context_size)

    def _recursive_overlap(
        self,
        text: str,
        level: int,
        method: Literal["prefix", "suffix"],
        context_size: int,
    ) -> str:
        """Calculate recursive overlap context.

        Args:
            text: The text to calculate overlap from.
            level: The recursive level.
            method: 'prefix' or 'suffix'.
            context_size: The context size in tokens.

        Returns:
            The overlap context text.

        """
        if text == "":
            return ""

        if level >= len(self._overlap_rules.levels) if self._overlap_rules.levels else False:
            return text

        recursive_level = self._overlap_rules[level] if self._overlap_rules.levels else None
        if recursive_level is None:
            return text

        splits = self._split_overlap_text(text, recursive_level, context_size)

        if method == "prefix":
            splits = splits[::-1]

        token_counts = [self._count_overlap_tokens_cached(split) for split in splits]

        grouped_splits = self._group_overlap_splits(splits, token_counts, context_size)

        if not grouped_splits:
            return self._recursive_overlap(splits[0], level + 1, method, context_size)

        if method == "prefix":
            grouped_splits = grouped_splits[::-1]

        return "".join(grouped_splits)

    def _split_overlap_text(
        self,
        text: str,
        recursive_level: RecursiveLevel,
        context_size: int,
    ) -> list:
        """Split text using overlap recursive rules."""
        if recursive_level.whitespace:
            return text.split(" ")
        elif recursive_level.delimiters:
            if recursive_level.include_delim == "prev":
                for d in recursive_level.delimiters:
                    text = text.replace(d, d + self._overlap_sep)
            elif recursive_level.include_delim == "next":
                for d in recursive_level.delimiters:
                    text = text.replace(d, self._overlap_sep + d)
            else:
                for d in recursive_level.delimiters:
                    text = text.replace(d, self._overlap_sep)
            return [s for s in text.split(self._overlap_sep) if s != ""]
        else:
            tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
            if tokenizer is None:
                return [text]
            encoded = tokenizer.encode(text)
            token_splits = [
                encoded[i : i + context_size] for i in range(0, len(encoded), context_size)
            ]
            return list(tokenizer.decode_batch(token_splits))

    def _group_overlap_splits(
        self,
        splits: list,
        token_counts: list,
        context_size: int,
    ) -> list:
        """Group splits within context size."""
        group = []
        current_count = 0
        for count, split in zip(token_counts, splits):
            if current_count + count < context_size:
                group.append(split)
                current_count += count
            else:
                break
        return group

    # ---- Main overlap methods ----

    def _apply_overlap_prefix(self, chunks: list, context_size: int) -> list:
        """Apply prefix overlap to chunks (context from next chunk prepended)."""
        for i, chunk in enumerate(chunks[1:]):
            prev_chunk = chunks[i]

            # Per-chunk effective context size if using float
            if isinstance(self._overlap_context_size, float):
                effective_size = int(self._overlap_context_size * prev_chunk.token_count)
            else:
                effective_size = context_size

            context = self._get_overlap_prefix_context(prev_chunk, effective_size)
            setattr(chunk, "context", context)

            if self._overlap_merge:
                chunk.text = context + chunk.text
                chunk.token_count += self._count_overlap_tokens_cached(context)

        return chunks

    def _apply_overlap_suffix(self, chunks: list, context_size: int) -> list:
        """Apply suffix overlap to chunks (context from previous chunk appended)."""
        for i, chunk in enumerate(chunks[:-1]):
            next_chunk = chunks[i + 1]

            if isinstance(self._overlap_context_size, float):
                effective_size = int(self._overlap_context_size * next_chunk.token_count)
            else:
                effective_size = context_size

            context = self._get_overlap_suffix_context(next_chunk, effective_size)
            setattr(chunk, "context", context)

            if self._overlap_merge:
                chunk.text = chunk.text + context
                chunk.token_count += self._count_overlap_tokens_cached(context)

        return chunks

    def _apply_overlap_to_chunks(self, chunks: list) -> list:
        """Apply overlap to all chunks.

        This is the main method called by chunkers during chunk creation
        to apply overlap context to each chunk.

        Args:
            chunks: The list of chunks to apply overlap to.

        Returns:
            The chunks with overlap applied.

        """
        if not self._overlap_enabled or len(chunks) < 2:
            return chunks

        if self._overlap_inplace:
            working_chunks = chunks
        else:
            working_chunks = [chunk.copy() for chunk in chunks]

        context_size = self._get_effective_context_size(working_chunks)

        if self._overlap_method == "prefix":
            working_chunks = self._apply_overlap_prefix(working_chunks, context_size)
        elif self._overlap_method == "suffix":
            working_chunks = self._apply_overlap_suffix(working_chunks, context_size)
        else:
            raise ValueError(f"Method must be 'prefix' or 'suffix'. Got: {self._overlap_method}")

        return working_chunks

    # ---- Backward-compatible refine method ----

    def refine(self, chunks: list) -> list:
        """Refine chunks with overlap context.

        This method is kept for backward compatibility with the pipeline API
        and for use as a standalone refinery.

        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks with overlap applied.

        """
        logger.debug(
            f"Starting overlap refinement for {len(chunks)} chunks "
            f"with method={self._overlap_method}, mode={self._overlap_mode}"
        )
        if not chunks:
            return chunks

        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type.")

        return self._apply_overlap_to_chunks(chunks)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OverlapRefinery(chunk_overlap={self.chunk_overlap}, "
            f"overlap_context_size={self._overlap_context_size}, "
            f"overlap_mode={self._overlap_mode}, overlap_method={self._overlap_method}, "
            f"overlap_merge={self._overlap_merge}, overlap_inplace={self._overlap_inplace})"
        )


# Register the refinery alias for backward-compatible pipeline usage
@refinery("overlap")
class _OverlapRefineryRefinery(OverlapRefinery):
    """Wrapper class for backward-compatible pipeline usage.

    This class registers 'overlap' as a refinery type so that existing
    pipeline calls like .refine_with("overlap", ...) continue to work.
    """

    def __init__(self, **kwargs):
        # Pass all kwargs through
        # Convert chunk_overlap to the right params
        chunk_overlap = kwargs.pop("chunk_overlap", 0)
        context_size = kwargs.pop("context_size", kwargs.pop("overlap_context_size", 0.25))
        mode = kwargs.pop("mode", kwargs.pop("overlap_mode", "token"))
        method = kwargs.pop("method", kwargs.pop("overlap_method", "suffix"))
        merge = kwargs.pop("merge", kwargs.pop("overlap_merge", True))
        inplace = kwargs.pop("inplace", kwargs.pop("overlap_inplace", True))
        rules = kwargs.pop("rules", kwargs.pop("overlap_rules", RecursiveRules()))
        tokenizer = kwargs.pop("overlap_tokenizer", kwargs.pop("tokenizer", None))

        OverlapRefinery.__init__(
            self,
            chunk_overlap=chunk_overlap,
            overlap_context_size=context_size,
            overlap_mode=mode,
            overlap_method=method,
            overlap_merge=merge,
            overlap_inplace=inplace,
            overlap_rules=rules,
            overlap_tokenizer=tokenizer,
        )

    # Keep refine() working for pipeline usage
    def refine(self, chunks: list) -> list:
        return self._apply_overlap_to_chunks(chunks)

    def refine_document(self, document: "Document") -> "Document":
        """Refine all chunks in a Document with overlap context."""
        if not document.chunks:
            return document
        refined = self._apply_overlap_to_chunks(document.chunks)
        document.chunks = refined
        return document

    async def arefine_document(self, document: "Document") -> "Document":
        """Async version of refine_document."""
        return self.refine_document(document)
