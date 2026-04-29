"""Refinery for adding overlap to chunks."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

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
    continuity between adjacent chunks.

    Can also be used standalone (backward-compatible with the old API)::

        # New style
        refinery = OverlapRefinery(chunk_overlap=5, overlap_tokenizer="character")

        # Old style (still works)
        refinery = OverlapRefinery(tokenizer="character", context_size=5)

    Usage in chunkers::

        class TokenChunker(OverlapRefinery, BaseChunker):
            def __init__(self, chunk_size=512, chunk_overlap=50, ...):
                BaseChunker.__init__(self, ...)
                OverlapRefinery.__init__(self, chunk_overlap=50, ...)
    """

    def __init__(
        self,
        chunk_overlap: Union[int, float] = 0,
        overlap_mode: Literal["token", "recursive"] = "token",
        overlap_method: Literal["suffix", "prefix"] = "suffix",
        overlap_rules: Optional[RecursiveRules] = None,
        overlap_tokenizer: Union[str, TokenizerProtocol, None] = None,
        # Backward-compatible aliases (old API)
        tokenizer: Union[str, TokenizerProtocol, None] = None,
        context_size: Union[int, float, None] = None,
        mode: Optional[Literal["token", "recursive"]] = None,
        method: Optional[Literal["suffix", "prefix"]] = None,
        rules: Optional[RecursiveRules] = None,
        merge: bool = True,
        inplace: bool = True,
    ) -> None:
        """Initialize the overlap mixin.

        Args:
            chunk_overlap: Overlap between chunks. An int is an absolute token count;
                a float between 0 and 1 is treated as a fraction of chunk size.
                Set to 0 to disable overlap.
            overlap_mode: Mode for overlap calculation: 'token' or 'recursive'.
            overlap_method: Method for overlap: 'suffix' (append context from next chunk)
                or 'prefix' (prepend context from previous chunk).
            overlap_rules: Rules for recursive overlap mode.
            overlap_tokenizer: Tokenizer for overlap calculations.
                Falls back to self.tokenizer if not provided.
            tokenizer: (Deprecated) Alias for overlap_tokenizer.
            context_size: (Deprecated) Alias for chunk_overlap.
            mode: (Deprecated) Alias for overlap_mode.
            method: (Deprecated) Alias for overlap_method.
            rules: (Deprecated) Alias for overlap_rules.
            merge: (Deprecated) Kept for backward compat, ignored.
            inplace: (Deprecated) Kept for backward compat, ignored.

        """
        # Resolve backward-compatible aliases
        if context_size is not None:
            chunk_overlap = context_size
        if mode is not None:
            overlap_mode = mode
        if method is not None:
            overlap_method = method
        if rules is not None:
            overlap_rules = rules
        if tokenizer is not None and overlap_tokenizer is None:
            overlap_tokenizer = tokenizer

        self.chunk_overlap = chunk_overlap
        self._overlap_mode = overlap_mode
        self._overlap_method = overlap_method
        self._overlap_rules = overlap_rules or RecursiveRules()

        self._overlap_enabled = (isinstance(chunk_overlap, int) and chunk_overlap > 0) or (
            isinstance(chunk_overlap, float) and chunk_overlap > 0.0
        )

        if overlap_tokenizer is not None:
            self._overlap_tokenizer_obj = AutoTokenizer(overlap_tokenizer)
        else:
            self._overlap_tokenizer_obj = None

        # Backward-compat attributes
        self._merge = merge
        self._inplace = inplace

        self._overlap_sep = "✄"
        self._overlap_cache_size = 8192

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
            return list(text)
        return list(tokenizer.encode(text))

    def _count_overlap_tokens_impl(self, text: str) -> int:
        """Count tokens in text using overlap tokenizer."""
        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            return len(text)
        return len(tokenizer.encode(text))

    def clear_overlap_cache(self) -> None:
        """Clear the LRU caches for overlap operations."""
        if hasattr(self, "_get_overlap_tokens_cached"):
            self._get_overlap_tokens_cached.cache_clear()
        if hasattr(self, "_count_overlap_tokens_cached"):
            self._count_overlap_tokens_cached.cache_clear()

    # ---- Backward-compatible properties and methods ----

    @property
    def context_size(self) -> Union[int, float]:
        """(Deprecated) Alias for chunk_overlap."""
        return self.chunk_overlap

    @context_size.setter
    def context_size(self, value: Union[int, float]) -> None:
        self.chunk_overlap = value

    @property
    def mode(self) -> str:
        """(Deprecated) Alias for _overlap_mode."""
        return self._overlap_mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._overlap_mode = value

    @property
    def method(self) -> str:
        """(Deprecated) Alias for _overlap_method."""
        return self._overlap_method

    @method.setter
    def method(self, value: str) -> None:
        self._overlap_method = value

    @property
    def merge(self) -> bool:
        """(Deprecated) Always True in new implementation."""
        return self._merge

    @merge.setter
    def merge(self, value: bool) -> None:
        self._merge = value

    @property
    def inplace(self) -> bool:
        """(Deprecated) Always True in new implementation."""
        return self._inplace

    @inplace.setter
    def inplace(self, value: bool) -> None:
        self._inplace = value

    def __getattr__(self, name: str):
        """Provide backward-compatible attribute access.

        Only called when normal attribute lookup fails, so it won't interfere
        with subclasses (like RecursiveChunker) that set self.rules directly,
        or BaseChunker that defines a tokenizer property.
        """
        if name == "rules":
            return self._overlap_rules
        if name == "tokenizer":
            return self._overlap_tokenizer_obj
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def clear_cache(self) -> None:
        """Clear the overlap cache (deprecated alias)."""
        self.clear_overlap_cache()

    def cache_info(self) -> dict:
        """Get cache information for monitoring."""
        info = {}
        if hasattr(self, "_get_overlap_tokens_cached"):
            info["tokens_cache"] = self._get_overlap_tokens_cached.cache_info()._asdict()
        if hasattr(self, "_count_overlap_tokens_cached"):
            info["count_cache"] = self._count_overlap_tokens_cached.cache_info()._asdict()
        return info

    def _get_effective_context_size(self, chunks: list) -> int:
        """Get the effective context size for a set of chunks."""
        if isinstance(self.chunk_overlap, float):
            max_tokens = max((chunk.token_count for chunk in chunks), default=0)
            return int(self.chunk_overlap * max_tokens) if max_tokens > 0 else 0
        return self.chunk_overlap

    # ---- Overlap context computation ----

    def _overlap_prefix_token(self, chunk: Chunk, context_size: int) -> str:
        """Take text from the chunk's end as context for the next chunk."""
        if context_size <= 0:
            return ""

        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            char_size = max(context_size, 1)
            return chunk.text[-char_size:] if len(chunk.text) >= char_size else chunk.text

        tokens = self._get_overlap_tokens_cached(chunk.text)
        if context_size > len(tokens):
            return chunk.text
        return tokenizer.decode(tokens[-context_size:])

    def _overlap_suffix_token(self, chunk: Chunk, context_size: int) -> str:
        """Take text from the chunk's start as context for the previous chunk."""
        if context_size <= 0:
            return ""

        tokenizer = self._overlap_tokenizer_obj or getattr(self, "tokenizer", None)
        if tokenizer is None:
            char_size = max(context_size, 1)
            return chunk.text[:char_size] if len(chunk.text) >= char_size else chunk.text

        tokens = self._get_overlap_tokens_cached(chunk.text)
        if context_size > len(tokens):
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
        """Calculate recursive overlap context."""
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
        """Apply prefix overlap to chunks (context from previous chunk prepended)."""
        for i, chunk in enumerate(chunks[1:]):
            prev_chunk = chunks[i]

            if isinstance(self.chunk_overlap, float):
                effective_size = int(self.chunk_overlap * prev_chunk.token_count)
            else:
                effective_size = context_size

            context = self._get_overlap_prefix_context(prev_chunk, effective_size)
            setattr(chunk, "context", context)
            chunk.text = context + chunk.text
            chunk.token_count += self._count_overlap_tokens_cached(context)

        return chunks

    def _apply_overlap_suffix(self, chunks: list, context_size: int) -> list:
        """Apply suffix overlap to chunks (context from next chunk appended)."""
        for i, chunk in enumerate(chunks[:-1]):
            next_chunk = chunks[i + 1]

            if isinstance(self.chunk_overlap, float):
                effective_size = int(self.chunk_overlap * next_chunk.token_count)
            else:
                effective_size = context_size

            context = self._get_overlap_suffix_context(next_chunk, effective_size)
            setattr(chunk, "context", context)
            chunk.text = chunk.text + context
            chunk.token_count += self._count_overlap_tokens_cached(context)

        return chunks

    def _apply_overlap_to_chunks(self, chunks: list) -> list:
        """Apply overlap to all chunks.

        Args:
            chunks: The list of chunks to apply overlap to.

        Returns:
            The chunks with overlap applied (modified in place).

        """
        if not self._overlap_enabled or len(chunks) < 2:
            return chunks

        context_size = self._get_effective_context_size(chunks)

        if self._overlap_method == "prefix":
            chunks = self._apply_overlap_prefix(chunks, context_size)
        elif self._overlap_method == "suffix":
            chunks = self._apply_overlap_suffix(chunks, context_size)
        else:
            raise ValueError(f"Method must be 'prefix' or 'suffix'. Got: {self._overlap_method}")

        return chunks

    # ---- Refine method for pipeline usage ----

    def refine(self, chunks: list) -> list:
        """Refine chunks with overlap context."""
        logger.debug(
            f"Starting overlap refinement for {len(chunks)} chunks "
            f"with method={self._overlap_method}, mode={self._overlap_mode}"
        )
        if not chunks:
            return chunks

        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type.")

        return self._apply_overlap_to_chunks(chunks)

    async def arefine(self, chunks: list) -> list:
        """Refine chunks asynchronously."""
        return await asyncio.to_thread(self.refine, chunks)

    def refine_document(self, document: "Document") -> "Document":
        """Refine all chunks in a Document with overlap context."""
        if not document.chunks:
            return document
        document.chunks = self.refine(document.chunks)
        return document

    async def arefine_document(self, document: "Document") -> "Document":
        """Async version of refine_document."""
        return self.refine_document(document)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the refinery."""
        chunks = args[0] if args else kwargs["chunks"]
        logger.info(f"Refining {len(chunks)} chunks with {self.__class__.__name__}")
        return self.refine(chunks)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OverlapRefinery(chunk_overlap={self.chunk_overlap}, "
            f"overlap_mode={self._overlap_mode}, overlap_method={self._overlap_method})"
        )


# Register the refinery alias for pipeline usage via .refine_with("overlap", ...)
@refinery("overlap")
class _OverlapRefineryRefinery(OverlapRefinery):
    """Wrapper class for pipeline registry.

    Accepts all kwargs and routes them to OverlapRefinery's backward-compatible init.
    """

    def __init__(self, **kwargs):
        OverlapRefinery.__init__(self, **kwargs)
