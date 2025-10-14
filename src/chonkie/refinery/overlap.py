"""Refinery for adding overlap to chunks."""

import warnings
from functools import lru_cache
from typing import List, Literal, Union

from chonkie.logger import get_logger
from chonkie.pipeline import refinery
from chonkie.refinery.base import BaseRefinery
from chonkie.tokenizer import AutoTokenizer, TokenizerProtocol
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

logger = get_logger(__name__)

# TODO: Fix the way that float context size is handled.
# Currently, it just estimates the context size to token count
# but it should ideally handle it on a chunk by chunk basis.

# TODO: Add support for `justified` method which is the best of
# both prefix and suffix overlap.


@refinery("overlap")
class OverlapRefinery(BaseRefinery):
    """Refinery for adding overlap to chunks.

    Uses LRU caching (maxsize=8192) for tokenization operations to improve
    performance when processing similar text repeatedly. The cache can be
    monitored with cache_info() and cleared with clear_cache() if needed.
    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        context_size: Union[int, float] = 0.25,
        mode: Literal["token", "recursive"] = "token",
        method: Literal["suffix", "prefix"] = "suffix",
        rules: RecursiveRules = RecursiveRules(),
        merge: bool = True,
        inplace: bool = True,
    ) -> None:
        """Initialize the refinery.

        When a tokenizer is not provided, the refinery defaults to character-level
        overlap. Otherwise, the refinery will use the tokenizer to calculate the overlap.

        Args:
            tokenizer: The tokenizer to use. Defaults to "character".
            context_size: The size of the context to add to the chunks.
            mode: The mode to use for overlapping. Could be token or recursive.
            method: The method to use for the context. Could be suffix or prefix.
            rules: The rules to use for the recursive overlap. Defaults to RecursiveRules().
            merge: Whether to merge the context with the chunk. Defaults to True.
            inplace: Whether to modify the chunks in place or make a copy. Defaults to True.

        """
        # Check if the context size is a valid number
        if isinstance(context_size, float) and (context_size <= 0 or context_size > 1):
            raise ValueError("Context size must be a number between 0 and 1.")
        elif isinstance(context_size, int) and context_size <= 0:
            raise ValueError("Context size must be a positive integer.")
        if mode not in ["token", "recursive"]:
            raise ValueError("Mode must be one of: token, recursive.")
        if method not in ["suffix", "prefix"]:
            raise ValueError("Method must be one of: suffix, prefix.")
        if not isinstance(merge, bool):
            raise ValueError("Merge must be a boolean.")
        if not isinstance(inplace, bool):
            raise ValueError("Inplace must be a boolean.")

        # Initialize the refinery
        self.tokenizer = AutoTokenizer(tokenizer)
        self.context_size = context_size
        self.mode = mode
        self.method = method
        self.merge = merge
        self.inplace = inplace
        self.rules = rules
        self.sep = "✄"

        # Performance optimization: Set cache size for LRU caches
        self._cache_size = 8192

        # Create LRU cached methods
        self._get_tokens_cached = lru_cache(maxsize=self._cache_size)(
            self._get_tokens_impl
        )
        self._count_tokens_cached = lru_cache(maxsize=self._cache_size)(
            self._count_tokens_impl
        )

    def _get_tokens_impl(self, text: str) -> list:
        """Get tokens from text."""
        return list(self.tokenizer.encode(text))

    def _count_tokens_impl(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def clear_cache(self) -> None:
        """Clear the LRU caches to free memory."""
        if hasattr(self, "_get_tokens_cached"):
            self._get_tokens_cached.cache_clear()
        if hasattr(self, "_count_tokens_cached"):
            self._count_tokens_cached.cache_clear()

    def cache_info(self) -> dict:
        """Get cache information for monitoring."""
        info = {}
        if hasattr(self, "_get_tokens_cached"):
            info["tokens_cache"] = self._get_tokens_cached.cache_info()._asdict()
        if hasattr(self, "_count_tokens_cached"):
            info["count_cache"] = self._count_tokens_cached.cache_info()._asdict()
        return info

    def _split_text(
        self, text: str, recursive_level: RecursiveLevel, effective_context_size: int
    ) -> list[str]:
        """Split the text into chunks using the delimiters."""
        # At every delimiter, replace it with the sep
        if recursive_level.whitespace:
            splits = text.split(" ")
        elif recursive_level.delimiters:
            if recursive_level.include_delim == "prev":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, delimiter + self.sep)
            elif recursive_level.include_delim == "next":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep + delimiter)
            else:
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep)
            splits = [split for split in text.split(self.sep) if split != ""]
        else:
            # Encode, Split, and Decode
            encoded = self.tokenizer.encode(text)
            token_splits = [
                encoded[i : i + effective_context_size]
                for i in range(0, len(encoded), effective_context_size)
            ]
            splits = list(self.tokenizer.decode_batch(token_splits))

        # Some splits may not be meaningful yet.
        # This will be handled during chunk creation.
        return splits

    def _get_token_counts_cached(self, splits: List[str]) -> List[int]:
        """Get token counts with LRU caching for performance optimization."""
        return [self._count_tokens_cached(split) for split in splits]

    def _group_splits(
        self, splits: List[str], token_counts: List[int], effective_context_size: int
    ) -> List[str]:
        """Group the splits.

        Args:
            splits: The splits to merge.
            token_counts: The token counts of the splits.
            effective_context_size: The effective context size to use.

        Returns:
            The grouped splits.

        """
        group = []
        current_token_count = 0
        for token_count, split in zip(token_counts, splits):
            if current_token_count + token_count < effective_context_size:
                group.append(split)
                current_token_count += token_count
            else:
                break
        return group

    def _prefix_overlap_token(self, chunk: Chunk, effective_context_size: int) -> str:
        """Calculate token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk end, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: The chunk to calculate the overlap context for.
            effective_context_size: The effective context size to use.

        Returns:
            The overlap context.

        """
        # Performance optimization: Use LRU cached tokenization
        tokens = self._get_tokens_cached(chunk.text)

        if effective_context_size > len(tokens):
            warnings.warn(
                "Context size is greater than the chunk size. The entire chunk will be returned as the context."
            )
            return chunk.text
        else:
            return self.tokenizer.decode(tokens[-effective_context_size:])

    def _recursive_overlap(
        self,
        text: str,
        level: int,
        method: Literal["prefix", "suffix"],
        effective_context_size: int,
    ) -> str:
        """Calculate recursive overlap context.

        Args:
            text: The text to calculate the overlap context for.
            level: The recursive level to use.
            method: The method to use for the context.
            effective_context_size: The effective context size to use.

        Returns:
            The overlap context.

        """
        if text == "":
            return ""

        # Check if we've exceeded the available recursive levels
        if level >= len(self.rules):
            return text

        # Split the Chunk text based on the recursive rules
        recursive_level = self.rules[level]
        if recursive_level is None:
            return text
        splits = self._split_text(text, recursive_level, effective_context_size)

        if method == "prefix":
            splits = splits[::-1]

        # Performance optimization: Get token counts with caching
        token_counts = self._get_token_counts_cached(splits)

        # Group the splits
        grouped_splits = self._group_splits(
            splits, token_counts, effective_context_size
        )

        # If the grouped splits is empty, then we need to recursively split the first split
        if not grouped_splits:
            return self._recursive_overlap(
                splits[0], level + 1, method, effective_context_size
            )

        if method == "prefix":
            grouped_splits = grouped_splits[::-1]

        # Return the final context
        context = "".join(grouped_splits)
        return context

    def _prefix_overlap_recursive(
        self, chunk: Chunk, effective_context_size: int
    ) -> str:
        """Calculate recursive overlap context.

        Takes a larger window of text from the chunk end, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: The chunk to calculate the overlap context for.
            effective_context_size: The effective context size to use.

        Returns:
            The overlap context.

        """
        return self._recursive_overlap(chunk.text, 0, "prefix", effective_context_size)

    def _get_prefix_overlap_context(
        self, chunk: Chunk, effective_context_size: int
    ) -> str:
        """Get the prefix overlap context.

        Args:
            chunk: The chunk to get the prefix overlap context for.
            effective_context_size: The effective context size to use.

        """
        # Route to the appropriate method
        if self.mode == "token":
            return self._prefix_overlap_token(chunk, effective_context_size)
        elif self.mode == "recursive":
            return self._prefix_overlap_recursive(chunk, effective_context_size)
        else:
            raise ValueError("Mode must be one of: token, recursive.")

    def _refine_prefix(
        self, chunks: List[Chunk], effective_context_size: int
    ) -> List[Chunk]:
        """Refine the prefix of the chunk.

        Args:
            chunks: The chunks to refine.
            effective_context_size: The effective context size to use.

        Returns:
            The refined chunks.

        """
        # Iterate over the chunks till the second to last chunk
        for i, chunk in enumerate(chunks[1:]):
            # Get the previous chunk, since i starts from 0
            prev_chunk = chunks[i]

            # Calculate the overlap context
            context = self._get_prefix_overlap_context(
                prev_chunk, effective_context_size
            )

            # Set it as a part of the chunk
            setattr(chunk, "context", context)

            # Merge the context if merge is True
            if self.merge:
                chunk.text = context + chunk.text
                # Note: We don't adjust start_index/end_index when adding context
                # because they should represent the original document positions.
                # The context is additional information, not part of the original chunk position.

                # Performance optimization: Update the token count with LRU caching
                if self.tokenizer:
                    context_tokens = self._count_tokens_cached(context)
                    chunk.token_count += context_tokens

        return chunks

    def _suffix_overlap_token(self, chunk: Chunk, effective_context_size: int) -> str:
        """Calculate token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk start, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: The chunk to calculate the overlap context for.
            effective_context_size: The effective context size to use.

        Returns:
            The overlap context.

        """
        # Performance optimization: Use LRU cached tokenization
        tokens = self._get_tokens_cached(chunk.text)

        if effective_context_size > len(tokens):
            warnings.warn(
                "Context size is greater than the chunk size. The entire chunk will be returned as the context."
            )
            return chunk.text
        else:
            return self.tokenizer.decode(tokens[:effective_context_size])

    def _suffix_overlap_recursive(
        self, chunk: Chunk, effective_context_size: int
    ) -> str:
        """Calculate recursive overlap context.

        Takes a larger window of text from the chunk start, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: The chunk to calculate the overlap context for.
            effective_context_size: The effective context size to use.

        Returns:
            The overlap context.

        """
        return self._recursive_overlap(chunk.text, 0, "suffix", effective_context_size)

    def _get_suffix_overlap_context(
        self, chunk: Chunk, effective_context_size: int
    ) -> str:
        """Get the suffix overlap context.

        Args:
            chunk: The chunk to get the suffix overlap context for.
            effective_context_size: The effective context size to use.

        """
        # Route to the appropriate method
        if self.mode == "token":
            return self._suffix_overlap_token(chunk, effective_context_size)
        elif self.mode == "recursive":
            return self._suffix_overlap_recursive(chunk, effective_context_size)
        else:
            raise ValueError("Mode must be one of: token, recursive.")

    def _refine_suffix(
        self, chunks: List[Chunk], effective_context_size: int
    ) -> List[Chunk]:
        """Refine the suffix of the chunk.

        Args:
            chunks: The chunks to refine.
            effective_context_size: The effective context size to use.

        Returns:
            The refined chunks.

        """
        # Iterate over the chunks till the second to last chunk
        for i, chunk in enumerate(chunks[:-1]):
            # Get the previous chunk
            prev_chunk = chunks[i + 1]

            # Calculate the overlap context
            context = self._get_suffix_overlap_context(
                prev_chunk, effective_context_size
            )

            # Set it as a part of the chunk
            setattr(chunk, "context", context)

            # Merge the context if merge is True
            if self.merge:
                chunk.text = chunk.text + context
                # Note: We don't adjust start_index/end_index when adding context
                # because they should represent the original document positions.
                # The context is additional information, not part of the original chunk position.

                # Performance optimization: Update the token count with LRU caching
                if self.tokenizer:
                    context_tokens = self._count_tokens_cached(context)
                    chunk.token_count += context_tokens

        return chunks

    def _get_overlap_context_size(self, chunks: List[Chunk]) -> int:
        """Get the overlap context size.

        Args:
            chunks: The chunks to get the overlap context size for.

        """
        # Calculate context size for each call (float context size depends on chunk set)
        if isinstance(self.context_size, float):
            return int(self.context_size * max(chunk.token_count for chunk in chunks))
        else:
            return self.context_size

    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the chunks based on the overlap.

        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        logger.debug(f"Starting overlap refinery for {len(chunks)} chunks with method={self.method}, mode={self.mode}")
        # Check if the chunks are empty
        if not chunks:
            logger.debug("No chunks to refine, returning empty list")
            return chunks

        # Check if all the chunks are of the same type
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type.")

        # If inplace is False, make a copy of the chunks
        if not self.inplace:
            chunks = [chunk.copy() for chunk in chunks]

        # Get the effective context size for this chunk set (don't overwrite self.context_size)
        effective_context_size = self._get_overlap_context_size(chunks)

        # Refine the chunks based on the method
        if self.method == "prefix":
            refined_chunks = self._refine_prefix(chunks, effective_context_size)
        elif self.method == "suffix":
            refined_chunks = self._refine_suffix(chunks, effective_context_size)
        else:
            raise ValueError("Method must be one of: prefix, suffix.")

        logger.info(f"Overlap refinement complete: added context to {len(refined_chunks)} chunks")
        return refined_chunks

    def __repr__(self) -> str:
        """Return the string representation of the refinery."""
        return (
            f"OverlapRefinery(tokenizer={self.tokenizer}, "
            f"context_size={self.context_size}, "
            f"mode={self.mode}, method={self.method}, "
            f"merge={self.merge}, inplace={self.inplace})"
        )
