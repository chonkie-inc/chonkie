"""Chonkie: Recursive Chunker.

Splits text into smaller chunks recursively. Express chunking logic through RecursiveLevel objects.
"""

from bisect import bisect_left
from functools import lru_cache
from itertools import accumulate
from typing import Any, Callable, List, Optional, Tuple, Union

from chonkie.chunker.base import BaseChunker
from chonkie.types import (
    Chunk,
    RecursiveLevel,
    RecursiveRules,
)

# Import the unified split function
try:
    from .c_extensions.split import split_text
    SPLIT_AVAILABLE = True
except ImportError:
    SPLIT_AVAILABLE = False

# Import optimized merge functions
try:
    from .c_extensions.merge import _merge_splits as _merge_splits_cython
    MERGE_CYTHON_AVAILABLE = True
except ImportError:
    MERGE_CYTHON_AVAILABLE = False


class RecursiveChunker(BaseChunker):
    """Chunker that recursively splits text into smaller chunks, based on the provided RecursiveRules.

    Args:
        tokenizer_or_token_counter (Union[str, Callable, Any]): Tokenizer or token counter to use
        rules (list[RecursiveLevel]): List of RecursiveLevel objects defining chunking rules at a level.
        chunk_size (int): Maximum size of each chunk.
        min_characters_per_chunk (int): Minimum number of characters per chunk.

    """

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
        chunk_size: int = 2048,
        rules: RecursiveRules = RecursiveRules(),
        min_characters_per_chunk: int = 24,
    ) -> None:
        """Create a RecursiveChunker object.

        Args:
            tokenizer_or_token_counter (Union[str, Callable, Any]): Tokenizer or token counter to use
            rules (list[RecursiveLevel]): List of RecursiveLevel objects defining chunking rules at a level.
            chunk_size (int): Maximum size of each chunk.
            min_characters_per_chunk (int): Minimum number of characters per chunk.

        Raises:
            ValueError: If chunk_size <=0
            ValueError: If min_characters_per_chunk < 1
            ValueError: If recursive_rules is not a RecursiveRules object.

        """
        super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if min_characters_per_chunk <= 0:
            raise ValueError("min_characters_per_chunk must be greater than 0")
        if not isinstance(rules, RecursiveRules):
            raise ValueError("`rules` must be a RecursiveRules object.")

        # Initialize the internal values
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.rules = rules
        self.sep = "✄"
        self._CHARS_PER_TOKEN = 6.5

    @classmethod
    def from_recipe(cls,
                    name: Optional[str] = 'default',
                    lang: Optional[str] = 'en',
                    path: Optional[str] = None,
                    tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
                    chunk_size: int = 2048,
                    min_characters_per_chunk: int = 24,
                    ) -> "RecursiveChunker":
        """Create a RecursiveChunker object from a recipe.

        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). If the recipe is not there, you can create your own recipe and share it with the community!

        Args:
            name (Optional[str]): The name of the recipe.
            lang (Optional[str]): The language that the recursive chunker should support.
            path (Optional[str]): The path to the recipe.
            tokenizer_or_token_counter (Union[str, Callable, Any]): The tokenizer or token counter to use.
            chunk_size (int): The chunk size.
            min_characters_per_chunk (int): The minimum number of characters per chunk.

        Returns:
            RecursiveChunker: The RecursiveChunker object.

        Raises:
            ValueError: If the recipe is not found.

        """
        # Create a recursive rules object
        rules = RecursiveRules.from_recipe(name, lang, path)
        return cls(
            tokenizer_or_token_counter=tokenizer_or_token_counter,
            rules=rules,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    @lru_cache(maxsize=4096)
    def _estimate_token_count(self, text: str) -> int:
        # Always return the actual token count for accuracy
        # The estimate was only used as an optimization hint
        return self.tokenizer.count_tokens(text)

    def _split_text(self, text: str, recursive_level: RecursiveLevel) -> list[str]:
        """Split the text into chunks using the delimiters."""
        if SPLIT_AVAILABLE and recursive_level.delimiters:
            # Use optimized Cython split function for delimiter-based splitting only
            return list(split_text(
                text=text,
                delim=recursive_level.delimiters,
                include_delim=recursive_level.include_delim,
                min_characters_per_segment=self.min_characters_per_chunk,
                whitespace_mode=False,
                character_fallback=False
            ))
        else:
            # Fallback to original implementation
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

                # Merge short splits
                current = ""
                merged = []
                for split in splits:
                    if len(split) < self.min_characters_per_chunk:
                        current += split
                    elif current:
                        current += split
                        merged.append(current)
                        current = ""
                    else:
                        merged.append(split)

                    if len(current) >= self.min_characters_per_chunk:
                        merged.append(current)
                        current = ""

                if current:
                    merged.append(current)

                splits = merged
            else:
                # Encode, Split, and Decode
                encoded = self.tokenizer.encode(text)
                token_splits = [
                    encoded[i : i + self.chunk_size]
                    for i in range(0, len(encoded), self.chunk_size)
                ]
                splits = list(self.tokenizer.decode_batch(token_splits))

            return splits

    def _make_chunks(
        self,
        text: str,
        token_count: int,
        level: int,
        start_offset: int
    ) -> Chunk:
        """Create a Chunk object with indices based on the current offset.

        This method calculates the start and end indices of the chunk using the provided start_offset and the length of the text,
        avoiding a slower full-text search for efficiency.

        Args:
            text (str): The text content of the chunk.
            token_count (int): The number of tokens in the chunk.
            level (int): The recursion level of the chunk.
            start_offset (int): The starting offset in the original text.

        Returns:
            Chunk: A chunk object with calculated start and end indices, text, and token count.

        """
        return Chunk(
            text=text,
            start_index=start_offset,
            end_index=start_offset + len(text),
            token_count=token_count,
        )

    def _merge_splits(
        self,
        splits: list[str],
        token_counts: list[int],
        combine_whitespace: bool = False,
    ) -> Tuple[List[str], List[int]]:
        """Merge short splits into larger chunks.

        Uses optimized Cython implementation when available, with Python fallback.
        """
        if MERGE_CYTHON_AVAILABLE:
            # Use optimized Cython implementation
            return _merge_splits_cython(
                splits,
                token_counts,
                self.chunk_size,
                combine_whitespace
            )
        else:
            # Fallback to original Python implementation
            return self._merge_splits_fallback(splits, token_counts, combine_whitespace)

    def _merge_splits_fallback(
        self,
        splits: list[str],
        token_counts: list[int],
        combine_whitespace: bool = False,
    ) -> Tuple[List[str], List[int]]:
        """Original Python implementation of _merge_splits (fallback)."""
        if not splits or not token_counts:
            return [], []

        # If the number of splits and token counts does not match, raise an error
        if len(splits) != len(token_counts):
            raise ValueError(
                f"Number of splits {len(splits)} does not match number of token counts {len(token_counts)}"
            )

        # If all splits are larger than the chunk size, return them
        if all(counts > self.chunk_size for counts in token_counts):
            return splits, token_counts

        # If the splits are too short, merge them
        merged = []
        if combine_whitespace:
            # +1 for the whitespace
            cumulative_token_counts = list(
                accumulate([0] + token_counts, lambda x, y: x + y + 1)
            )
        else:
            cumulative_token_counts = list(accumulate([0] + token_counts))
        current_index = 0
        combined_token_counts = []

        while current_index < len(splits):
            current_token_count = cumulative_token_counts[current_index]
            required_token_count = current_token_count + self.chunk_size

            # Find the index to merge at
            index = min(
                bisect_left(
                    cumulative_token_counts,
                    required_token_count,
                    lo=current_index,
                )
                - 1,
                len(splits),
            )

            # If current_index == index,
            # we need to move to the next index
            if index == current_index:
                index += 1

            # Merge splits
            if combine_whitespace:
                merged.append(" ".join(splits[current_index:index]))
            else:
                merged.append("".join(splits[current_index:index]))

            # Adjust token count
            combined_token_counts.append(
                cumulative_token_counts[min(index, len(splits))] - current_token_count
            )
            current_index = index

        return merged, combined_token_counts

    def _recursive_chunk(
        self, text: str, level: int = 0, start_offset: int =0
    ) -> List[Chunk]:
        """Recursive helper for core chunking."""
        if not text:
            return []

        if level >= len(self.rules):
            return [
                self._make_chunks(text, self._estimate_token_count(text), level, start_offset)
            ]

        curr_rule = self.rules[level]
        if curr_rule is None:
            return [
                self._make_chunks(text, self._estimate_token_count(text), level, start_offset)
            ]

        splits = self._split_text(text, curr_rule)
        token_counts = [self._estimate_token_count(split) for split in splits]

        if curr_rule.delimiters is None and not curr_rule.whitespace:
            merged, combined_token_counts = splits, token_counts

        elif curr_rule.delimiters is None and curr_rule.whitespace:
            merged, combined_token_counts = self._merge_splits(
                splits, token_counts, combine_whitespace=True
            )
            # NOTE: This is a hack to fix the reconstruction issue when whitespace is used.
            # When whitespace is there, " ".join only adds space between words, not before the first word.
            # To make it combine back properly, all splits except the first one are prefixed with a space.
            merged = merged[:1] + [" " + text for (i, text) in enumerate(merged) if i != 0]

        else:
            merged, combined_token_counts = self._merge_splits(
                splits, token_counts, combine_whitespace=False
            )

        # Chunk long merged splits
        chunks: List[Chunk] = []
        current_offset = start_offset
        for split, token_count in zip(merged, combined_token_counts):
            if token_count > self.chunk_size:
                recursive_result = self._recursive_chunk(split, level + 1, current_offset)
                chunks.extend(recursive_result)
            else:
                chunks.append(self._make_chunks(split, token_count, level, current_offset))
            # Update the offset by the length of the processed split.
            current_offset += len(split)
        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        """Recursively chunk text.

        Args:
            text (str): Text to chunk.

        """
        return self._recursive_chunk(text=text, level=0, start_offset=0)

    def __repr__(self) -> str:
        """Get a string representation of the recursive chunker."""
        return (
            f"RecursiveChunker(tokenizer_or_token_counter={self.tokenizer},"
            f" rules={self.rules}, chunk_size={self.chunk_size}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk})"
        )
