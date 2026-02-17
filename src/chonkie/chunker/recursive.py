"""Chonkie: Recursive Chunker.

Splits text into smaller chunks recursively. Express chunking logic through RecursiveLevel objects.
"""

import re
from functools import lru_cache
from typing import Optional, Union

import chonkie_core

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import (
    Chunk,
    RecursiveLevel,
    RecursiveRules,
)

logger = get_logger(__name__)


@chunker("recursive")
class RecursiveChunker(BaseChunker):
    """Chunker that recursively splits text into smaller chunks, based on the provided RecursiveRules.

    Args:
        tokenizer: Tokenizer to use
        rules (list[RecursiveLevel]): List of RecursiveLevel objects defining chunking rules at a level.
        chunk_size (int): Maximum size of each chunk.
        min_characters_per_chunk (int): Minimum number of characters per chunk.

    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        rules: RecursiveRules = RecursiveRules(),
        min_characters_per_chunk: int = 24,
    ) -> None:
        """Create a RecursiveChunker object.

        Args:
            tokenizer: Tokenizer to use
            rules (list[RecursiveLevel]): List of RecursiveLevel objects defining chunking rules at a level.
            chunk_size (int): Maximum size of each chunk.
            min_characters_per_chunk (int): Minimum number of characters per chunk.

        Raises:
            ValueError: If chunk_size <=0
            ValueError: If min_characters_per_chunk < 1
            ValueError: If recursive_rules is not a RecursiveRules object.

        """
        super().__init__(tokenizer=tokenizer)

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
    def from_recipe(
        cls,
        name: Optional[str] = "default",
        lang: Optional[str] = "en",
        path: Optional[str] = None,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        min_characters_per_chunk: int = 24,
    ) -> "RecursiveChunker":
        """Create a RecursiveChunker object from a recipe.

        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). If the recipe is not there, you can create your own recipe and share it with the community!

        Args:
            name (Optional[str]): The name of the recipe.
            lang (Optional[str]): The language that the recursive chunker should support.
            path (Optional[str]): The path to the recipe.
            tokenizer: The tokenizer to use.
            chunk_size (int): The chunk size.
            min_characters_per_chunk (int): The minimum number of characters per chunk.

        Returns:
            RecursiveChunker: The RecursiveChunker object.

        Raises:
            ValueError: If the recipe is not found.

        """
        logger.info("Loading RecursiveChunker recipe", recipe_name=name, lang=lang)
        # Create a recursive rules object
        rules = RecursiveRules.from_recipe(name, lang, path)
        logger.debug(f"Recipe loaded successfully with {len(rules.levels or [])} levels")
        return cls(
            tokenizer=tokenizer,
            rules=rules,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    @lru_cache(maxsize=4096)
    def _estimate_token_count(self, text: str) -> int:
        # Always return the actual token count for accuracy
        # The estimate was only used as an optimization hint
        return self.tokenizer.count_tokens(text)

    def _validate_regex_safety(self, pattern: str) -> None:
        """Apply conservative safety checks for user-provided regex patterns."""
        if len(pattern) > 1024:
            raise ValueError("Regex pattern is too long (max 1024 characters)")

        # Backreferences can be expensive and are not required for chunk splitting.
        if re.search(r"\\[1-9]", pattern):
            raise ValueError("Regex backreferences are not supported for safety reasons")

        # Heuristic guard against catastrophic backtracking like (a+)+ or (.*)+.
        if re.search(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]", pattern):
            raise ValueError(
                "Regex pattern appears to use nested quantifiers that may cause excessive backtracking",
            )

    def _split_text_pattern(self, text: str, recursive_level: RecursiveLevel) -> list[str]:
        """Split text using regex pattern.

        Args:
            text: Text to split
            recursive_level: RecursiveLevel with pattern and pattern_mode set

        Returns:
            List of text splits

        Raises:
            ValueError: If regex pattern is invalid

        """
        if not recursive_level.pattern:
            return []

        try:
            # Compile the pattern to validate it
            pattern = recursive_level.pattern
            self._validate_regex_safety(pattern)
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{recursive_level.pattern}': {e}") from e

        include_mode = recursive_level.include_delim or "prev"
        pattern_mode = recursive_level.pattern_mode or "split"

        if pattern_mode == "extract":
            # Extract full matches to avoid tuple outputs with capturing groups.
            splits = [m.group(0) for m in compiled.finditer(text) if m.group(0)]
        else:
            # Split mode: reconstruct from match spans to avoid captured-group index shifts.
            splits = []
            cursor = 0
            carry_next_delim = ""
            for match in compiled.finditer(text):
                start, end = match.span()
                if start < cursor:
                    continue

                content = text[cursor:start]
                delim = text[start:end]

                if include_mode == "prev":
                    if content:
                        splits.append(content + delim)
                    elif splits:
                        splits[-1] += delim
                    elif delim:
                        splits.append(delim)
                elif include_mode == "next":
                    if content or carry_next_delim:
                        splits.append(carry_next_delim + content)
                    carry_next_delim = delim
                else:
                    if content:
                        splits.append(content)

                cursor = end

            tail = text[cursor:]
            if include_mode == "next":
                if tail or carry_next_delim:
                    splits.append(carry_next_delim + tail)
            elif tail:
                splits.append(tail)

        # Keep all splits; tiny segments are merged later to preserve full reconstruction.

        return splits

    def _split_text(self, text: str, recursive_level: RecursiveLevel) -> list[str]:
        """Split the text into chunks using the delimiters or pattern."""
        # Handle regex pattern first
        if recursive_level.pattern:
            return self._split_text_pattern(text, recursive_level)

        # Handle delimiters
        elif recursive_level.delimiters:
            include_mode = recursive_level.include_delim or "prev"
            text_bytes = text.encode("utf-8")

            # Check if we have multi-byte delimiters
            delimiters = recursive_level.delimiters
            if isinstance(delimiters, str):
                delimiters = [delimiters]

            has_multibyte = any(len(d) > 1 for d in delimiters)

            if has_multibyte:
                # Use split_pattern_offsets for multi-byte patterns
                patterns = [d.encode("utf-8") for d in delimiters]
                offsets = chonkie_core.split_pattern_offsets(
                    text_bytes,
                    patterns=patterns,
                    include_delim=include_mode,
                    min_chars=self.min_characters_per_chunk,
                )
            else:
                # Use faster split_offsets for single-byte delimiters
                delim_bytes = "".join(delimiters).encode("utf-8")
                offsets = chonkie_core.split_offsets(
                    text_bytes,
                    delimiters=delim_bytes,
                    include_delim=include_mode,
                    min_chars=self.min_characters_per_chunk,
                )

            # Convert offsets to strings
            splits = [text_bytes[start:end].decode("utf-8") for start, end in offsets]
            return [s for s in splits if s]  # Filter empty strings
        elif recursive_level.whitespace:
            # Split on whitespace using split_offsets (preserves spaces for reconstruction)
            text_bytes = text.encode("utf-8")
            include_mode = recursive_level.include_delim or "prev"
            offsets = chonkie_core.split_offsets(
                text_bytes,
                delimiters=b" ",
                include_delim=include_mode,
                min_chars=self.min_characters_per_chunk,
            )
            splits = [text_bytes[start:end].decode("utf-8") for start, end in offsets]
            return [s for s in splits if s]
        else:
            # Encode, Split, and Decode
            encoded = self.tokenizer.encode(text)
            token_splits = [
                encoded[i : i + self.chunk_size] for i in range(0, len(encoded), self.chunk_size)
            ]
            splits = list(self.tokenizer.decode_batch(token_splits))
            return splits

    def _make_chunks(self, text: str, token_count: int, level: int, start_offset: int) -> Chunk:
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
    ) -> tuple[list[str], list[int]]:
        """Merge short splits into larger chunks using chonkie-core."""
        if not splits or not token_counts:
            return [], []

        if len(splits) != len(token_counts):
            raise ValueError(
                f"Number of splits {len(splits)} does not match number of token counts {len(token_counts)}",
            )

        # If all splits are larger than the chunk size, return them
        if all(counts > self.chunk_size for counts in token_counts):
            return splits, token_counts

        # Use chonkie-core to merge (string joining done in Rust for performance)
        result = chonkie_core.merge_splits(splits, token_counts, self.chunk_size)
        return result.merged, result.token_counts

    def _recursive_chunk(self, text: str, level: int = 0, start_offset: int = 0) -> list[Chunk]:
        """Recursive helper for core chunking."""
        if not text:
            return []

        if level >= len(self.rules):
            return [self._make_chunks(text, self._estimate_token_count(text), level, start_offset)]

        curr_rule = self.rules[level]
        if curr_rule is None:
            return [self._make_chunks(text, self._estimate_token_count(text), level, start_offset)]

        splits = self._split_text(text, curr_rule)
        token_counts = [self._estimate_token_count(split) for split in splits]

        # Determine if we should merge splits
        # Merge for: pattern-based, delimiters, or whitespace splitting
        # No merge for: token-level fallback (when none of the above are set)
        should_merge = (
            curr_rule.pattern is not None
            or curr_rule.delimiters is not None
            or curr_rule.whitespace
        )

        if should_merge:
            merged, combined_token_counts = self._merge_splits(splits, token_counts)
        else:
            # Token-level fallback: no merging needed
            merged, combined_token_counts = splits, token_counts

        # Chunk long merged splits
        chunks: list[Chunk] = []
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

    def chunk(self, text: str) -> list[Chunk]:
        """Recursively chunk text.

        Args:
            text (str): Text to chunk.

        """
        logger.debug(f"Starting recursive chunking for text of length {len(text)}")
        chunks = self._recursive_chunk(text=text, level=0, start_offset=0)
        logger.info(f"Created {len(chunks)} chunks using recursive chunking")
        return chunks

    def __repr__(self) -> str:
        """Get a string representation of the recursive chunker."""
        return (
            f"RecursiveChunker(tokenizer={self.tokenizer},"
            f" rules={self.rules}, chunk_size={self.chunk_size}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk})"
        )
