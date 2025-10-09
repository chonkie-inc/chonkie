"""Chonkie: Recursive Chunker.

Splits text into smaller chunks recursively. Express chunking logic through RecursiveLevel objects.
"""

from bisect import bisect_left
from functools import lru_cache
from itertools import accumulate
from typing import Any, Callable, List, Optional, Tuple, Union

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

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
    """Chunker that recursively splits text into smaller chunks, based on the provided RecursiveRules."""
    
    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
        chunk_size: int = 2048,
        rules: RecursiveRules = RecursiveRules(),
        min_characters_per_chunk: int = 24,
    ) -> None:
        super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if min_characters_per_chunk <= 0:
            raise ValueError("min_characters_per_chunk must be greater than 0")
        if not isinstance(rules, RecursiveRules):
            raise ValueError("`rules` must be a RecursiveRules object.")

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
        rules = RecursiveRules.from_recipe(name, lang, path)
        return cls(
            tokenizer_or_token_counter=tokenizer_or_token_counter,
            rules=rules,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    @lru_cache(maxsize=4096)
    def _estimate_token_count(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def _split_text(self, text: str, recursive_level: RecursiveLevel) -> list[str]:
        if SPLIT_AVAILABLE and recursive_level.delimiters:
            return list(split_text(
                text=text,
                delim=recursive_level.delimiters,
                include_delim=recursive_level.include_delim,
                min_characters_per_segment=self.min_characters_per_chunk,
                whitespace_mode=False,
                character_fallback=False
            ))
        else:
            # Original Python fallback
            if recursive_level.whitespace:
                splits = text.split(" ")
            elif recursive_level.delimiters:
                sep = self.sep
                if recursive_level.include_delim == "prev":
                    for d in recursive_level.delimiters:
                        text = text.replace(d, d + sep)
                elif recursive_level.include_delim == "next":
                    for d in recursive_level.delimiters:
                        text = text.replace(d, sep + d)
                else:
                    for d in recursive_level.delimiters:
                        text = text.replace(d, sep)
                splits = [s for s in text.split(sep) if s != ""]
                # Merge short splits
                current, merged = "", []
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
                # Encode, split, decode
                encoded = self.tokenizer.encode(text)
                token_splits = [encoded[i:i+self.chunk_size] for i in range(0, len(encoded), self.chunk_size)]
                splits = list(self.tokenizer.decode_batch(token_splits))
            return splits

    def _make_chunks(self, text: str, token_count: int, level: int, start_offset: int) -> Chunk:
        return Chunk(text=text, start_index=start_offset, end_index=start_offset+len(text), token_count=token_count)

    def _merge_splits(self, splits: list[str], token_counts: list[int], combine_whitespace: bool=False) -> Tuple[List[str], List[int]]:
        if MERGE_CYTHON_AVAILABLE:
            return _merge_splits_cython(splits, token_counts, self.chunk_size, combine_whitespace)
        else:
            return self._merge_splits_fallback(splits, token_counts, combine_whitespace)

    def _merge_splits_fallback(self, splits: list[str], token_counts: list[int], combine_whitespace: bool=False) -> Tuple[List[str], List[int]]:
        if not splits or not token_counts:
            return [], []
        if len(splits) != len(token_counts):
            raise ValueError("Number of splits and token counts must match")
        if all(tc > self.chunk_size for tc in token_counts):
            return splits, token_counts

        merged, combined_token_counts = [], []
        current_index = 0
        cumulative = list(accumulate([0]+token_counts))
        while current_index < len(splits):
            required = cumulative[current_index] + self.chunk_size
            index = min(bisect_left(cumulative, required, lo=current_index)-1, len(splits))
            if index == current_index: index += 1
            merged.append(" ".join(splits[current_index:index]) if combine_whitespace else "".join(splits[current_index:index]))
            combined_token_counts.append(cumulative[min(index, len(splits))]-cumulative[current_index])
            current_index = index
        return merged, combined_token_counts

    def _recursive_chunk(self, text: str, level: int=0, start_offset: int=0) -> List[Chunk]:
        if not text: return []
        if level >= len(self.rules): return [self._make_chunks(text, self._estimate_token_count(text), level, start_offset)]
        curr_rule = self.rules[level]
        if curr_rule is None: return [self._make_chunks(text, self._estimate_token_count(text), level, start_offset)]
        splits = self._split_text(text, curr_rule)
        token_counts = [self._estimate_token_count(s) for s in splits]
        if curr_rule.delimiters is None and not curr_rule.whitespace:
            merged, combined_token_counts = splits, token_counts
        elif curr_rule.delimiters is None and curr_rule.whitespace:
            merged, combined_token_counts = self._merge_splits(splits, token_counts, combine_whitespace=True)
            merged = merged[:1] + [" " + text for i, text in enumerate(merged) if i != 0]
        else:
            merged, combined_token_counts = self._merge_splits(splits, token_counts, combine_whitespace=False)
        chunks, offset = [], start_offset
        for split, token_count in zip(merged, combined_token_counts):
            if token_count > self.chunk_size:
                chunks.extend(self._recursive_chunk(split, level+1, offset))
            else:
                chunks.append(self._make_chunks(split, token_count, level, offset))
            offset += len(split)
        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        return self._recursive_chunk(text)

    def __repr__(self) -> str:
        return f"RecursiveChunker(tokenizer_or_token_counter={self.tokenizer}, rules={self.rules}, chunk_size={self.chunk_size}, min_characters_per_chunk={self.min_characters_per_chunk})"


# =============================
# Optional test block
# =============================
if __name__ == "__main__":
    rc = RecursiveChunker()
    sample_text = "This is a test sentence. Let's see how it splits into smaller chunks recursively!"
    chunks = rc.chunk(sample_text)
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c.text} (Tokens: {c.token_count})")
