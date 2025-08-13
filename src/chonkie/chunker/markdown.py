"""MarkdownChunker - Heading-aware Markdown splitter with optional cleaning.

This chunker implements a lightweight, dependency-free Markdown parsing strategy
inspired by the in-repo contributions in `my_contribution/custom_md_node_parser.py`
and `my_contribution/easy_splittonic.py` while using native Chonkie types.

Behavior:
- First split by strong separator lines (===== / -----)
- Then split by Markdown headings (#..######) with hierarchy-aware grouping
- Combine undersized sections into prior ones up to a maximum section size
- Enforce a target chunk_size via recursive, sentence-aware size splitting
- Optionally clean Markdown formatting (bold/italics/inline code/links/URLs)

Notes:
- No external llama_index dependencies; only native `Chunk` is returned
- Start/end indices refer to the original text span that produced the chunk
- If cleaning is enabled, chunk text can differ from the original substring
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from chonkie.types.base import Chunk

from .base import BaseChunker


class MarkdownChunker(BaseChunker):
    """Chunk Markdown text using heading-aware parsing and optional cleaning.

    Args:
        tokenizer_or_token_counter: Tokenizer or callable used to count tokens
        chunk_size: Desired maximum size (in characters) per output chunk
        chunk_overlap: Reserved for future use; not currently applied
        heading_level: Maximum heading level to consider (1..6); informational only
        min_characters_per_chunk: Minimum characters to consider a section meaningful
        max_characters_per_section: Maximum characters when combining undersized sections
        clean_text: If True, apply minimal markdown cleanup on chunk text
    """

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        heading_level: int = 3,
        min_characters_per_chunk: int = 50,
        max_characters_per_section: int = 4000,
        clean_text: bool = False,
    ) -> None:
        super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if heading_level < 1 or heading_level > 6:
            raise ValueError("heading_level must be between 1 and 6")
        if min_characters_per_chunk < 1:
            raise ValueError("min_characters_per_chunk must be >= 1")
        if max_characters_per_section < min_characters_per_chunk:
            raise ValueError("max_characters_per_section must be >= min_characters_per_chunk")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_level = heading_level
        self.min_characters_per_chunk = min_characters_per_chunk
        self.max_characters_per_section = max_characters_per_section
        self.clean_text = clean_text

    @classmethod
    def from_recipe(
        cls,
        name: Optional[str] = "default",
        lang: Optional[str] = "en",
        path: Optional[str] = None,
        tokenizer_or_token_counter: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        heading_level: int = 3,
        min_characters_per_chunk: int = 50,
        max_characters_per_section: int = 4000,
        clean_text: bool = False,
    ) -> "MarkdownChunker":
        # Recipes are not used for Markdown rules yet; this mirrors other chunkers' API
        return cls(
            tokenizer_or_token_counter=tokenizer_or_token_counter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            heading_level=heading_level,
            min_characters_per_chunk=min_characters_per_chunk,
            max_characters_per_section=max_characters_per_section,
            clean_text=clean_text,
        )

    # ------------------------
    # Public API
    # ------------------------
    def chunk(self, text: str) -> Sequence[Chunk]:  # type: ignore[override]
        if not text or not text.strip():
            return []

        # Stage 1: split by strong separator lines
        pattern_sections = self._split_on_separator_lines(text)

        # Stage 2: heading-aware split in each section
        ranges: List[Tuple[int, int]] = []
        for start, end in pattern_sections:
            ranges.extend(self._split_on_markdown_headings(text, start, end))

        # Stage 3: combine undersized sections
        ranges = self._combine_small_ranges(ranges)

        # Stage 4: enforce chunk_size via recursive, sentence-aware splitting
        final_ranges: List[Tuple[int, int]] = []
        for s, e in ranges:
            final_ranges.extend(self._split_range_recursively(text, s, e))

        # Build Chunk objects
        chunks: List[Chunk] = []
        for s, e in final_ranges:
            raw = text[s:e]
            if self.clean_text:
                out_text = self._clean_chunk_text(raw).strip()
            else:
                out_text = raw
            if not out_text:
                continue
            token_count = self.tokenizer.count_tokens(out_text)
            chunks.append(
                Chunk(
                    text=out_text,
                    start_index=s,
                    end_index=e,
                    token_count=token_count,
                )
            )
        return chunks

    # ------------------------
    # Splitting helpers (by indices)
    # ------------------------
    def _split_on_separator_lines(self, text: str) -> List[Tuple[int, int]]:
        """Return ranges split by strong separator lines of '=' or '-' characters.

        Separator is a line with 5 or more '=' or '-' characters.
        Ranges are returned as [(start, end), ...].
        """
        if not text:
            return []
        sep_pattern = re.compile(r"^\s*(?:={5,}|-{5,})\s*$", flags=re.MULTILINE)
        ranges: List[Tuple[int, int]] = []
        last_end = 0
        for m in sep_pattern.finditer(text):
            if m.start() > last_end:
                left = (last_end, m.start())
                if text[left[0]:left[1]].strip():
                    ranges.append(left)
            last_end = m.end()
        if last_end < len(text):
            tail = (last_end, len(text))
            if text[tail[0]:tail[1]].strip():
                ranges.append(tail)
        if not ranges:
            return [(0, len(text))]
        return ranges

    def _split_on_markdown_headings(self, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Split a section by Markdown headings with hierarchy-aware grouping.

        For level 1 headings (#), collect content until next level 1.
        For level >= 2, collect until next heading with level <= current.
        Also emits a leading prelude if content exists before the first heading.
        """
        section = text[start:end]
        heading_re = re.compile(r"^(#{1,6})\s+.+$", flags=re.MULTILINE)

        matches = [
            (start + m.start(), len(m.group(1)))
            for m in heading_re.finditer(section)
        ]

        if not matches:
            # No headings: return whole section (will be size-corrected later)
            return [(start, end)]

        ranges: List[Tuple[int, int]] = []
        # Prelude before first heading
        first_heading_start = matches[0][0]
        if first_heading_start > start and text[start:first_heading_start].strip():
            ranges.append((start, first_heading_start))

        # Add sentinel at end
        matches_with_end = matches + [(end, 0)]

        # Walk headings
        for idx in range(len(matches)):
            h_start, h_level = matches[idx]
            # Find boundary depending on level
            boundary = end
            for j in range(idx + 1, len(matches)):
                next_start, next_level = matches[j]
                if h_level == 1 and next_level == 1:
                    boundary = next_start
                    break
                if h_level >= 2 and next_level <= h_level:
                    boundary = next_start
                    break
            # Emit heading section
            if h_start < boundary and text[h_start:boundary].strip():
                ranges.append((h_start, boundary))

        return ranges

    def _combine_small_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Combine consecutive undersized ranges up to max_characters_per_section."""
        if not ranges:
            return []
        combined: List[Tuple[int, int]] = []
        cur_start, cur_end = ranges[0]
        for s, e in ranges[1:]:
            cur_len = cur_end - cur_start
            seg_len = e - s
            if seg_len < self.min_characters_per_chunk and cur_len + seg_len <= self.max_characters_per_section:
                # Merge forward
                cur_end = e
            else:
                combined.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        combined.append((cur_start, cur_end))
        return combined

    def _split_range_recursively(
        self, text: str, start: int, end: int, depth: int = 0, max_depth: int = 4
    ) -> List[Tuple[int, int]]:
        """Ensure a range does not exceed chunk_size by recursively splitting.

        Uses sentence-aware split points when possible and falls back to hard cut.
        """
        length = end - start
        if length <= self.chunk_size or depth >= max_depth:
            return [(start, end)]

        segment = text[start:end]
        local_split = self._find_split_point(segment, self.chunk_size)
        if local_split == -1:
            # Hard cut as last resort
            local_split = self.chunk_size
        left_end = start + local_split

        left_parts = self._split_range_recursively(text, start, left_end, depth + 1, max_depth)
        right_parts = self._split_range_recursively(text, left_end, end, depth + 1, max_depth)
        return left_parts + right_parts

    def _find_split_point(self, text: str, max_size: int) -> int:
        """Find a good split point up to max_size using sentence and paragraph cues."""
        if len(text) <= max_size:
            return -1
        window = text[:max_size]
        # Prefer sentence endings
        for ending in [". ", "! ", "? ", "\n\n"]:
            pos = window.rfind(ending)
            if pos >= int(max_size * 0.6):
                return pos + len(ending)
        # Next, try single newline as softer boundary
        pos_nl = window.rfind("\n")
        if pos_nl >= int(max_size * 0.5):
            return pos_nl + 1
        # Finally, try last space
        pos_sp = window.rfind(" ")
        if pos_sp >= int(max_size * 0.5):
            return pos_sp + 1
        return -1

    # ------------------------
    # Cleaning
    # ------------------------
    def _clean_chunk_text(self, chunk: str) -> str:
        """Minimal cleanup for Markdown content.

        - Remove leading question/search prompt lines (Search Results for/Question/Query)
        - Strip bold/italic/inline code markup
        - Convert [text](url) → text and drop bare URLs
        - Normalize whitespace and collapse excessive blank lines
        - Keep heading markers intact is not needed here since we are returning final chunks
        """
        if not chunk:
            return ""
        cleaned = chunk
        # Drop common prompt lines
        # cleaned = re.sub(r"^(?:Search Results for:|Question:|Query:).*$", "", cleaned, flags=re.MULTILINE)
        # Inline markdown
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        # Links and URLs
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        # Excess whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"^[ \t]+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def __repr__(self) -> str:
        return (
            f"MarkdownChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"heading_level={self.heading_level}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk}, "
            f"max_characters_per_section={self.max_characters_per_section}, "
            f"clean_text={self.clean_text})"
        )



