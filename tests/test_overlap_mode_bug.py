"""Test script to verify the overlap_mode=None bug (Issue report for v1.5.2).

This script checks whether passing overlap_mode=None to RecursiveChunker
causes an AttributeError (original bug) or is handled correctly.

Run with: pytest tests/test_overlap_mode_bug.py -v
"""

import pytest

from chonkie import RecursiveChunker


class TestOverlapModeBug:
    """Tests for the overlap_mode=None bug report."""

    def test_overlap_mode_none_raises_type_error(self) -> None:
        """overlap_mode parameter no longer exists, so passing it should raise TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            RecursiveChunker(overlap_mode=None)

    # def test_overlap_mode_string_raises_type_error(self) -> None:
    #     """Any value for overlap_mode should raise TypeError since the param was removed."""
    #     with pytest.raises(TypeError, match="unexpected keyword argument"):
    #         RecursiveChunker(overlap_mode="some_mode")
    #
    # def test_normal_usage_works(self) -> None:
    #     """Normal RecursiveChunker usage without overlap_mode should work fine."""
    #     chunker = RecursiveChunker(chunk_size=512)
    #     chunks = chunker("Sample text for chunking")
    #
    #     assert len(chunks) >= 1
    #     assert chunks[0].text == "Sample text for chunking"
    #
    # def test_chunker_with_long_text(self) -> None:
    #     """RecursiveChunker should produce multiple chunks for long text."""
    #     long_text = "This is a test sentence. " * 500
    #     chunker = RecursiveChunker(chunk_size=256)
    #     chunks = chunker(long_text)
    #
    #     assert len(chunks) > 1
    #     for chunk in chunks:
    #         assert chunk.text is not None
    #         assert len(chunk.text) > 0
