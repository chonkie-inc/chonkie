"""Fast chunker powered by memchunk."""

from typing import Any, Dict, List, Optional, Sequence

from chonkie.chunker.base import BaseChunker
from chonkie.pipeline import chunker
from chonkie.types import Chunk


@chunker("fast")
class FastChunker(BaseChunker):
    r"""Fast byte-based chunker using SIMD-accelerated boundary detection.

    Unlike other chonkie chunkers that use token counts, FastChunker uses
    byte size limits for maximum performance (~100+ GB/s throughput).

    This is a thin wrapper around memchunk's chunking functionality.

    Args:
        chunk_size: Target chunk size in bytes (default: 4096)
        delimiters: Delimiter characters for splitting (default: "\n.?")
        pattern: Multi-byte pattern to split on (overrides delimiters)
        prefix: Put delimiter at start of next chunk (default: False)
        consecutive: Split at START of consecutive runs (default: False)
        forward_fallback: Search forward if no delimiter in backward window

    Example:
        >>> chunker = FastChunker(chunk_size=1024)
        >>> chunks = chunker("Your long document here...")
        >>> for chunk in chunks:
        ...     print(chunk.text[:50])

    """

    def __init__(
        self,
        chunk_size: int = 4096,
        delimiters: str = "\n.?",
        pattern: Optional[str] = None,
        prefix: bool = False,
        consecutive: bool = False,
        forward_fallback: bool = False,
    ):
        """Initialize the FastChunker."""
        # Don't call super().__init__() - we don't need a tokenizer
        # But set required attributes for BaseChunker compatibility
        self._tokenizer = None  # type: ignore[assignment]
        self._use_multiprocessing = False

        self.chunk_size = chunk_size
        self.delimiters = delimiters
        self.pattern = pattern
        self.prefix = prefix
        self.consecutive = consecutive
        self.forward_fallback = forward_fallback
        # Lazy import memchunk.
        try:
            from memchunk import chunk_offsets
        except ImportError:
            raise ImportError(
                "memchunk is required for FastChunker. Install it with: pip install chonkie[fast]"
            )

        # Verify memchunk is available
        self._chunk_offsets = chunk_offsets

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return (
            f"FastChunker(chunk_size={self.chunk_size}, delimiters={self.delimiters!r}, "
            f"pattern={self.pattern!r}, prefix={self.prefix}, "
            f"consecutive={self.consecutive}, forward_fallback={self.forward_fallback})"
        )

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text at delimiter boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects

        """
        if not text:
            return []

        # Build kwargs for memchunk
        kwargs: Dict[str, Any] = {
            "size": self.chunk_size,
            "prefix": self.prefix,
            "consecutive": self.consecutive,
            "forward_fallback": self.forward_fallback,
        }

        if self.pattern:
            kwargs["pattern"] = self.pattern
        else:
            kwargs["delimiters"] = self.delimiters

        # Get chunk offsets from memchunk
        offsets = self._chunk_offsets(text, **kwargs)

        # Convert to Chunk objects
        return [
            Chunk(
                text=text[start:end],
                start_index=start,
                end_index=end,
                token_count=0,
            )
            for start, end in offsets
        ]

    def chunk_batch(self, texts: Sequence[str], show_progress: bool = True) -> List[List[Chunk]]:
        """Chunk a batch of texts.

        Args:
            texts: The texts to chunk.
            show_progress: Whether to show progress (ignored, always fast).

        Returns:
            A list of lists of Chunks.

        """
        return [self.chunk(text) for text in texts]
