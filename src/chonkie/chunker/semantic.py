"""SemanticChunker with advanced peak detection and window embedding calculation.

Uses Savitzky-Golay filtering for smoother boundary detection.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    import numpy as np

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk, Sentence
from chonkie.utils import Hubbie
from .base import BaseChunker

# Optional optimized C split
try:
    from .c_extensions.split import split_text
    SPLIT_AVAILABLE = True
except ImportError:
    SPLIT_AVAILABLE = False

# Optional optimized C Savitzky-Golay functions
try:
    from .c_extensions.savgol import (
        filter_split_indices,
        find_local_minima_interpolated,
        windowed_cross_similarity,
    )
    SAVGOL_AVAILABLE = True
except ImportError:
    SAVGOL_AVAILABLE = False


class SemanticChunker(BaseChunker):
    """SemanticChunker uses peak detection and window embeddings for chunking."""

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
        **kwargs: Dict[str, Any],
    ) -> None:
        # Parameter validation
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if similarity_window <= 0:
            raise ValueError("similarity_window must be positive")
        if min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be positive")
        if skip_window < 0:
            raise ValueError("skip_window must be non-negative")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if type(delim) not in [str, list]:
            raise ValueError("delim must be a string or list of strings")
        if filter_window <= 0:
            raise ValueError("filter_window must be positive")
        if not (0 <= filter_polyorder < filter_window):
            raise ValueError("filter_polyorder must be >=0 and < filter_window")
        if not 0 < filter_tolerance < 1:
            raise ValueError("filter_tolerance must be between 0 and 1")

        # Load embedding mo
