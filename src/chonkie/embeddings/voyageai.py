"""VoyageAI embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="voyageai") directly.
"""

import importlib.util as importutil
import os
from typing import Any, Literal, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class VoyageAIEmbeddings(BaseEmbeddings):
    """VoyageAI embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="voyageai") directly.

    Args:
        model: VoyageAI embedding model name (default: "voyage-3").
        api_key: VoyageAI API key (or set VOYAGE_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 60).
        output_dimension: Ignored; kept for backward compatibility.
        batch_size: Number of texts per API call (default: 128).
        truncation: Ignored; kept for backward compatibility.

    """

    AVAILABLE_MODELS = {
        "voyage-3-large": ((1024, 256, 512, 2048), 32000),
        "voyage-3": ((1024,), 32000),
        "voyage-3-lite": ((512,), 32000),
        "voyage-code-3": ((1024, 256, 512, 2048), 32000),
        "voyage-finance-2": ((1024,), 32000),
        "voyage-law-2": ((1024,), 16000),
        "voyage-code-2": ((1536,), 16000),
    }

    DEFAULT_MODEL = "voyage-3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        output_dimension: Optional[Literal[256, 512, 1024, 2048]] = None,
        batch_size: int = 128,
        truncation: bool = True,
    ):
        """Initialize VoyageAI embeddings wrapper.

        Args:
            model: VoyageAI embedding model name.
            api_key: VoyageAI API key (falls back to VOYAGE_API_KEY env var).
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            output_dimension: Ignored; kept for backward compatibility.
            batch_size: Number of texts per API call (max 128).
            truncation: Ignored; kept for backward compatibility.

        Raises:
            ImportError: If the catsu package is not installed.

        """
        super().__init__()

        if not self._is_available():
            raise ImportError(
                "One (or more) of the following packages is not available: catsu. "
                'Please install it via `pip install "chonkie[catsu]"`',
            )

        self.model = model
        api_key = api_key or os.getenv("VOYAGE_API_KEY") or os.getenv("VOYAGEAI_API_KEY")
        api_keys = {"voyageai": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="voyageai",
            api_keys=api_keys,
            max_retries=max_retries,
            timeout=int(timeout),
            batch_size=min(batch_size, 128),
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self._catsu.embed(text)

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts using batched API calls."""
        return self._catsu.embed_batch(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._catsu.dimension

    def get_tokenizer(self) -> Any:
        """Return a tokenizer object for token counting."""
        return self._catsu.get_tokenizer()

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the catsu package is available."""
        return importutil.find_spec("catsu") is not None

    def __repr__(self) -> str:
        """Return a string representation of the VoyageAIEmbeddings object."""
        return f"VoyageAIEmbeddings(model={self.model})"
