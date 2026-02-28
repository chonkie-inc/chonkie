"""Google Gemini embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="gemini") directly.
"""

import importlib.util as importutil
import os
import warnings
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class GeminiEmbeddings(BaseEmbeddings):
    """Google Gemini embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="gemini") directly.

    Args:
        model: Gemini embedding model name (default: "gemini-embedding-001").
        api_key: Gemini API key (or set GEMINI_API_KEY env var).
        task_type: Ignored; kept for backward compatibility.
        max_retries: Maximum retry attempts (default: 3).
        batch_size: Number of texts per API call (default: 100).
        show_warnings: Ignored; kept for backward compatibility.

    """

    DEFAULT_MODEL = "gemini-embedding-001"

    AVAILABLE_MODELS = {
        "text-embedding-004": (768, 2048),
        "embedding-001": (768, 2048),
        "gemini-embedding-001": (3072, 8192),
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        max_retries: int = 3,
        batch_size: int = 100,
        show_warnings: bool = True,
    ):
        """Initialize Gemini embeddings wrapper.

        Args:
            model: Gemini embedding model name.
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
            task_type: Ignored; kept for backward compatibility.
            max_retries: Maximum retry attempts.
            batch_size: Number of texts per API call.
            show_warnings: Ignored; kept for backward compatibility.

        Raises:
            ImportError: If the catsu package is not installed.

        """
        super().__init__()

        if not self._is_available():
            raise ImportError(
                "One (or more) of the following packages is not available: catsu. "
                'Please install it via `pip install "chonkie[catsu]"`',
            )

        if task_type != "SEMANTIC_SIMILARITY":
            warnings.warn(
                "The `task_type` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if show_warnings is not True:
            warnings.warn(
                "The `show_warnings` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model = model if model else self.DEFAULT_MODEL
        self.task_type = task_type
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        api_keys = {"gemini": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=self.model,
            provider="gemini",
            api_keys=api_keys,
            max_retries=max_retries,
            batch_size=batch_size,
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self._catsu.embed(text)

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts using batched API calls."""
        return self._catsu.embed_batch(texts)

    async def aembed(self, text: str) -> np.ndarray:
        """Embed a single text string asynchronously."""
        return await self._catsu.aembed(text)

    async def aembed_batch(self, texts: list) -> list:
        """Embed multiple texts asynchronously using batched API calls."""
        return await self._catsu.aembed_batch(texts)

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
        """Representation of the GeminiEmbeddings instance."""
        return f"GeminiEmbeddings(model={self.model}, task_type={self.task_type})"
