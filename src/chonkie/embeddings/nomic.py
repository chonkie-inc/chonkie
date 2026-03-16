"""Nomic embeddings - thin wrapper around CatsuEmbeddings."""

import importlib.util as importutil
import os
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class NomicEmbeddings(BaseEmbeddings):
    """Nomic embeddings via CatsuEmbeddings.

    Wraps CatsuEmbeddings with provider="nomic" for convenient access to
    Nomic's embedding models. Equivalent to:
    CatsuEmbeddings(model=..., provider="nomic")

    Args:
        model: Nomic embedding model name (default: "nomic-embed-text-v1.5").
        api_key: Nomic API key (or set NOMIC_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 30).
        batch_size: Number of texts per API call (default: 128).

    """

    DEFAULT_MODEL = "nomic-embed-text-v1.5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 128,
    ):
        """Initialize Nomic embeddings.

        Args:
            model: Nomic embedding model name.
            api_key: Nomic API key (falls back to NOMIC_API_KEY env var).
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            batch_size: Number of texts per API call.

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
        api_key = api_key or os.getenv("NOMIC_API_KEY")
        api_keys = {"nomic": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="nomic",
            api_keys=api_keys,
            max_retries=max_retries,
            timeout=timeout,
            batch_size=batch_size,
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
        """Return a string representation of the NomicEmbeddings instance."""
        return f"NomicEmbeddings(model={self.model})"
