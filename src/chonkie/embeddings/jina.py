"""Jina AI embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="jinaai") directly.
"""

import importlib.util as importutil
import os
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class JinaEmbeddings(BaseEmbeddings):
    """Jina AI embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="jinaai") directly.

    Args:
        model: Jina embedding model name (default: "jina-embeddings-v4").
        task: Ignored; kept for backward compatibility.
        batch_size: Number of texts per API call (default: 32).
        max_retries: Maximum retry attempts (default: 3).
        api_key: Jina API key (or set JINA_API_KEY env var).

    """

    AVAILABLE_MODELS = {
        "jina-embeddings-v3": 1024,
        "jina-embeddings-v2-base-en": 768,
        "jina-embeddings-v2-base-es": 768,
        "jina-embeddings-v2-base-de": 768,
        "jina-embeddings-v2-base-zh": 768,
        "jina-embeddings-v2-base-code": 768,
        "jina-embeddings-b-en-v1": 768,
        "jina-embeddings-v4": 2048,
    }

    DEFAULT_MODEL = "jina-embeddings-v4"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        task: str = "text-matching",
        batch_size: int = 32,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        """Initialize Jina embeddings wrapper.

        Args:
            model: Jina embedding model name.
            task: Ignored; kept for backward compatibility.
            batch_size: Number of texts per API call.
            max_retries: Maximum retry attempts.
            api_key: Jina API key (falls back to JINA_API_KEY env var).

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
        self.task = task
        api_key = api_key or os.getenv("JINA_API_KEY")
        api_keys = {"jina": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="jinaai",
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
        """Return a string representation of the JinaEmbeddings instance."""
        return f"JinaEmbeddings(model={self.model})"
