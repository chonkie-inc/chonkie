"""Together AI embeddings - thin wrapper around CatsuEmbeddings."""

import importlib.util as importutil
import os
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class TogetherEmbeddings(BaseEmbeddings):
    """Together AI embeddings via CatsuEmbeddings.

    Wraps CatsuEmbeddings with provider="together" for convenient access to
    Together AI's embedding models. Equivalent to:
    CatsuEmbeddings(model=..., provider="together")

    Args:
        model: Together embedding model name
               (default: "togethercomputer/m2-bert-80M-8k-retrieval").
        api_key: Together API key (or set TOGETHER_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 30).
        batch_size: Number of texts per API call (default: 128).

    """

    DEFAULT_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 128,
    ):
        """Initialize Together embeddings.

        Args:
            model: Together embedding model name.
            api_key: Together API key (falls back to TOGETHER_API_KEY env var).
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
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        api_keys = {"together": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="together",
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
        """Return a string representation of the TogetherEmbeddings instance."""
        return f"TogetherEmbeddings(model={self.model})"
