"""Cloudflare Workers AI embeddings - thin wrapper around CatsuEmbeddings."""

import importlib.util as importutil
import os
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class CloudflareEmbeddings(BaseEmbeddings):
    """Cloudflare Workers AI embeddings via CatsuEmbeddings.

    Wraps CatsuEmbeddings with provider="cloudflare" for convenient access to
    Cloudflare Workers AI embedding models. Equivalent to:
    CatsuEmbeddings(model=..., provider="cloudflare")

    Args:
        model: Cloudflare model name (default: "@cf/baai/bge-base-en-v1.5").
        api_key: Cloudflare API key (or set CLOUDFLARE_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 30).
        batch_size: Number of texts per API call (default: 128).

    """

    DEFAULT_MODEL = "@cf/baai/bge-base-en-v1.5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 128,
    ):
        """Initialize Cloudflare embeddings.

        Args:
            model: Cloudflare Workers AI model name.
            api_key: Cloudflare API key (falls back to CLOUDFLARE_API_KEY env var).
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
        api_key = api_key or os.getenv("CLOUDFLARE_API_KEY")
        api_keys = {"cloudflare": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="cloudflare",
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
        """Return a string representation of the CloudflareEmbeddings instance."""
        return f"CloudflareEmbeddings(model={self.model})"
