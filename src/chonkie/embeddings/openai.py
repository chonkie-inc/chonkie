"""OpenAI embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="openai") directly.
"""

import importlib.util as importutil
import os
import warnings
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="openai") directly.

    Args:
        model: OpenAI embedding model name (default: "text-embedding-3-small").
        api_key: OpenAI API key (or set OPENAI_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 60).
        batch_size: Number of texts per API call (default: 128).
        tokenizer: Ignored; kept for backward compatibility.
        dimension: Ignored; kept for backward compatibility.
        max_tokens: Ignored; kept for backward compatibility.
        base_url: Ignored; kept for backward compatibility.
        organization: Ignored; kept for backward compatibility.
        **kwargs: Additional keyword arguments (ignored for compatibility).

    """

    DEFAULT_MODEL = "text-embedding-3-small"

    AVAILABLE_MODELS = {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8192},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8192},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8192},
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tokenizer: Optional[Any] = None,
        dimension: Optional[int] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        **kwargs: Any,
    ):
        """Initialize OpenAI embeddings wrapper.

        Args:
            model: OpenAI embedding model name.
            tokenizer: Ignored; kept for backward compatibility.
            dimension: Ignored; kept for backward compatibility.
            max_tokens: Ignored; kept for backward compatibility.
            base_url: Ignored; kept for backward compatibility.
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
            organization: Ignored; kept for backward compatibility.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            batch_size: Number of texts per API call.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            ImportError: If the catsu package is not installed.

        """
        super().__init__()

        if not self._is_available():
            raise ImportError(
                "One (or more) of the following packages is not available: catsu. "
                'Please install it via `pip install "chonkie[catsu]"`',
            )

        if tokenizer is not None:
            warnings.warn(
                "The `tokenizer` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if dimension is not None:
            warnings.warn(
                "The `dimension` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_tokens is not None:
            warnings.warn(
                "The `max_tokens` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if base_url is not None:
            warnings.warn(
                "The `base_url` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if organization is not None:
            warnings.warn(
                "The `organization` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        api_keys = {"openai": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="openai",
            api_keys=api_keys,
            max_retries=max_retries,
            timeout=round(timeout),
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
        """Representation of the OpenAIEmbeddings instance."""
        return f"OpenAIEmbeddings(model={self.model})"
