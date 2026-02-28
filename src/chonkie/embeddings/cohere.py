"""Cohere embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="cohere") directly.
"""

import importlib.util as importutil
import os
import warnings
from typing import Any, Optional

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class CohereEmbeddings(BaseEmbeddings):
    """Cohere embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="cohere") directly.

    Args:
        model: Cohere embedding model name (default: "embed-english-light-v3.0").
        api_key: Cohere API key (or set COHERE_API_KEY env var).
        client_name: Ignored; kept for backward compatibility.
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 60).
        batch_size: Number of texts per API call (default: 96).
        show_warnings: Ignored; kept for backward compatibility.

    """

    DEFAULT_MODEL = "embed-english-light-v3.0"

    AVAILABLE_MODELS = {
        "embed-english-v3.0": (True, 1024),
        "embed-multilingual-v3.0": (False, 1024),
        "embed-english-light-v3.0": (True, 384),
        "embed-multilingual-light-v3.0": (False, 384),
        "embed-english-v2.0": (False, 4096),
        "embed-english-light-v2.0": (False, 1024),
        "embed-multilingual-v2.0": (True, 768),
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 96,
        show_warnings: bool = True,
    ):
        """Initialize Cohere embeddings wrapper.

        Args:
            model: Cohere embedding model name.
            api_key: Cohere API key (falls back to COHERE_API_KEY env var).
            client_name: Ignored; kept for backward compatibility.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            batch_size: Number of texts per API call (max 96 for Cohere).
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

        if client_name is not None:
            warnings.warn(
                "The `client_name` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if show_warnings is not True:
            warnings.warn(
                "The `show_warnings` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model = model
        api_key = api_key or os.getenv("COHERE_API_KEY")
        api_keys = {"cohere": api_key} if api_key else None

        self._catsu = CatsuEmbeddings(
            model=model,
            provider="cohere",
            api_keys=api_keys,
            max_retries=max_retries,
            timeout=round(timeout),
            batch_size=min(batch_size, 96),
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        response = self._catsu.client.embed(
            model=self._catsu.model,
            input=text,
            provider=self._catsu.provider,
            input_type="document",
        )
        return response.to_numpy()[0]

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts using batched API calls."""
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), self._catsu._batch_size):
            batch = texts[i : i + self._catsu._batch_size]
            response = self._catsu.client.embed(
                model=self._catsu.model,
                input=batch,
                provider=self._catsu.provider,
                input_type="document",
            )
            arr = response.to_numpy()
            all_embeddings.extend([arr[j] for j in range(len(batch))])
        return all_embeddings

    async def aembed(self, text: str) -> np.ndarray:
        """Embed a single text string asynchronously."""
        response = await self._catsu.client.aembed(
            model=self._catsu.model,
            input=text,
            provider=self._catsu.provider,
            input_type="document",
        )
        return response.to_numpy()[0]

    async def aembed_batch(self, texts: list) -> list:
        """Embed multiple texts asynchronously using batched API calls."""
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), self._catsu._batch_size):
            batch = texts[i : i + self._catsu._batch_size]
            response = await self._catsu.client.aembed(
                model=self._catsu.model,
                input=batch,
                provider=self._catsu.provider,
                input_type="document",
            )
            arr = response.to_numpy()
            all_embeddings.extend([arr[j] for j in range(len(batch))])
        return all_embeddings

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
        """Return a string representation of the CohereEmbeddings object."""
        return f"CohereEmbeddings(model={self.model})"
