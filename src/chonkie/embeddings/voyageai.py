"""VoyageAI embeddings - backward-compatible wrapper around CatsuEmbeddings.

Note: Consider using CatsuEmbeddings(model=..., provider="voyageai") directly.
"""

import asyncio
import importlib.util as importutil
import os
import warnings
from typing import Any, Literal, Optional, Sequence

import numpy as np

from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings


class VoyageAIEmbeddings(BaseEmbeddings):
    """VoyageAI embeddings via CatsuEmbeddings.

    This is a backward-compatible wrapper around CatsuEmbeddings.
    Consider using CatsuEmbeddings(model=..., provider="voyageai") directly.

    Args:
        model: VoyageAI embedding model name (default: "voyage-3").
        api_key: VoyageAI API key (falls back to VOYAGE_API_KEY or VOYAGEAI_API_KEY env var).
        max_retries: Maximum retry attempts (default: 3).
        timeout: Request timeout in seconds (default: 60).
        output_dimension: Optional output dimension for contextual Voyage models.
        batch_size: Number of texts per API call (default: 128).
        truncation: Ignored for contextual Voyage models and delegated otherwise.

    """

    CONTEXTUALIZED_MODELS = {"voyage-context-3"}
    AVAILABLE_MODELS = {
        "voyage-3-large": ((1024, 256, 512, 2048), 32000),
        "voyage-3": ((1024,), 32000),
        "voyage-3-lite": ((512,), 32000),
        "voyage-code-3": ((1024, 256, 512, 2048), 32000),
        "voyage-context-3": ((1024, 256, 512, 2048), 120000),
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
            api_key: VoyageAI API key (falls back to VOYAGE_API_KEY or VOYAGEAI_API_KEY env var).
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            output_dimension: Optional output dimension for contextual Voyage models.
            batch_size: Number of texts per API call (max 128).
            truncation: Ignored for contextual Voyage models and delegated otherwise.

        Raises:
            ImportError: If the catsu package is not installed.

        """
        super().__init__()

        self.model = model
        self._is_contextualized = model in self.CONTEXTUALIZED_MODELS
        api_key = api_key or os.getenv("VOYAGE_API_KEY") or os.getenv("VOYAGEAI_API_KEY")

        if self._is_contextualized:
            if importutil.find_spec("voyageai") is None:
                raise ImportError(
                    "The voyageai package is not available. "
                    'Please install it via `pip install "chonkie[voyageai]"`',
                )
            import voyageai

            allowed_dims, _ = self.AVAILABLE_MODELS[model]
            if output_dimension is None:
                output_dimension = allowed_dims[0]
            elif output_dimension not in allowed_dims:
                raise ValueError(
                    f"Invalid output_dimension={output_dimension} for model={model}. "
                    f"Allowed: {sorted(allowed_dims)}",
                )
            if truncation is not True:
                warnings.warn(
                    "The `truncation` parameter is not supported for contextualized "
                    "Voyage embeddings and will be ignored.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            self.output_dimension = output_dimension
            self._dimension = output_dimension
            self._tokenizer: Any = None
            self._client = voyageai.Client(
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
            )
            return

        if not self._is_available():
            raise ImportError(
                "One (or more) of the following packages is not available: catsu. "
                'Please install it via `pip install "chonkie[catsu]"`',
            )

        if output_dimension is not None:
            warnings.warn(
                "The `output_dimension` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if truncation is not True:
            warnings.warn(
                "The `truncation` parameter is not supported in this version and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        api_keys = {"voyageai": api_key} if api_key else None
        self._catsu = CatsuEmbeddings(
            model=model,
            provider="voyageai",
            api_keys=api_keys,
            max_retries=max_retries,
            timeout=round(timeout),
            batch_size=min(batch_size, 128),
        )

    def _contextualized_embed(
        self,
        inputs: Sequence[Sequence[str]],
        input_type: Literal["query", "document"],
    ) -> list[np.ndarray]:
        """Embed grouped text inputs with Voyage contextualized embeddings."""
        groups = [list(group) for group in inputs if group]
        if not groups:
            return []

        response = self._client.contextualized_embed(
            inputs=groups,
            model=self.model,
            input_type=input_type,
            output_dimension=self.output_dimension,
        )
        return [
            np.array(embedding, dtype=np.float32)
            for result in response.results
            for embedding in result.embeddings
        ]

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self._is_contextualized:
            return self._contextualized_embed([[text]], input_type="query")[0]
        return self._catsu.embed(text)

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts using batched API calls."""
        if self._is_contextualized:
            return self.embed_documents([texts])
        return self._catsu.embed_batch(texts)

    def embed_documents(self, documents: Sequence[Sequence[str]]) -> list[np.ndarray]:
        """Embed document chunk groups while preserving document boundaries."""
        if self._is_contextualized:
            return self._contextualized_embed(documents, input_type="document")
        return self.embed_batch([text for document in documents for text in document])

    async def aembed(self, text: str) -> np.ndarray:
        """Embed a single text string asynchronously."""
        if self._is_contextualized:
            return await asyncio.to_thread(self.embed, text)
        return await self._catsu.aembed(text)

    async def aembed_batch(self, texts: list) -> list:
        """Embed multiple texts asynchronously using batched API calls."""
        if self._is_contextualized:
            return await asyncio.to_thread(self.embed_batch, texts)
        return await self._catsu.aembed_batch(texts)

    async def aembed_documents(self, documents: Sequence[Sequence[str]]) -> list[np.ndarray]:
        """Embed document chunk groups asynchronously."""
        return await asyncio.to_thread(self.embed_documents, documents)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._is_contextualized:
            return self._dimension
        return self._catsu.dimension

    def get_tokenizer(self) -> Any:
        """Return a tokenizer object for token counting."""
        if self._is_contextualized:
            if self._tokenizer is None:
                try:
                    from tokenizers import Tokenizer

                    self._tokenizer = Tokenizer.from_pretrained(f"voyageai/{self.model}")
                except Exception as e:
                    raise ValueError(
                        f"Failed to initialize tokenizer for model {self.model}: {e}"
                    ) from e
            return self._tokenizer
        return self._catsu.get_tokenizer()

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the catsu package is available."""
        return importutil.find_spec("catsu") is not None

    def __repr__(self) -> str:
        """Return a string representation of the VoyageAIEmbeddings object."""
        return f"VoyageAIEmbeddings(model={self.model})"
