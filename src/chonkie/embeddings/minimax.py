"""MiniMax embeddings via native API.

MiniMax provides the embo-01 embedding model with 1536 dimensions.
The API uses a non-OpenAI-compatible format with ``texts`` and ``type`` fields.
"""

import importlib.util as importutil
import os
from typing import Any, List, Optional

import httpx
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseEmbeddings

# MiniMax embedding API endpoint
_MINIMAX_EMBEDDINGS_URL = "https://api.minimax.io/v1/embeddings"


class MiniMaxEmbeddings(BaseEmbeddings):
    """MiniMax embeddings using the native MiniMax Embedding API.

    Uses the embo-01 model (1536 dimensions) via MiniMax's native endpoint,
    which expects ``texts`` and ``type`` fields instead of the OpenAI format.

    Args:
        model: MiniMax embedding model name (default: "embo-01").
        api_key: MiniMax API key (or set MINIMAX_API_KEY env var).
        embedding_type: Embedding type — "db" for storage, "query" for search
                        (default: "db").
        batch_size: Maximum texts per API call (default: 64).
        timeout: Request timeout in seconds (default: 30).

    Examples:
        >>> embeddings = MiniMaxEmbeddings(api_key="your-key")
        >>> vector = embeddings.embed("hello world")
        >>> vectors = embeddings.embed_batch(["text1", "text2"])

    """

    DEFAULT_MODEL = "embo-01"

    AVAILABLE_MODELS = {
        "embo-01": 1536,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        embedding_type: str = "db",
        batch_size: int = 64,
        timeout: float = 30.0,
    ):
        """Initialize MiniMax embeddings.

        Args:
            model: MiniMax embedding model name.
            api_key: MiniMax API key (falls back to MINIMAX_API_KEY env var).
            embedding_type: "db" for storage or "query" for search queries.
            batch_size: Maximum texts per API call.
            timeout: Request timeout in seconds.

        Raises:
            ImportError: If httpx is not installed.
            ValueError: If no API key is provided or embedding_type is invalid.

        """
        super().__init__()

        if not self._is_available():
            raise ImportError(
                "The httpx package is required for MiniMaxEmbeddings. "
                "Please install it via `pip install httpx`"
            )

        self.model = model
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMaxEmbeddings requires an API key. "
                "Either pass the `api_key` parameter or set the `MINIMAX_API_KEY` "
                "environment variable.",
            )

        if embedding_type not in ("db", "query"):
            raise ValueError("embedding_type must be 'db' or 'query'")

        self.embedding_type = embedding_type
        self._batch_size = batch_size
        self._timeout = timeout
        self._dimension = self.AVAILABLE_MODELS.get(model, 1536)

        self._client = httpx.Client(timeout=self._timeout)

    # -- core methods ---------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        retry=retry_if_exception_type((
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.TimeoutException,
        )),
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the MiniMax embedding API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            ValueError: If the API returns an error.

        """
        response = self._client.post(
            _MINIMAX_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "texts": texts,
                "type": self.embedding_type,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Check for API-level errors
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise ValueError(f"MiniMax API error: {base_resp.get('status_msg', 'unknown error')}")

        vectors = data.get("vectors")
        if vectors is None:
            raise ValueError("MiniMax API response missing 'vectors' field")
        return vectors

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        retry=retry_if_exception_type((
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.TimeoutException,
        )),
    )
    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        """Call the MiniMax embedding API asynchronously.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            ValueError: If the API returns an error.

        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                _MINIMAX_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "texts": texts,
                    "type": self.embedding_type,
                },
            )
            response.raise_for_status()
            data = response.json()

            base_resp = data.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                raise ValueError(
                    f"MiniMax API error: {base_resp.get('status_msg', 'unknown error')}"
                )

            vectors = data.get("vectors")
            if vectors is None:
                raise ValueError("MiniMax API response missing 'vectors' field")
            return vectors

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as a numpy array.

        """
        vectors = self._call_api([text])
        return np.array(vectors[0], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using batched API calls.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors as numpy arrays.

        """
        if not texts:
            return []

        results: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = self._call_api(batch)
            results.extend(np.array(v, dtype=np.float32) for v in vectors)
        return results

    async def aembed(self, text: str) -> np.ndarray:
        """Embed a single text string asynchronously.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as a numpy array.

        """
        vectors = await self._acall_api([text])
        return np.array(vectors[0], dtype=np.float32)

    async def aembed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts asynchronously using batched API calls.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors as numpy arrays.

        """
        if not texts:
            return []

        results: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = await self._acall_api(batch)
            results.extend(np.array(v, dtype=np.float32) for v in vectors)
        return results

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (1536 for embo-01)."""
        return self._dimension

    def get_tokenizer(self) -> Any:
        """Return a basic tokenizer for token counting.

        MiniMax does not expose a dedicated tokenizer API, so we use
        chonkie's built-in WordTokenizer as an approximation.

        Returns:
            WordTokenizer instance.

        """
        from chonkie.tokenizer import WordTokenizer

        return WordTokenizer()

    @classmethod
    def _is_available(cls) -> bool:
        """Check if httpx is available."""
        return importutil.find_spec("httpx") is not None

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"MiniMaxEmbeddings(model={self.model})"
