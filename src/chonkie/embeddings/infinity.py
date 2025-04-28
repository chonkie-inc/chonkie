import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util as importutil
import aiohttp
import requests
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
from .base import BaseEmbeddings

if TYPE_CHECKING:
    import numpy as np


class InfinityEmbeddings(BaseEmbeddings):
    """Infinity embeddings implementation using their API."""

    AVAILABLE_MODELS: Dict[str, int] = {
        "jinaai/jina-clip-v1": 768,
        "michaelfeil/bge-small-en-v1.5": 384,
        "mixedbread-ai/mxbai-rerank-xsmall-v1": 384,
        "philschmid/tiny-bert-sst2-distilled": 384,
    }
    
    DEFAULT_MODEL: str = "michaelfeil/bge-small-en-v1.5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        batch_size: int = 32,
        timeout: float = 60.0,
        infinity_api_url: str = "https://infinity.modal.michaelfeil.eu", # default url can be http://localhost:7997
    ):
        """
        Initialize the InfinityEmbeddings.

        Args:
            model: Name of the embedding model to use (must be in AVAILABLE_MODELS).
            batch_size: Maximum number of texts sent in a single request (max 32).
            timeout: HTTP request timeout, in seconds.
            infinity_api_url: Base URL for the Infinity API endpoint.

        Raises:
            ValueError: If the model is unsupported or the URL is invalid.
            ImportError: If numpy is not installed.
        """
        super().__init__()
        self._import_dependencies()

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not supported. "
                             f"Choose from {list(self.AVAILABLE_MODELS)}")
        if not infinity_api_url or len(infinity_api_url) < 5:
            raise ValueError("`infinity_api_url` must be a valid URL.")

        self.model: str = model
        self.timeout: float = timeout
        self._dimension: int = self.AVAILABLE_MODELS[model]
        self._batch_size: int = min(batch_size,32) # batch size assumed to max 32
        self.infinity_api_url: str = infinity_api_url.rstrip("/")
        self._session = requests.Session()

    def _prepare_batches(
        self,
        texts: List[str],
        key: Callable[[str], Any] = len,
    ) -> Tuple[List[List[str]], Callable[[List[Any]], List[Any]]]:
        """
        Sort `texts` by descending `key` (e.g. length), split into batches of size
        `self._batch_size`, and return (batches, restore_fn) where restore_fn
        reorders a flat embedding list back to original order.
        """
        if not texts:
            return [], lambda _: []

        # Compute sort order and inverse
        sort_keys = [-key(t) for t in texts]
        sorted_idx = np.argsort(sort_keys)
        sorted_texts = [texts[i] for i in sorted_idx]
        inverse_idx = np.argsort(sorted_idx)

        def restore(flat_embs: List[Any]) -> List[Any]:
            return [flat_embs[i] for i in inverse_idx]

        # Chunk into batches
        batches: List[List[str]] = [
            sorted_texts[i : i + self._batch_size]
            for i in range(0, len(sorted_texts), self._batch_size)
        ]

        return batches, restore
    

    def _request_kwargs(self, texts: List[str]) -> Dict[str, Any]:
        return {
            "url": f"{self.infinity_api_url}/embeddings",
            "headers": {"Accept": "application/json", "Content-Type": "application/json"},
            "json": {"input": texts, "model": self.model},
            "timeout": self.timeout,
        }

    def _sync_request(self, batch_texts: List[str]) -> List["np.ndarray"]:
        """Send a synchronous request to embed a batch of texts."""
        if not batch_texts:
            return []
        kwargs = self._request_kwargs(batch_texts)
        
        response = self._session.post(**kwargs)
        if response.status_code != 200:
            raise RuntimeError(
                f"Infinity API error {response.status_code}: {response.text}"
            )
        data = response.json().get("data", [])
        return [np.array(item.get("embedding", []), dtype=np.float32) for item in data]


    async def _async_request(
        self, session: aiohttp.ClientSession, batch_texts: List[str]
    ) -> List["np.ndarray"]:
        """Send an asynchronous request to embed a batch of texts."""
        if not batch_texts:
            return []
        kwargs = self._request_kwargs(batch_texts)
        async with session.post(**kwargs) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(
                    f"Infinity API error {response.status}: {text}"
                )
            payload = await response.json()
            data = payload.get("data", [])
            return [np.array(item.get("embedding", []), dtype=np.float32) for item in data]


    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Synchronous batch embedding."""
        if not texts:
            return []

        batches, restore = self._prepare_batches(texts)
        if len(batches) == 1:
            results = self._sync_request(batches[0])
        else:
            workers = min(len(batches), 32)
            with ThreadPoolExecutor(max_workers=workers) as exe:
                chunks = list(exe.map(self._sync_request, batches))
            # flatten list of lists
            results = [emb for chunk in chunks for emb in chunk]
        return restore(results)

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text string synchronously."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        embeddings = self.embed_batch([text])
        return np.array(embeddings[0] if embeddings else [], dtype=np.float32)

    async def aembed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Asynchronous batch embedding."""
        if not texts:
            return []
        batches, restore = self._prepare_batches(texts)
        conn = aiohttp.TCPConnector(limit=32)
        async with aiohttp.ClientSession(connector=conn, trust_env=True) as session:
            tasks = [self._async_request(session, b) for b in batches]
            chunks: List[List[np.ndarray]] = await asyncio.gather(*tasks)
        flat: List[np.ndarray] = [emb for chunk in chunks for emb in chunk]
        return restore(flat)

    async def aembed(self, text: str) -> "np.ndarray":
        """Embed a single text string asynchronously."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        embeddings = await self.aembed_batch([text])
        return np.array(embeddings[0] if embeddings else [], dtype=np.float32)


    @classmethod
    def _import_dependencies(cls) -> None:
        """Lazily import the numpy package."""
        if cls.is_available():
            global np
            import numpy as np
        else:
            raise ImportError(
                "Missing dependency 'numpy'. "
                "Install with `pip install numpy`"
            )

    @property
    def dimension(self) -> int:
        """Embedding vector dimension for the selected model."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> None:
        """InfinityEmbeddings does not support token counting."""
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if numpy is installed, required for embeddings."""
        return importutil.find_spec("numpy") is not None

    def __repr__(self) -> str:
        """String representation of the InfinityEmbeddings instance."""
        return f"InfinityEmbeddings(model={self.model!r})"
