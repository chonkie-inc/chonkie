"""Azure OpenAI embeddings implementation."""

import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .base import BaseEmbeddings

if TYPE_CHECKING:
    from tiktoken import Encoding


class AzureOpenAIEmbeddings(BaseEmbeddings):
    """Embedding class using Azure OpenAI service.

    Args:
        model: Logical model name (used for tokenizer + dimension, not API).
        azure_endpoint: Azure OpenAI resource endpoint URL (or use AZURE_OPENAI_ENDPOINT env var).
        deployment: Name of the Azure deployment (required unless same as model).
        azure_api_key: Optional Azure API key (or use AZURE_OPENAI_API_KEY env var or Entra ID).
        tokenizer: Optional tokenizer override.
        dimension: Optional embedding dimension override.
        batch_size: Number of texts to embed per batch.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.

    """

    AVAILABLE_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        azure_endpoint: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        dimension: Optional[int] = None,
        azure_api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
        deployment: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        **kwargs: dict[str, Any],
    ):
        """Initialize Azure OpenAI embeddings.

        Args:
            model: Name of the Azure OpenAI embedding model to use.
            azure_endpoint: Azure OpenAI resource endpoint URL (defaults to AZURE_OPENAI_ENDPOINT env var).
            tokenizer: Optional tokenizer override.
            dimension: Optional embedding dimension override.
            azure_api_key: Optional Azure API key (defaults to AZURE_OPENAI_API_KEY env var, or use Entra ID).
            api_version: Azure OpenAI API version.
            deployment: Name of the Azure deployment (required unless same as model).
            max_retries: Maximum number of retries for failed requests.
            timeout: Timeout in seconds for API requests.
            batch_size: Maximum number of texts to embed in one API call.
            **kwargs: Additional keyword arguments to pass to the OpenAI client.

        """
        super().__init__()

        # Get azure_endpoint from env var if not provided
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not azure_endpoint:
            raise ValueError(
                "`azure_endpoint` is required for Azure OpenAI. "
                "Provide it as a parameter or set the AZURE_OPENAI_ENDPOINT environment variable.",
            )

        # Get azure_api_key from env var if not provided
        if azure_api_key is None:
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        try:
            import tiktoken
            from openai import AzureOpenAI
        except ImportError as ie:
            raise ImportError(
                "Required packages not found. Install with `pip install chonkie[azure-openai]`.",
            ) from ie

        self.model = model
        self._deployment = deployment or model
        self.base_url = azure_endpoint
        self._batch_size = batch_size

        # Initialize Azure client
        if azure_api_key:
            self.client = AzureOpenAI(  # type: ignore
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            self.client = AzureOpenAI(  # type: ignore
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )

        # Tokenizer
        if tokenizer is not None:
            self._tokenizer = tokenizer
        elif model in self.AVAILABLE_MODELS:
            import tiktoken

            self._tokenizer = tiktoken.encoding_for_model(model)
        else:
            raise ValueError(f"Tokenizer not available for model '{model}'.")

        # Embedding dimension
        if dimension is not None:
            self._dimension = dimension
        elif model in self.AVAILABLE_MODELS:
            self._dimension = self.AVAILABLE_MODELS[model]
        else:
            raise ValueError("Embedding dimension must be provided for unknown models.")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string."""
        response = self.client.embeddings.create(
            model=self._deployment,
            input=text,
        )

        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of strings."""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self._deployment,
                    input=batch,
                )
                sorted_embeddings = sorted(response.data, key=lambda x: x.index)
                embeddings = [np.array(e.embedding, dtype=np.float32) for e in sorted_embeddings]
                all_embeddings.extend(embeddings)
            except Exception as e:
                if len(batch) > 1:
                    warnings.warn(f"Batch failed: {e}. Falling back to single embedding calls.")
                    all_embeddings.extend(self.embed(t) for t in batch)
                else:
                    raise
        return all_embeddings

    def similarity(self, u: np.ndarray, v: np.ndarray) -> np.float32:
        """Compute cosine similarity between two vectors."""
        return np.float32(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer(self) -> "Encoding":
        """Return a tiktoken tokenizer object."""
        return self._tokenizer

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the required dependencies are available."""
        return (
            importutil.find_spec("openai") is not None
            and importutil.find_spec("tiktoken") is not None
            and importutil.find_spec("azure.identity") is not None
        )

    def __repr__(self) -> str:
        """Representation of the AzureOpenAIEmbeddings instance."""
        return f"AzureOpenAIEmbeddings(model={self.model}, deployment={self._deployment}, endpoint={self.base_url})"
