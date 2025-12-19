"""LiteLLM embeddings implementation for unified access to 100+ providers."""

import importlib.util as importutil
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseEmbeddings


class LiteLLMEmbeddings(BaseEmbeddings):
    """LiteLLM embeddings implementation for unified access to multiple providers.

    LiteLLM provides a unified, OpenAI-compatible API for 100+ embedding providers
    including OpenAI, VoyageAI, Cohere, Bedrock, and more. This solves provider-specific
    issues like VoyageAI's circular import bug while providing consistent error handling
    and retry logic across all providers.

    Args:
        model: Model name in LiteLLM format. Examples:
            - "text-embedding-3-small" (OpenAI, default provider)
            - "voyage/voyage-3-large" (VoyageAI)
            - "cohere/embed-english-v3.0" (Cohere)
            - "bedrock/amazon.titan-embed-text-v1" (AWS Bedrock)
        api_key: API key for the provider (falls back to environment variables)
        api_base: Optional custom API endpoint
        timeout: Timeout in seconds for API requests
        max_retries: Maximum number of retries for failed requests
        batch_size: Maximum number of texts to embed in one API call
        dimension: Optional embedding dimension (auto-detected if possible)
        **kwargs: Additional arguments passed to litellm.embedding()

    Examples:
        >>> # Using VoyageAI through LiteLLM (avoids circular import issues)
        >>> embeddings = LiteLLMEmbeddings(model="voyage/voyage-3-large")
        >>> vector = embeddings.embed("Hello world")

        >>> # Using Cohere
        >>> embeddings = LiteLLMEmbeddings(model="cohere/embed-english-v3.0")

        >>> # Using OpenAI
        >>> embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")

    """

    # Known model dimensions for caching
    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "voyage/voyage-3-large": 1024,
        "voyage/voyage-3": 1024,
        "voyage/voyage-3-lite": 512,
        "voyage/voyage-code-3": 1024,
        "voyage/voyage-finance-2": 1024,
        "voyage/voyage-law-2": 1024,
        "voyage/voyage-code-2": 1536,
        "cohere/embed-english-v3.0": 1024,
        "cohere/embed-english-light-v3.0": 384,
        "cohere/embed-multilingual-v3.0": 1024,
        "cohere/embed-multilingual-light-v3.0": 384,
    }

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 128,
        dimension: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize LiteLLM embeddings.

        Args:
            model: Model name in LiteLLM format
            api_key: API key (falls back to environment variables)
            api_base: Optional custom API endpoint
            timeout: Timeout in seconds for API requests
            max_retries: Maximum number of retries
            batch_size: Maximum number of texts per API call
            dimension: Optional embedding dimension (auto-detected if possible)
            **kwargs: Additional arguments passed to litellm.embedding()

        """
        super().__init__()

        try:
            import litellm
        except ImportError as ie:
            raise ImportError(
                'litellm package is not available. Please install it via `pip install "chonkie[litellm]"`',
            ) from ie

        # Store configuration
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries
        self._batch_size = batch_size
        self._extra_kwargs = kwargs

        # Configure LiteLLM settings
        litellm.set_verbose = False

        # Determine dimension
        if dimension is not None:
            self._dimension = dimension
        elif model in self.KNOWN_DIMENSIONS:
            self._dimension = self.KNOWN_DIMENSIONS[model]
        else:
            # Auto-detect dimension by embedding a test string
            self._dimension = self._detect_dimension()

        # Initialize tokenizer (provider-specific)
        self._tokenizer = self._initialize_tokenizer()

    def _detect_dimension(self) -> int:
        """Auto-detect embedding dimension by making a test API call.

        Returns:
            int: Detected embedding dimension

        """
        try:
            test_embedding = self.embed("test")
            return len(test_embedding)
        except Exception as e:
            warnings.warn(
                f"Failed to auto-detect embedding dimension: {e}. "
                "Please provide dimension explicitly.",
                UserWarning,
            )
            # Default fallback
            return 1536

    def _initialize_tokenizer(self) -> Any:
        """Initialize tokenizer based on the model provider.

        Returns:
            Any: Tokenizer object (provider-specific)

        """
        import tiktoken

        # Extract provider from model name
        provider = self._get_provider()

        try:
            # For OpenAI models, use tiktoken
            if provider in ("openai", None):
                # Extract base model name
                base_model = self.model.split("/")[-1]
                return tiktoken.encoding_for_model(base_model)

            # For VoyageAI, try to load their tokenizer
            elif provider == "voyage":
                try:
                    from tokenizers import Tokenizer

                    model_name = self.model.split("/")[-1]
                    return Tokenizer.from_pretrained(f"voyageai/{model_name}")
                except Exception:
                    # Fallback to cl100k_base for token estimation
                    return tiktoken.get_encoding("cl100k_base")

            # For other providers, use cl100k_base as fallback
            else:
                return tiktoken.get_encoding("cl100k_base")

        except Exception as e:
            warnings.warn(
                f"Failed to initialize tokenizer: {e}. Using cl100k_base fallback.",
                UserWarning,
            )
            return tiktoken.get_encoding("cl100k_base")

    def _get_provider(self) -> Optional[str]:
        """Extract provider name from model string.

        Returns:
            Optional[str]: Provider name or None if not specified

        """
        if "/" in self.model:
            return self.model.split("/")[0]
        return None

    def _prepare_api_call_kwargs(self) -> Dict[str, Any]:
        """Prepare kwargs for litellm.embedding() call.

        Returns:
            Dict[str, Any]: Dictionary of kwargs for API call

        """
        kwargs = {
            "model": self.model,
            "timeout": self.timeout,
            **self._extra_kwargs,
        }

        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Add API base if provided
        if self.api_base:
            kwargs["api_base"] = self.api_base

        return kwargs

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            np.ndarray: Embedding vector

        Raises:
            RuntimeError: If the API call fails after retries

        """
        import litellm

        try:
            kwargs = self._prepare_api_call_kwargs()
            response = litellm.embedding(
                input=[text],
                **kwargs,
            )

            # Extract embedding from response
            embedding = response.data[0]["embedding"]
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"LiteLLM API error during embedding: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts using batched API calls.

        Args:
            texts: List of text strings to embed

        Returns:
            List[np.ndarray]: List of embedding vectors

        Raises:
            RuntimeError: If the API call fails after retries

        """
        if not texts:
            return []

        import litellm

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                kwargs = self._prepare_api_call_kwargs()
                response = litellm.embedding(
                    input=batch,
                    **kwargs,
                )

                # Extract embeddings and sort by index
                sorted_data = sorted(response.data, key=lambda x: x["index"])
                embeddings = [
                    np.array(item["embedding"], dtype=np.float32) for item in sorted_data
                ]
                all_embeddings.extend(embeddings)

            except Exception as e:
                # If the batch fails, try one by one
                if len(batch) > 1:
                    warnings.warn(f"Batch embedding failed: {str(e)}. Trying one by one.")
                    individual_embeddings = [self.embed(text) for text in batch]
                    all_embeddings.extend(individual_embeddings)
                else:
                    raise RuntimeError(f"LiteLLM API error during embedding: {e}") from e

        return all_embeddings

    def similarity(self, u: np.ndarray, v: np.ndarray) -> np.float32:
        """Compute cosine similarity between two embeddings.

        Args:
            u: First embedding vector
            v: Second embedding vector

        Returns:
            np.float32: Cosine similarity score

        """
        return np.float32(np.divide(np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: Dimension of the embedding vectors

        """
        return self._dimension

    def get_tokenizer(self) -> Any:
        """Return the tokenizer object.

        Returns:
            Any: Tokenizer object (provider-specific)

        """
        return self._tokenizer

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the litellm package is available.

        Returns:
            bool: True if litellm is installed, False otherwise

        """
        return importutil.find_spec("litellm") is not None

    def __repr__(self) -> str:
        """Return string representation of the LiteLLMEmbeddings instance.

        Returns:
            str: String representation

        """
        return f"LiteLLMEmbeddings(model={self.model}, dimension={self.dimension})"
