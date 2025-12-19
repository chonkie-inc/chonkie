"""Catsu embeddings adapter for unified embedding provider access.

This adapter wraps the Catsu client library to provide a unified interface
for accessing 11+ embedding providers through Chonkie's BaseEmbeddings interface.

Supported providers: VoyageAI, OpenAI, Cohere, Gemini, Jina AI, Mistral,
Nomic, Cloudflare, MixedBread, DeepInfra, TogetherAI.
"""

import importlib.util as importutil
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseEmbeddings


class CatsuEmbeddings(BaseEmbeddings):
    """Unified embedding provider using Catsu client.

    This class wraps the Catsu client library to provide access to 11+ embedding
    providers through a single, consistent interface. Features include:
    - Automatic retry logic with exponential backoff
    - Built-in usage and cost tracking
    - Rich model metadata with MTEB/RTEB benchmark scores
    - Local tokenization without API calls
    - Full async/await support

    Args:
        model: Model name (e.g., "voyage-3", "text-embedding-3-small")
        provider: Optional provider name (e.g., "voyageai", "openai").
                  If not provided, Catsu will auto-detect from model name.
        api_keys: Optional dict of API keys by provider name
                  (e.g., {"voyageai": "key123"}). If not provided,
                  Catsu will look for environment variables.
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 30)
        verbose: Enable verbose logging (default: False)
        batch_size: Maximum number of texts to embed in one API call (default: 128)
        **kwargs: Additional keyword arguments to pass to the Catsu client

    Examples:
        >>> # Auto-detect provider from model name
        >>> embeddings = CatsuEmbeddings(model="voyage-3")
        >>> vector = embeddings.embed("hello world")

        >>> # Explicit provider specification
        >>> embeddings = CatsuEmbeddings(
        ...     model="voyage-3",
        ...     provider="voyageai",
        ...     api_keys={"voyageai": "your-key"}
        ... )

        >>> # Batch embedding
        >>> vectors = embeddings.embed_batch(["text1", "text2", "text3"])

    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        timeout: int = 30,
        verbose: bool = False,
        batch_size: int = 128,
        **kwargs: Dict[str, Any],
    ):
        """Initialize Catsu embeddings adapter.

        Args:
            model: Model name to use for embeddings
            provider: Optional provider name for explicit provider selection
            api_keys: Optional dict of API keys by provider name
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            batch_size: Maximum number of texts to embed in one API call
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If catsu package is not installed
            ValueError: If model or provider is invalid

        """
        super().__init__()

        # Store configuration
        self.model = model
        self.provider = provider
        self._batch_size = batch_size
        self._verbose = verbose

        # Initialize Catsu client
        try:
            import catsu
        except ImportError as e:
            raise ImportError(
                "The catsu package is not available. "
                'Please install it via `pip install "chonkie[catsu]"` or `pip install catsu`',
            ) from e

        self.client = catsu.Client(
            verbose=verbose,
            max_retries=max_retries,
            timeout=timeout,
            api_keys=api_keys,
        )

        # Cache for model metadata
        self._dimension: Optional[int] = None
        self._model_info: Optional[Any] = None

        # Validate model exists and is supported
        try:
            self._load_model_info()
        except Exception as e:
            warnings.warn(
                f"Could not load model info for {model}: {e}. Will attempt to use model anyway.",
            )

    def _load_model_info(self) -> None:
        """Load model metadata from Catsu catalog."""
        try:
            models = self.client.list_models(provider=self.provider)
            for model_info in models:
                if model_info.name == self.model:
                    self._model_info = model_info
                    self._dimension = model_info.dimensions
                    break

            if self._model_info is None and self._verbose:
                warnings.warn(
                    f"Model '{self.model}' not found in Catsu catalog. "
                    "Embedding may fail if model name is incorrect.",
                )
        except Exception as e:
            if self._verbose:
                warnings.warn(f"Failed to load model info: {e}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a vector representation.

        Args:
            text: Text string to embed

        Returns:
            np.ndarray: Embedding vector for the text (1D array)

        Raises:
            Various Catsu exceptions for API errors, auth failures, etc.

        """
        response = self.client.embed(
            model=self.model,
            input=text,
            provider=self.provider,
        )

        # Catsu returns List[List[float]], we need first embedding as 1D array
        embeddings_array = response.to_numpy()
        return embeddings_array[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using batched API calls.

        This method automatically handles batching to respect API limits
        and provider batch size constraints.

        Args:
            texts: List of text strings to embed

        Returns:
            List[np.ndarray]: List of embedding vectors (1D arrays)

        Raises:
            Various Catsu exceptions for API errors, auth failures, etc.

        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            try:
                response = self.client.embed(
                    model=self.model,
                    input=batch,
                    provider=self.provider,
                )

                # Convert to list of 1D numpy arrays
                embeddings_array = response.to_numpy()
                batch_embeddings = [embeddings_array[j] for j in range(len(batch))]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                # If batch fails, try one by one (with Catsu's built-in retries)
                if len(batch) > 1:
                    warnings.warn(f"Batch embedding failed: {str(e)}. Trying one by one.")
                    for text in batch:
                        individual_embedding = self.embed(text)
                        all_embeddings.append(individual_embedding)
                else:
                    raise e

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            int: Dimension of the embedding vectors

        Raises:
            RuntimeError: If dimension cannot be determined

        """
        if self._dimension is None:
            # Try to load model info if not already loaded
            self._load_model_info()

            # If still None, try to infer from actual embedding
            if self._dimension is None:
                try:
                    # Make a test embedding to determine dimension
                    test_embedding = self.embed("test")
                    self._dimension = len(test_embedding)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not determine embedding dimension for model {self.model}: {e}",
                    )

        return self._dimension

    def get_tokenizer(self) -> Any:
        """Return a tokenizer object that can be used for token counting.

        This returns a wrapper around Catsu's tokenize() method that provides
        a simple interface compatible with Chonkie's tokenizer protocol.

        Returns:
            CatsuTokenizerWrapper: Tokenizer wrapper object

        Examples:
            >>> embeddings = CatsuEmbeddings(model="voyage-3")
            >>> tokenizer = embeddings.get_tokenizer()
            >>> token_count = tokenizer.count("hello world")

        """
        return CatsuTokenizerWrapper(
            client=self.client,
            model=self.model,
            provider=self.provider,
        )

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the Catsu package is available.

        Returns:
            bool: True if catsu is installed, False otherwise

        """
        return importutil.find_spec("catsu") is not None

    def __repr__(self) -> str:
        """Return a string representation of the CatsuEmbeddings instance."""
        if self.provider:
            return f"CatsuEmbeddings(provider={self.provider}, model={self.model})"
        return f"CatsuEmbeddings(model={self.model})"


class CatsuTokenizerWrapper:
    """Wrapper around Catsu's tokenize method to provide a tokenizer interface.

    This class provides a simple tokenizer interface compatible with Chonkie's
    tokenizer protocol, using Catsu's built-in tokenization capabilities.

    Args:
        client: Catsu client instance
        model: Model name to use for tokenization
        provider: Optional provider name

    """

    def __init__(
        self,
        client: Any,
        model: str,
        provider: Optional[str] = None,
    ):
        """Initialize the tokenizer wrapper.

        Args:
            client: Catsu client instance
            model: Model name for tokenization
            provider: Optional provider name

        """
        self.client = client
        self.model = model
        self.provider = provider

    def count(self, text: str) -> int:
        """Count tokens in text without making an API call.

        Uses Catsu's local tokenization capabilities.

        Args:
            text: Text to count tokens for

        Returns:
            int: Number of tokens

        """
        response = self.client.tokenize(
            model=self.model,
            input=text,
            provider=self.provider,
        )
        return response.token_count

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Note: Not all providers expose token IDs. This method uses
        count() as a fallback.

        Args:
            text: Text to encode

        Returns:
            List[int]: Token IDs (or empty list if not supported)

        """
        # Catsu doesn't expose token IDs for all providers
        # Return empty list as fallback
        warnings.warn(
            "Token encoding not supported via Catsu. Use count() for token counting instead.",
        )
        return []

    def __repr__(self) -> str:
        """Return a string representation of the tokenizer wrapper."""
        return f"CatsuTokenizerWrapper(model={self.model})"
