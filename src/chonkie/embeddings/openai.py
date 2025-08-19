"""OpenAI embeddings."""

import importlib.util as importutil
import os
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseEmbeddings

if TYPE_CHECKING:
    import numpy as np
    import tiktoken


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings implementation using their API.
    
    Args:
        model: The model to use.
        tokenizer: The tokenizer to use. Can be loaded directly if it's a OpenAI model, otherwise needs to be provided.
        dimension: The dimension of the embedding model to use. Can be inferred if it's a OpenAI model, otherwise needs to be provided.
        base_url: The base URL to use.
        api_key: The API key to use.
        organization: The organization to use.
        max_retries: The maximum number of retries to use.
        timeout: The timeout to use.
        batch_size: The batch size to use.

    """

    AVAILABLE_MODELS = {
        "text-embedding-3-small": {
            "dimension": 1536,
            "max_tokens": 8192,
        },
        "text-embedding-3-large": {
            "dimension": 3072,
            "max_tokens": 8192,
        },
        "text-embedding-ada-002": {
            "dimension": 1536,
            "max_tokens": 8192,
        },
    }

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tokenizer: Optional[Any] = None,
        dimension: Optional[int] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        **kwargs: Dict[str, Any],
    ):
        """Initialize OpenAI embeddings.

        Args:
            model: Name of the OpenAI embedding model to use
            tokenizer: The tokenizer to use. Can be loaded directly if it's a OpenAI model, otherwise needs to be provided.
            dimension: The dimension of the embedding model to use. Can be inferred if it's a OpenAI model, otherwise needs to be provided.
            max_tokens: The maximum number of tokens to use. Can be inferred if it's a OpenAI model, otherwise needs to be provided.
            base_url: The base URL to use.
            api_key: OpenAI API key (if not provided, looks for OPENAI_API_KEY env var)
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for API requests
            batch_size: Maximum number of texts to embed in one API call
            **kwargs: Additional keyword arguments to pass to the OpenAI client.

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        # Initialize the model
        self.model = model
        self.base_url = base_url
        self._batch_size = batch_size

        # Do something for the tokenizer
        if tokenizer is not None: 
            self._tokenizer = tokenizer
        elif model in self.AVAILABLE_MODELS:
            self._tokenizer = tiktoken.encoding_for_model(model)
        else:
            raise ValueError(f"Tokenizer not found for model {model}. Please provide a tokenizer.")

        # Do something for the dimension
        if dimension is not None:
            self._dimension = dimension
        elif model in self.AVAILABLE_MODELS:
            self._dimension = self.AVAILABLE_MODELS[model]['dimension']
        else: 
            raise ValueError(f"Dimension not found for model {model}. Please provide a dimension.")

        # Do something for the max tokens
        if max_tokens is not None:
            self._max_tokens = max_tokens
        elif model in self.AVAILABLE_MODELS:
            self._max_tokens = self.AVAILABLE_MODELS[model]['max_tokens']
        else:
            raise ValueError(f"Max tokens not found for model {model}. Please provide a max tokens.")

        # Setup OpenAI client
        self.client = OpenAI(               # type: ignore
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        if self.client.api_key is None:
            raise ValueError(
                "OpenAI API key not found. Either pass it as api_key or set OPENAI_API_KEY environment variable."
            )

    @lru_cache(maxsize=4096)
    def _truncate(self, text: str) -> str:
        """Truncate the text to be below the max token count."""
        max_tokens = self._max_tokens
        token_estimate = len(text) // 5 
        if token_estimate > max_tokens:
          tokens = self._tokenizer.encode(text)
          if len(tokens) > max_tokens:
              warnings.warn(f"OpenAIEmbeddings encountered a text that is too long. Truncating to {max_tokens} tokens.")
              return self._tokenizer.decode(tokens[:max_tokens])
        return text

    def embed(self, text: str) -> "np.ndarray":
        """Get embeddings for a single text."""
        text = self._truncate(text)
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Get embeddings for multiple texts using batched API calls."""
        if not texts:
            return []

        all_embeddings = []

        # Truncate all the texts
        texts = [self._truncate(text) for text in texts]

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # Sort embeddings by index as OpenAI might return them in different order
                sorted_embeddings = sorted(response.data, key=lambda x: x.index)
                embeddings = [
                    np.array(e.embedding, dtype=np.float32) for e in sorted_embeddings
                ]
                all_embeddings.extend(embeddings)

            except Exception as e:
                # If the batch fails, try one by one
                if len(batch) > 1:
                    warnings.warn(
                        f"Batch embedding failed: {str(e)}. Trying one by one."
                    )
                    individual_embeddings = [self.embed(text) for text in batch]
                    all_embeddings.extend(individual_embeddings)
                else:
                    raise e

        return all_embeddings

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute cosine similarity between two embeddings."""
        return np.float32(np.divide(np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> "tiktoken.Encoding":
        """Return a tiktoken tokenizer object."""
        return self._tokenizer  # type: ignore[return-value]

    def _is_available(self) -> bool:
        """Check if the OpenAI package is available."""
        # We should check for OpenAI package alongside Numpy and tiktoken
        return (
            importutil.find_spec("openai") is not None
            and importutil.find_spec("numpy") is not None
            and importutil.find_spec("tiktoken") is not None
        )

    def _import_dependencies(self) -> None:
        """Lazy import dependencies for the embeddings implementation.

        This method should be implemented by all embeddings implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.
        """
        if self._is_available():
            global np, tiktoken, OpenAI
            import numpy as np
            import tiktoken
            from openai import OpenAI
        else:
            raise ImportError(
                'One (or more) of the following packages is not available: openai, numpy, tiktoken. Please install it via `pip install "chonkie[openai]"`'
            )

    def __repr__(self) -> str:
        """Representation of the OpenAIEmbeddings instance."""
        return f"OpenAIEmbeddings(model={self.model})"