"""OpenAI embeddings."""

import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseEmbeddings

if TYPE_CHECKING:
    try:
        import numpy as np
        import tiktoken
    except ImportError:
        np = Any
        tiktoken = Any 


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
        show_warnings: Whether to show warnings about token usage.

    """

    AVAILABLE_MODELS = {
        "text-embedding-3-small": 1536,  # Latest model, best performance/cost ratio
        "text-embedding-3-large": 3072,  # Latest model, highest performance
        "text-embedding-ada-002": 1536,  # Legacy model
    }

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tokenizer: Optional[str] = None,
        dimension: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        show_warnings: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Initialize OpenAI embeddings.

        Args:
            model: Name of the OpenAI embedding model to use
            tokenizer: The tokenizer to use. Can be loaded directly if it's a OpenAI model, otherwise needs to be provided.
            dimension: The dimension of the embedding model to use. Can be inferred if it's a OpenAI model, otherwise needs to be provided.
            base_url: The base URL to use.
            api_key: OpenAI API key (if not provided, looks for OPENAI_API_KEY env var)
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for API requests
            batch_size: Maximum number of texts to embed in one API call
            show_warnings: Whether to show warnings about token usage
            **kwargs: Additional keyword arguments to pass to the OpenAI client.

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        # Initialize the model
        self.model = model
        self.base_url = base_url
        self._batch_size = batch_size
        self._show_warnings = show_warnings

        # Do something for the tokenizer
        if tokenizer is not None: 
            self._tokenizer = tokenizer
        elif model in self.AVAILABLE_MODELS:
            self._tokenizer = tiktoken.encoding_for_model(model) # type: ignore
        else:
            raise ValueError(f"Tokenizer not found for model {model}. Please provide a tokenizer.")

        # Do something for the dimension
        if dimension is not None:
            self._dimension = dimension
        elif model in self.AVAILABLE_MODELS:
            self._dimension = self.AVAILABLE_MODELS[model]

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

    def embed(self, text: str) -> "np.ndarray":
        """Get embeddings for a single text."""
        token_count = self.count_tokens(text)
        if token_count > 8191 and self._show_warnings:  # OpenAI's token limit
            warnings.warn(
                f"Text has {token_count} tokens which exceeds the model's limit of 8191. "
                "It will be truncated."
            )

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

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # Check token counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self._show_warnings:
                for text, count in zip(batch, token_counts):
                    if count > 8191:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's limit of 8191. "
                            "It will be truncated."
                        )

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

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self._tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        tokens = self._tokenizer.encode_batch(texts)
        return [len(t) for t in tokens]

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute cosine similarity between two embeddings."""
        return np.float32(np.divide(np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> "tiktoken.Encoding":
        """Return a tiktoken tokenizer object."""
        return self._tokenizer

    def is_available(self) -> bool:
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
        if self.is_available():
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
