"""Embeddings implementation using VoyageAi."""

import asyncio
import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from .base import BaseEmbeddings

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class VoyageAIEmbeddings(BaseEmbeddings):
    """Voyage Embeddings client for interfacing with the VoyageAI API."""

    # Supported models with (allowed dimension, max_tokens)
    AVAILABLE_MODELS = {
        "voyage-3-large": ((1024, 256, 512, 2048), 32000),
        "voyage-3": ((1024,), 32000),
        "voyage-3-lite": ((512,), 32000),
        "voyage-code-3": ((1024, 256, 512, 2048), 32000),
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
        """Initialize the VoyageAI embeddings client.

        Args:
            model: Name of the Voyage model to use (must be in AVAILABLE_MODELS).
            api_key: API key for authentication (or set VOYAGEAI_API_KEY env var).
            max_retries: Maximum retry attempts for API calls.
            timeout: Timeout in seconds for API requests.
            output_dimension: Optional target embedding dimension.
            batch_size: Number of texts per batch (max 128).
            truncation: Whether to truncate inputs exceeding model token limit.

        Raises:
            ValueError: If model is unsupported or invalid output_dimension.
            ImportError: If voyageai package is not installed.

        """
        super().__init__()

        # Check if the API key is provided or set in the environment variable
        key = api_key or os.getenv("VOYAGE_API_KEY")
        if key is None:
            raise ValueError(
                "No API key provided. Please set VOYAGE_API_KEY environment variable or pass in an api_key parameter.",
            )

        try:
            import voyageai
            from tokenizers import Tokenizer
        except ImportError as ie:
            raise ImportError(
                "One (or more) of the following packages is not available: tokenizers or voyageai. "
                "Please install it via `pip install chonkie[voyageai]`",
            ) from ie

        # Initialize the API clients
        self._client = voyageai.Client(api_key=key, max_retries=max_retries, timeout=timeout)
        self._aclient = voyageai.AsyncClient(api_key=key, max_retries=max_retries, timeout=timeout)

        # Check if the model is supported
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model!r} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}",
            )

        # Set the model and tokenizer
        self.model = model
        allowed_dims, self._token_limit = self.AVAILABLE_MODELS[model]
        self._allowed_dims = set(allowed_dims)

        # first entry of allowed dimensions is the default dimension for that model
        self._dimension = allowed_dims[0]

        try:
            self._tokenizer = Tokenizer.from_pretrained(f"voyageai/{model}")
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model {model}: {e}")

        # Set the truncation, batch size, and output dimension
        self.truncation = truncation
        self.batch_size = min(batch_size, 128)
        if output_dimension is None:
            self.output_dimension = self._dimension
        elif output_dimension in self._allowed_dims:
            self.output_dimension = output_dimension
        else:
            raise ValueError(
                f"Invalid output_dimension={output_dimension} for model={model}. "
                f"Allowed: {sorted(self._allowed_dims)}",
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self._tokenizer.encode(text))

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens in multiple texts."""
        tokens = self._tokenizer.encode_batch(texts)
        return [len(t) for t in tokens]

    def embed(
        self,
        text: str,
        input_type: Optional[Literal["query", "document"]] = None,
    ) -> np.ndarray:
        """Obtain embedding for a single text synchronously.

        Args:
            text: The input string to embed.
            input_type: Optional tag indicating 'query' or 'document'.

        Returns:
            A NumPy array of the embedding vector.

        """
        tokens = self.count_tokens(text)
        if tokens > self._token_limit and self.truncation:
            warnings.warn(
                f"Input has {tokens} tokens (>{self._token_limit}); truncating.",
                UserWarning,
            )
        try:
            response = self._client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
            )
        except Exception as e:
            raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e

        return np.array(response.embeddings[0], dtype=np.float32)

    async def aembed(
        self,
        text: str,
        input_type: Optional[Literal["query", "document"]] = None,
    ) -> "np.ndarray":
        """Obtain embedding for a single text asynchronously.

        Args:
            text: The input string to embed.
            input_type: Optional tag indicating 'query' or 'document'.

        Returns:
            A NumPy array of the embedding vector.

        """
        tokens = self.count_tokens(text)
        if tokens > self._token_limit and self.truncation:
            warnings.warn(
                f"Input has {tokens} tokens (>{self._token_limit}); truncating.",
                UserWarning,
            )

        try:
            response = await self._aclient.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
            )
        except Exception as e:
            raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e

        return np.array(response.embeddings[0], dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        input_type: Optional[Literal["query", "document"]] = None,
    ) -> list[np.ndarray]:
        """Obtain embeddings for a batch of texts synchronously.

        Args:
            texts: List of input strings to embed.
            input_type: Optional tag indicating 'query' or 'document'.

        Returns:
            List of NumPy arrays representing embedding vectors.

        """
        embeddings: list["np.ndarray"] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Check token counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self.truncation:
                for count in token_counts:
                    if count > self._token_limit:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's limit of {self._token_limit}. "
                            "It will be truncated.",
                        )
            try:
                response = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension,
                )
                embeddings.extend(np.array(emb, dtype=np.float32) for emb in response.embeddings)
            except Exception as e:
                raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e

        return embeddings

    async def __process_batch(
        self,
        batch: list[str],
        input_type: Optional[Literal["query", "document"]] = None,
    ) -> list["np.ndarray"]:
        """Process a single batch of texts to obtain embeddings.

        This method is intended for internal use only.

        Args:
            batch: List of input strings to embed.
            input_type: Optional tag indicating 'query' or 'document'.

        Returns:
            List of NumPy arrays representing embedding vectors for the batch.

        """
        if not batch:
            return []

        # Check token counts and warn if necessary
        token_counts = self.count_tokens_batch(batch)
        if self.truncation:
            for count in token_counts:
                if count > self._token_limit:
                    warnings.warn(
                        f"Text has {count} tokens which exceeds the model's limit of {self._token_limit}. "
                        "It will be truncated.",
                    )
        try:
            response = await self._aclient.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
            )
            return [np.array(emb, dtype=np.float32) for emb in response.embeddings]
        except Exception as e:
            raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e

    async def aembed_batch(
        self,
        texts: list[str],
        input_type: Optional[Literal["query", "document"]] = None,
    ) -> list["np.ndarray"]:
        """Obtain embeddings for a batch of texts asynchronously.

        Args:
            texts: List of input strings to embed.
            input_type: Optional tag indicating 'query' or 'document'.

        Returns:
            List of NumPy arrays representing embedding vectors.

        """
        if not texts:
            return []

        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        # Process each batch asynchronously
        tasks = [self.__process_batch(batch, input_type) for batch in batches]
        # Gather results
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists into a single list
        embeddings = []
        for result in results:
            embeddings.extend(result)
        return embeddings

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the voyageai package is available."""
        return (
            importutil.find_spec("voyageai") is not None
            and importutil.find_spec("tokenizers") is not None
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer(self) -> "Tokenizer":
        """Get the tokenizer instance used by the embeddings model.

        Returns:
            Tokenizer: A Tokenizer instance for the voyageai embeddings model.

        """
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the VoyageAIEmbeddings object."""
        return f"VoyageAIEmbeddings(model={self.model}, dimension={self.dimension})"
