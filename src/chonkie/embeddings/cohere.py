"""Embeddings implementation using Cohere's API."""

import importlib
import os
import warnings
from typing import TYPE_CHECKING, List, Optional

import requests

if TYPE_CHECKING:
    import numpy as np
    import tokenizers
    try:
        from cohere import ClientV2
    except ImportError:
        class ClientV2:  # type: ignore
            """Stub class for cohere ClientV2 when not available."""

            pass

from .base import BaseEmbeddings


class CohereEmbeddings(BaseEmbeddings):
    """Cohere embeddings implementation using their API."""

    AVAILABLE_MODELS = {
        # cohere v3.0 models
        "embed-english-v3.0": (True, 1024),  # tokenizer from tokenizers
        "embed-multilingual-v3.0": (
            False,
            1024,
        ),  # not listed in the cohere models api list
        "embed-english-light-v3.0": (True, 384),  # from tokenizers
        "embed-multilingual-light-v3.0": (
            False,
            384,
        ),  # not listed in the cohere models api list
        # cohere v2.0 models
        "embed-english-v2.0": (
            False,
            4096,
        ),  # url is not available in the cohere models api list
        "embed-english-light-v2.0": (False, 1024),  # not listed in the models list
        "embed-multilingual-v2.0": (True, 768),  # from tokenizers
    }

    DEFAULT_MODEL = "embed-english-light-v3.0"
    TOKENIZER_BASE_URL = "https://storage.googleapis.com/cohere-public/tokenizers/"

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
        """Initialize Cohere embeddings.

        Args:
            model: name of the Cohere embedding model to use
            api_key: (optional) Cohere API key (if not provided, looks for COHERE_API_KEY environment variable)
            client_name: (optional) client name for API requests
            max_retries: maximum number of retries for failed requests
            timeout: timeout in seconds for API requests
            batch_size: maximum number of texts to embed in one API call (maximum allowed by Cohere is 96)
            show_warnings: whether to show warnings about token usage and truncation

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model} is not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model = model
        self._dimension = self.AVAILABLE_MODELS[model][1]
        tokenizer_url = (
            self.TOKENIZER_BASE_URL
            + (model if self.AVAILABLE_MODELS[model][0] else self.DEFAULT_MODEL)
            + ".json"
        )
        response = requests.get(tokenizer_url)
        self._tokenizer = tokenizers.Tokenizer.from_str(response.text)
        self._batch_size = min(batch_size, 96)  # max batch size for cohere is 96
        self._show_warnings = show_warnings
        self._max_retries = max_retries
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

        if self._api_key is None:
            raise ValueError(
                "Cohere API key not found. Either pass it as api_key or set COHERE_API_KEY environment variable."
            )

        self.model = model
        self._dimension = self.AVAILABLE_MODELS[model][1]
        tokenizer_url = (
            self.TOKENIZER_BASE_URL
            + (model if self.AVAILABLE_MODELS[model][0] else self.DEFAULT_MODEL)
            + ".json"
        )
        response = requests.get(tokenizer_url)
        self._tokenizer = tokenizers.Tokenizer.from_str(response.text)
        self._batch_size = min(batch_size, 96)  # max batch size for cohere is 96
        self._show_warnings = show_warnings
        self._max_retries = max_retries
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

        if self._api_key is None:
            raise ValueError(
                "Cohere API key not found. Either pass it as api_key or set COHERE_API_KEY environment variable."
            )

        # setup Cohere client
        self.client = ClientV2(
            api_key=api_key or os.getenv("COHERE_API_KEY"),
            client_name=client_name,
            timeout=timeout,
        )

    def embed(self, text: str) -> "np.ndarray":
        """Generate embeddings for a single text."""
        token_count = self.count_tokens(text)
        if (
            token_count > 512 and self._show_warnings
        ):  # Cohere models max_context_length
            warnings.warn(
                f"Text has {token_count} tokens which exceeds the model's context length of 512."
                "Generation may not be optimal"
            )

        for _ in range(self._max_retries):
            try:
                response = self.client.embed(
                    model=self.model,
                    input_type="search_document",
                    embedding_types=["float"],
                    texts=[text],
                )

                return np.array(response.embeddings.float_[0], dtype=np.float32)  # type: ignore[index]
            except Exception as e:
                if self._show_warnings:
                    warnings.warn(
                        f"There was an exception while generating embeddings. Exception: {str(e)}. Retrying..."
                    )

        raise RuntimeError("Unable to generate embeddings through Cohere.")

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Get embeddings for multiple texts using batched API calls."""
        if not texts:
            return []

        all_embeddings = []

        # process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # check token_counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self._show_warnings:
                for _, count in zip(batch, token_counts):
                    if count > 512:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's context length of 512."
                            "Generation may not be optimal."
                        )

            try:
                for _ in range(self._max_retries):
                    try:
                        response = self.client.embed(
                            model=self.model,
                            input_type="search_document",
                            embedding_types=["float"],
                            texts=batch,
                        )

                        embeddings = [
                            np.array(e, dtype=np.float32)
                            for e in response.embeddings.float_  # type: ignore[union-attr]
                        ]
                        all_embeddings.extend(embeddings)
                        break
                    except Exception as e:
                        if self._show_warnings:
                            warnings.warn(
                                f"There was an exception while generating embeddings. Exception: {str(e)}. Retrying..."
                            )

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
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        tokens = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        return [len(t) for t in tokens]

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute cosine similarity between two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=np.float32
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> "tokenizers.Tokenizer":
        """Return a tokenizers tokenizer object of the current model."""
        return self._tokenizer

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the Cohere package is available."""
        return importlib.util.find_spec("cohere") is not None

    @classmethod
    def _import_dependencies(cls) -> None:
        """Lazy import dependencies for the embeddings implementation.

        This method should be implemented by all embeddings implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.

        """
        if cls._is_available():
            global np, tokenizers, ClientV2
            import numpy as np
            import tokenizers
            from cohere import ClientV2
        else:
            raise ImportError(
                "cohere is not available. Please install it via `pip install chonkie[cohere]`"
            )

    def __repr__(self) -> str:
        """Return a string representation of the CohereEmbeddings object."""
        return f"CohereEmbeddings(model={self.model})"
