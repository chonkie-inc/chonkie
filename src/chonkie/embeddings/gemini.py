"""Google Gemini embeddings implementation using the GenAI API."""

import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

from .base import BaseEmbeddings

if TYPE_CHECKING:
    import numpy as np
    from google import genai
    from google.genai import types


class GeminiEmbeddings(BaseEmbeddings):
    """Google Gemini embeddings implementation using the GenAI API.
    
    Args:
        model: The model to use.
        api_key: The API key to use.
        task_type: The task type for embeddings (SEMANTIC_SIMILARITY, CLASSIFICATION, etc.).
        max_retries: The maximum number of retries to use.
        batch_size: The batch size to use.
        show_warnings: Whether to show warnings about token usage.

    """

    AVAILABLE_MODELS = {
        "text-embedding-004": (768, 2048),  # (dimension, max_tokens)
        "embedding-001": (768, 2048),
        "gemini-embedding-exp-03-07": (3072, 8192),  # Experimental model
    }

    DEFAULT_MODEL = "gemini-embedding-exp-03-07"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        max_retries: int = 3,
        batch_size: int = 100,
        show_warnings: bool = True,
    ):
        """Initialize Gemini embeddings.

        Args:
            model: Name of the Gemini embedding model to use
            api_key: Gemini API key (if not provided, looks for GEMINI_API_KEY env var)
            task_type: Task type for embeddings (SEMANTIC_SIMILARITY, CLASSIFICATION, etc.)
            max_retries: Maximum number of retries for failed requests
            batch_size: Maximum number of texts to embed in one API call
            show_warnings: Whether to show warnings about token usage

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        # Initialize the model - use default if empty string provided
        self.model = model if model else self.DEFAULT_MODEL
        self.task_type = task_type
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._show_warnings = show_warnings
        self._chars_per_token = 6.5

        # Get dimension and max tokens for the model
        if self.model in self.AVAILABLE_MODELS:
            self._dimension, self._max_tokens = self.AVAILABLE_MODELS[self.model]
        else:
            # Use default model values if model not in list
            self._dimension, self._max_tokens = self.AVAILABLE_MODELS[self.DEFAULT_MODEL]
            if show_warnings:
                warnings.warn(
                    f"Model {self.model} not in known models list. Using default model '{self.DEFAULT_MODEL}' with dimension {self._dimension} and max tokens {self._max_tokens}."
                )

        # Setup Gemini client
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self._api_key is None:
            raise ValueError(
                "Gemini API key not found. Either pass it as api_key or set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=self._api_key)  # type: ignore

    def embed(self, text: str) -> "np.ndarray":
        """Get embeddings for a single text."""
        # Check token count and warn if necessary
        if self._show_warnings:
            token_count = self.count_tokens(text)
            if token_count > self._max_tokens:
                warnings.warn(
                    f"Text has {token_count} tokens which exceeds the model's limit of {self._max_tokens}. "
                    "Consider chunking the text."
                )

        for attempt in range(self._max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=self.task_type),  # type: ignore
                )
                
                # Extract embedding from result
                if hasattr(result, 'embeddings') and result.embeddings:
                    embedding = result.embeddings[0].values
                    return np.array(embedding, dtype=np.float32)
                else:
                    raise ValueError("No embeddings returned from API")

            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise RuntimeError(f"Failed to get embeddings after {self._max_retries} attempts: {str(e)}")
                if self._show_warnings:
                    warnings.warn(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying...")

        raise RuntimeError("Failed to get embeddings")

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # Check token counts and warn if necessary
            if self._show_warnings:
                for text in batch:
                    token_count = self.count_tokens(text)
                    if token_count > self._max_tokens:
                        warnings.warn(
                            f"Text has {token_count} tokens which exceeds the model's limit of {self._max_tokens}."
                        )

            try:
                for attempt in range(self._max_retries):
                    try:
                        # Process each text in the batch individually for now
                        # (Gemini API may not support true batch processing)
                        batch_embeddings = []
                        for text in batch:
                            result = self.client.models.embed_content(
                                model=self.model,
                                contents=text,
                                config=types.EmbedContentConfig(task_type=self.task_type),  # type: ignore
                            )
                            if hasattr(result, 'embeddings') and result.embeddings:
                                embedding = result.embeddings[0].values
                                batch_embeddings.append(np.array(embedding, dtype=np.float32))
                            else:
                                raise ValueError("No embeddings returned from API")
                        
                        all_embeddings.extend(batch_embeddings)
                        break

                    except Exception as e:
                        if attempt == self._max_retries - 1:
                            # If the batch fails, try one by one
                            if len(batch) > 1:
                                warnings.warn(
                                    f"Batch embedding failed: {str(e)}. Trying one by one."
                                )
                                individual_embeddings = [self.embed(text) for text in batch]
                                all_embeddings.extend(individual_embeddings)
                                break
                            else:
                                raise e
                        if self._show_warnings:
                            warnings.warn(f"Batch attempt {attempt + 1} failed: {str(e)}. Retrying...")

            except Exception as e:
                raise RuntimeError(f"Failed to process batch: {str(e)}")

        return all_embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Google's token counting API."""
        try:
            response = self.client.models.count_tokens(
                model=self.model,
                contents=text
            )
            # CountTokensResponse has a total_tokens attribute
            if response.total_tokens is not None:
                return int(response.total_tokens)
            else:
                # Fallback if total_tokens is None
                return int(len(text) / self._chars_per_token)
        except Exception:
            # Fallback to character-based estimation if API call fails
            return int(len(text) / self._chars_per_token)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts using Google's token counting API."""
        return [self.count_tokens(text) for text in texts]

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute cosine similarity between two embeddings."""
        return np.float32(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> Any:
        """Return the token counter function.
        
        Since Gemini doesn't provide a public tokenizer, we return the count_tokens method.
        """
        return self.count_tokens

    def _is_available(self) -> bool:
        """Check if the Google GenAI package is available."""
        return (
            importutil.find_spec("google.genai") is not None
            and importutil.find_spec("numpy") is not None
        )

    def _import_dependencies(self) -> None:
        """Lazy import dependencies for the embeddings implementation."""
        if self._is_available():
            global np, genai, types
            import numpy as np
            from google import genai
            from google.genai import types
        else:
            raise ImportError(
                'One (or more) of the following packages is not available: google-genai, numpy. Please install it via `pip install "chonkie[gemini]"`'
            )

    def __repr__(self) -> str:
        """Representation of the GeminiEmbeddings instance."""
        return f"GeminiEmbeddings(model={self.model}, task_type={self.task_type})"