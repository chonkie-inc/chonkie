"""Module for Jina AI embeddings integration."""

import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import requests

if TYPE_CHECKING:
    try: 
        import numpy as np
        from tokenizers import Tokenizer
    except ImportError:
        np = Any # type: ignore
        Tokenizer = Any # type: ignore

from .base import BaseEmbeddings


class JinaEmbeddings(BaseEmbeddings):
    """Jina embeddings implementation using their API."""

    AVAILABLE_MODELS = {
        "jina-embeddings-v3": 1024,
        "jina-embeddings-v2-base-en": 768,
        "jina-embeddings-v2-base-es": 768,
        "jina-embeddings-v2-base-de": 768,
        "jina-embeddings-v2-base-zh": 768,   
        "jina-embeddings-v2-base-code": 768,
        "jina-embeddings-b-en-v1": 768
    }   

    def __init__(
            self,
            model: str = "jina-embeddings-v3",
            task: str = "text-matching",
            batch_size: int = 32,
            max_retries: int = 3,
            api_key: Optional[str] = None
    ):
        """Initialize Jina embeddings.

        Args:
            model (str): Name of the Jina embedding model to use.
            task (str): Task for the Jina model.
            late_chunking (bool): Whether to use late chunking.
            embedding_type (str): Type of the embedding.
            api_key (Optional[str]): Jina API key (if not provided, looks for
                JINA_API_KEY env var).
            batch_size (int): Maximum number of texts to embed in one API call.
            max_retries (int): Maximum number of retries for API calls.

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        # Check if the API key is provided
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key is required. Provide via api_key parameter or JINA_API_KEY environment variable")

        # Initialize the Jina embeddings model
        self.model = model
        self.task = task
        self._dimension = self.AVAILABLE_MODELS[model]
        self.embedding_type = "float"
        self.late_chunking = False # Set to False since we don't need it! Chonkie can handle this!
        self._batch_size = batch_size
        self._max_retries = max_retries
        try:
            self._tokenizer = Tokenizer.from_pretrained(f'jinaai/{model}')
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model {model}: {e}")
        
        # Initialize the URL for the API request
        self.url = 'https://api.jina.ai/v1/embeddings'
        
        # Initialize the headers for the API request
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _is_available(self) -> bool:
        """Check if the Jina package is available."""
        return (
            importutil.find_spec("numpy") is not None
            and importutil.find_spec("tokenizers") is not None
        )

    def _import_dependencies(self) -> None:
        """Lazy import dependencies if they are not already imported.""" 
        if self._is_available():
            global np, Tokenizer
            import numpy as np
            from tokenizers import Tokenizer
        else:
            raise ImportError("One (or more) of the following packages is not available: numpy, tokenizers." +
             " Please install it via `pip install chonkie[jina]`")
    
    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text using the Jina embeddings API.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: Numpy array with the embedding for the text.

        Raises:
            ValueError: If the input `text` is empty or the API response is unexpected.
            requests.exceptions.RequestException: If the API request fails after retries.

        """
        if not text:
            raise ValueError("Input text cannot be empty")

        data = {
            "model": self.model,
            "task": self.task,
            "late_chunking": self.late_chunking,
            "embedding_type": self.embedding_type,
            "input": [text]  # API expects a list, even for single text
        }

        for attempt in range(self._max_retries):
            try:
                response = requests.post(self.url, json=data, headers=self.headers)
                response.raise_for_status()
                vector = response.json()
                response_data = vector.get('data')
                if not response_data or not response_data[0] or 'embedding' not in response_data[0]:
                    raise ValueError(f"Unexpected API response format: {vector}")
                # Assuming the API returns a list with one embedding
                return np.array(response_data[0]['embedding'], dtype=np.float32)
            except requests.exceptions.RequestException as e:
                if attempt == self._max_retries - 1:
                    # Raise a more informative error including the text that failed
                    raise ValueError(f"Failed to embed text '{text[:50]}...' after {self._max_retries} attempts due to: {e}")
                warnings.warn(f"Attempt {attempt + 1} failed for text '{text[:50]}...': {str(e)}. Retrying...")

        # This point should theoretically not be reached if max_retries > 0,
        # as the loop either returns successfully or raises an exception on the last attempt.
        # Adding a fallback raise to satisfy linters and catch unexpected scenarios.
        raise RuntimeError(f"Embedding failed for text '{text[:50]}...' after multiple retries, but no exception was raised.")

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts using the Jina embeddings API.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List["np.ndarray"]: List of numpy arrays with embeddings for each text.

        Raises:
            requests.exceptions.HTTPError: If the initial API request for a batch fails
                and the batch contained only one text.
            ValueError: If the API response format is unexpected, or if the fallback
                to single embedding fails for a text within a failed batch.
            requests.exceptions.RequestException: If an API request fails after all retries
                (either batch or single fallback).

        """
        if not texts:
            return []
            
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            payload = {
                "model": self.model,
                "task": self.task,
                "late_chunking": self.late_chunking,
                "embedding_type": self.embedding_type,
                "input": batch
            }
            
            try:
                response = requests.post(self.url, json=payload, headers=self.headers)
                response.raise_for_status()
                response_data = response.json()
                embeddings = [
                    np.array(item['embedding'], dtype=np.float32) 
                    for item in response_data['data']
                ]
                all_embeddings.extend(embeddings)
            except requests.exceptions.HTTPError as e:
                if len(batch) > 1:
                    warnings.warn(f"Failed to embed batch: {batch} due to: {e}. Falling back to sequential embedding texts.")
                    # Fall back to single embeddings
                    single_embeddings = []
                    for text in batch:
                        if isinstance(text, str):
                            single_embeddings.append(self.embed(text))
                        else:
                            raise ValueError(f"Invalid text type found in batch: {type(text)}")
                    all_embeddings.extend(single_embeddings)
                else:
                    raise ValueError(f"Failed to embed text: {batch} due to: {e}")                    
        return all_embeddings

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute cosine similarity between two embeddings.
        
        Args:
            u (np.ndarray): First embedding vector.
            v (np.ndarray): Second embedding vector.

        Returns:
            np.float32: Cosine similarity between u and v.

        """
        return np.float32(np.divide(np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v)))
    
    @property
    def dimension(self) -> int:
        """Return the dimensions of the embeddings.

        Returns:
            int: The embedding dimension size.

        """
        return self._dimension
        
    def get_tokenizer_or_token_counter(self) -> "Tokenizer":
        """Get the tokenizer instance used by the embeddings model.

        Returns:
            Tokenizer: A Tokenizer instance for the Jina embeddings model.

        """
        return self._tokenizer      

    def __repr__(self) -> str:
        """Return a string representation of the JinaEmbeddings instance.

        Returns:
            str: A string representation of the instance.

        """
        return f"JinaEmbeddings(model={self.model}, dimensions={self._dimension})"