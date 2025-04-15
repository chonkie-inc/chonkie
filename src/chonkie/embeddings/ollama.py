import importlib
import os
import warnings
from typing import List, Optional

import numpy as np

from .base import BaseEmbeddings


class OllamaEmbeddings(BaseEmbeddings):
    """Ollama embeddings implementation for the local ollama serve."""

    DEFAULT_MODEL = "all-minilm"                # a light-weight model

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        endpoint: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        show_warnings: bool = True
    ):
        """Initialize Ollama Embeddings.
        
        Args:
            model: Name of the embedding model available through Ollama to use
            max_retries: Maximum number of retries for failed requests,
            timeout: Timeout in seconds for API requests
            batch_size: Maximum number of texts to embed in one API call
            show_warnings: Whether to show warnings about token usage

        """
        super().__init__()
        if not self.is_available():
            raise ImportError(
                "Ollama package is not available. Please install it via pip."
            )
        else:
            if endpoint is None: endpoint = os.getenv("OLLAMA_ENDPOINT")
            self.is_ollama_running(endpoint)
            from ollama import Client

        # initialize the client
        self.client = Client(
            host=endpoint,
            timeout=timeout
        )
        self.model = model
        self._show_warnings = show_warnings
        self._batch_size = batch_size
        self._max_retries = max_retries

        model_parameters = self.fetch_model()   # pull and get the model parameters
        self._dimension = model_parameters["dimensions"]
        self._context_length = model_parameters["context_length"]
        self._num_ctx = model_parameters["num_ctx"]
        self._tokenizer = self.count_tokens

    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a single text."""
        token_count = self.count_tokens(text)
        if token_count > self._context_length and self._show_warnings:
            warnings.warn(
                f"Text has {token_count} tokens which exceeds the maximum context window of {self._context_length}. "
                "It will be truncated."
            )
        for _ in range(self._max_retries):
            try:
                response = self.client.embed(
                    model=self.model,
                    input=text,
                    truncate=True,  # ensures input stays in the context window
                    options={"num_ctx": max(self._num_ctx, token_count + 100)} 
                )
                # adding a buffer of 100 tokens here as token count is estimated.

                return np.array(response.embeddings[0], dtype=np.float32)
            except Exception as e:
                if self._show_warnings:
                    warnings.warn(
                        f"There was an exception while generating embeddings. Exception: {str(e)}. Retrying..."
                    )

        raise RuntimeError(
            "Unable to generate embeddings through Ollama."
        )

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts using batched calls"""
        if not texts:
            return []
        
        all_embeddings = []

        # process in batches 
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # check token counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self._show_warnings:
                for _, count in zip(batch, token_counts):
                    if count > self._context_length:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's context length of {self._context_length}. "
                            "It will be truncated."
                        )
            try:
                for _ in range(self._max_retries):
                    try:
                        response = self.client.embed(
                            model=self.model,
                            input=batch,
                            truncate=True,  # ensures input stays in the context window
                            options={"num_ctx": max(self._num_ctx, count + 100)} 
                        )

                        embeddings = [
                            np.array(e, dtype=np.float32) for e in response.embeddings
                        ]

                        all_embeddings.extend(embeddings)
                        break
                    except Exception as e:
                        if self._show_warnings:
                            warnings.warn(
                                f"There was an exception while generating embeddings. Exception: {str(e)}. Retrying..."
                            )
            except Exception as e:
                # if the batch processing fails, try one by one
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
        # using token estimation
        # refer: sentence.py -> _estimate_token_counts
        CHARS_PER_TOKEN = 6.0  # Avg. char per token for llama3 is b/w 6-7
        return max(1, len(text) // CHARS_PER_TOKEN)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [self.count_tokens(text) for text in texts] 

    def similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=float
        )
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def get_tokenizer_or_token_counter(self):
        """Returns None till the tokenization part is updated."""
        return self._tokenizer

    def fetch_model(self) -> dict:
        """
        Pull the model from the ollama library and return the required parameters.
        Raises a ValueError if the model is neither present in the ollama hub nor loaded manually.
        """
        try:
            import ollama
            ollama.show(self.model)
        except Exception:
            if self._show_warnings:
                warnings.warn(
                    f"{self.model} is not available locally. Downloading from the Ollama hub. This may take a while."
                )
            try:
                ollama.pull(self.model)    # pull the model from the hub
            except Exception:
                raise ValueError(
                    f"Ollama doesn't provide the given model in the hub. Please make sure the model's ID is correct or load the model manually."
                )

        model_file = self.client.show(self.model)
        model_architecture = model_file.modelinfo["general.architecture"]
        
        context_length = model_file.modelinfo[f"{model_architecture}.context_length"]
        dimensions = model_file.modelinfo[f"{model_architecture}.embedding_length"]
        for param in model_file.parameters.split("\n"):
            key, value = param.split()
            if key == "num_ctx": num_ctx = float(value)
            break

    
        return {
            "context_length": context_length,
            "num_ctx": num_ctx,
            "dimensions": dimensions
        }

        
    @classmethod
    def is_available(cls):
        """Check if the Ollama package is available."""
        return importlib.util.find_spec("ollama") is not None

    @classmethod
    def is_ollama_running(cls, url: str) -> bool:
        """Check if the Ollama server is up and running"""
        try:
            import requests
            response = requests.get(url)
        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama is not running at {url}. Please provide a valid endpoint or make sure the Ollama is up and running at the default/given endpoint. Refer https://github.com/ollama/ollama."
            )
        return response.status_code == 200 and response.text == "Ollama is running"
    
    def __repr__(self) -> str:
        return f"OllamaEmbeddings(model={self.model})"