"""Embeddings Refinery for Chonkie Cloud."""
import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
import requests

from .base import BaseRefinery


class EmbeddingsRefinery(BaseRefinery):
    """Embeddings Refinery for Chonkie Cloud."""

    def __init__(self, embedding_model: str = "minishlab/potion-retrieval-32M", api_key: Optional[str] = None):
        """Initialize the EmbeddingsRefinery.

        Args:
            embedding_model: The embedding model to use.
            api_key: Your Chonkie Cloud API Key.

        """
        super().__init__()

        # Get the API key
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        self.embedding_model = embedding_model
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable "
                + "or pass an API key to the EmbeddingsRefinery constructor."
            )

    def refine(self, chunks: List[Any]) -> List[Any]:
        """Refine the chunks.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        Raises:
            ValueError: If all chunks are not of the same type.

        """
        # Define the payload for the request
        if any(type(chunk) != type(chunks[0]) for chunk in chunks):
            raise ValueError("All chunks must be of the same type.")
        og_type = type(chunks[0])
        payload = {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "embedding_model": self.embedding_model,
        }

        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/refine/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Parse the response
        result: List[Dict] = cast(List[Dict], response.json())
        # Take out the embeddings from each chunk
        embeddings = [chunk.pop("embedding") for chunk in result]
        # Convert the chunks back to their original type
        result_chunks = [og_type.from_dict(chunk) for chunk in result]
        # Add the embeddings back to the chunks
        for chunk, embedding in zip(result_chunks, embeddings):
            chunk.embedding = np.array(embedding)
        return result_chunks

    def __call__(self, chunks: List[Any]) -> List[Any]:
        """Call the EmbeddingsRefinery."""
        return self.refine(chunks)
