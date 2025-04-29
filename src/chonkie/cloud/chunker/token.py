"""Cloud Token Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from .base import CloudChunker


class TokenChunker(CloudChunker):
    """Token Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(
        self,
        tokenizer: str,
        chunk_size: int,
        chunk_overlap: int,
        return_type: Literal["texts", "chunks"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Cloud TokenChunker."""
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the TokenChunker constructor."
            )

        # Check if chunk_size and chunk_overlap are valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be greater than or equal to 0.")

        # Check if return_type is valid
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        # Assign all the attributes to the instance
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.return_type = return_type

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai or raise an issue on GitHub."
            )

    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of chunks."""
        # Define the payload for the request
        payload = {
            "text": text,
            "tokenizer": self.tokenizer,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "return_type": self.return_type,
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/token",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError(
                f"Error from the Chonkie API: {response.status_code} {response.text}"
            )

        # Parse the response
        try:
            result: List[Dict] = cast(List[Dict], response.json())
        except Exception as error:
            raise ValueError(f"Error parsing the response: {error}") from error

        # Return the result
        return result

    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the chunker."""
        return self.chunk(text)
