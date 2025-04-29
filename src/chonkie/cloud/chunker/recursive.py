"""Recursive Chunking for Chonkie API."""

import os
from typing import Callable, Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.types import RecursiveRules

from .base import CloudChunker


class RecursiveChunker(CloudChunker):
    """Recursive Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable] = "gpt2",
        chunk_size: int = 512,
        min_characters_per_chunk: int = 12,
        rules: RecursiveRules = RecursiveRules(),
        return_type: Literal["texts", "chunks"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the RecursiveChunker."""
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the RecursiveChunker constructor."
            )

        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if min_characters_per_chunk < 1:
            raise ValueError("Minimum characters per chunk must be greater than 0.")
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        # Add attributes
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.rules = rules
        self.return_type = return_type

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            )

    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of chunks."""
        # Make the payload
        payload = {
            "text": text,
            "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
            "chunk_size": self.chunk_size,
            "min_characters_per_chunk": self.min_characters_per_chunk,
            "rules": self.rules.to_dict(),
            "return_type": self.return_type,
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/recursive",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Try to parse the response
        try:
            result: List[Dict] = cast(List[Dict], response.json())
        except Exception as error:
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid response."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from error

        return result

    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the RecursiveChunker."""
        return self.chunk(text)
