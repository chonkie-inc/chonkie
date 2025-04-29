"""Sentence Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from .base import CloudChunker


class SentenceChunker(CloudChunker):
    """Sentence Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(
        self,
        tokenizer_or_token_counter: str = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = True,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        include_delim: Union[Literal["prev", "next"], None] = "prev",
        return_type: Literal["texts", "chunks"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the SentenceChunker."""
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the SentenceChunker constructor."
            )

        # Check if chunk_size and chunk_overlap are valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be greater than or equal to 0.")
        if min_sentences_per_chunk < 1:
            raise ValueError("Minimum sentences per chunk must be greater than 0.")
        if min_characters_per_sentence < 1:
            raise ValueError("Minimum characters per sentence must be greater than 0.")
        if approximate not in [True, False]:
            raise ValueError("Approximate must be either True or False.")
        if include_delim not in ["prev", "next", None]:
            raise ValueError("Include delim must be either 'prev', 'next' or None.")
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        # Assign all the attributes to the instance
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
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
        """Chunk the text via sentence boundaries."""
        # Define the payload for the request
        payload = {
            "text": text,
            "tokenizer": self.tokenizer_or_token_counter,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_sentences_per_chunk": self.min_sentences_per_chunk,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "approximate": self.approximate,
            "delim": self.delim,
            "include_delim": self.include_delim,
            "return_type": self.return_type,
        }

        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/sentence",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Parse the response
        result: List[Dict] = cast(List[Dict], response.json())
        return result

    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the SentenceChunker."""
        return self.chunk(text)
