"""Semantic Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from .base import CloudChunker


class SemanticChunker(CloudChunker):
    """Semantic Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(
        self,
        embedding_model: str = "minishlab/potion-base-32M",
        chunk_size: int = 512,
        threshold: Union[Literal["auto"], float, int] = "auto",
        similarity_window: int = 1,
        min_sentences: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        return_type: Literal["chunks", "texts"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Chonkie Cloud Semantic Chunker."""
        # Get the API key
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the SemanticChunker constructor."
            )

        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")

        # Check if the threshold is valid
        if isinstance(threshold, str) and threshold != "auto":
            raise ValueError(
                "Threshold must be either 'auto' or a number between 0 and 1."
            )
        elif isinstance(threshold, (float, int)) and (threshold <= 0 or threshold > 1):
            raise ValueError("Threshold must be between 0 and 1.")

        # Check if the similarity window is valid
        if similarity_window <= 0:
            raise ValueError("Similarity window must be greater than 0.")

        # Check if the minimum sentences is valid
        if min_sentences <= 0:
            raise ValueError("Minimum sentences must be greater than 0.")

        # Check if the minimum chunk size is valid
        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be greater than 0.")

        # Check if the minimum characters per sentence is valid
        if min_characters_per_sentence <= 0:
            raise ValueError("Minimum characters per sentence must be greater than 0.")

        # Check if the threshold step is valid
        if threshold_step <= 0 or threshold_step > 1:
            raise ValueError(
                "Threshold step must be greater than 0 and less than or equal to 1."
            )

        # Check if the delim is valid
        if not isinstance(delim, (list, str)):
            raise ValueError("Delim must be a list or a string.")

        # Check if the include delim is valid
        if include_delim not in ["prev", "next", None]:
            raise ValueError("Include delim must be either 'prev', 'next', or None.")

        # Check if the return type is valid
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Return type must be either 'chunks' or 'texts'.")

        # Add all the attributes
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.similarity_window = similarity_window
        self.min_sentences = min_sentences
        self.min_chunk_size = min_chunk_size
        self.min_characters_per_sentence = min_characters_per_sentence
        self.threshold_step = threshold_step
        self.delim = delim
        self.include_delim = include_delim
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
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "threshold": self.threshold,
            "similarity_window": self.similarity_window,
            "min_sentences": self.min_sentences,
            "min_chunk_size": self.min_chunk_size,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "threshold_step": self.threshold_step,
            "delim": self.delim,
            "include_delim": self.include_delim,
            "return_type": self.return_type,
        }

        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/semantic",
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
        """Call the chunker."""
        return self.chunk(text)
