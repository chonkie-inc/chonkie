"""Recursive Chunking for Chonkie API."""

import os
from typing import Any, Callable, Dict, List, Optional, Union, cast

import requests

from chonkie.cloud.file import FileManager
from chonkie.types import Chunk

from .base import CloudChunker


class RecursiveChunker(CloudChunker):
    """Recursive Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable] = "gpt2",
        chunk_size: int = 512,
        min_characters_per_chunk: int = 12,
        recipe: str = "default",
        lang: str = "en",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the RecursiveChunker.
        
        Args:
            tokenizer_or_token_counter: The tokenizer or token counter to use.
            chunk_size: The target maximum size of each chunk (in tokens, as defined by the tokenizer).
            min_characters_per_chunk: The minimum number of characters a chunk should have.
            recipe: The name of the recursive rules recipe to use. Find all available recipes at https://hf.co/datasets/chonkie-ai/recipes
            lang: The language of the recipe. Please make sure a valid recipe with the given `recipe` value and `lang` values exists on https://hf.co/datasets/chonkie-ai/recipes
            api_key: The Chonkie API key. If None, it's read from the CHONKIE_API_KEY environment variable.

        """
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

        # Add attributes
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.recipe = recipe
        self.lang = lang

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            )

        # Initialize the file manager to upload files if needed
        self.file_manager = FileManager(api_key=self.api_key)

    def chunk(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Any:
        """Chunk the text or file into a list of chunks."""
        # Make the payload
        payload: Dict[str, Any]
        if text is not None:
            payload = {
                "text": text,
                "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
                "chunk_size": self.chunk_size,
                "min_characters_per_chunk": self.min_characters_per_chunk,
                "recipe": self.recipe,
                "lang": self.lang,
            }
        elif file is not None:
            file_response = self.file_manager.upload(file)
            payload = {
                "file": {
                    "type": "document",
                    "content": file_response.name,
                },
                "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
                "chunk_size": self.chunk_size,
                "min_characters_per_chunk": self.min_characters_per_chunk,
                "recipe": self.recipe,
                "lang": self.lang,
            }
        else:
            raise ValueError("No text or file provided. Please provide either text or a file path.")
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/recursive",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Try to parse the response
        try:
            if isinstance(text, list):
                batch_result: List[List[Dict]] = cast(List[List[Dict]], response.json())
                batch_chunks: List[List[Chunk]] = []
                for chunk_list in batch_result:
                    curr_chunks = []
                    for chunk in chunk_list:
                        curr_chunks.append(Chunk.from_dict(chunk))
                    batch_chunks.append(curr_chunks)
                return batch_chunks
            else:
                single_result: List[Dict] = cast(List[Dict], response.json())
                single_chunks: List[Chunk] = [Chunk.from_dict(chunk) for chunk in single_result]
                return single_chunks
        except Exception as error:
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid response."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from error

    def __call__(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Any:
        """Call the RecursiveChunker."""
        return self.chunk(text=text, file=file)
