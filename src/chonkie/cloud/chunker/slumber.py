"""Slumber Chunking for Chonkie API."""

import os
from typing import Any, Callable, Dict, List, Optional, Union, cast

import requests

from chonkie.cloud.file import FileManager
from chonkie.types import Chunk

from .base import CloudChunker


class SlumberChunker(CloudChunker):
    """Slumber Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable] = "gpt2",
        chunk_size: int = 1024,
        recipe: str = "default",
        lang: str = "en",
        candidate_size: int = 128,
        min_characters_per_chunk: int = 24,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the SlumberChunker.

        Args:
            tokenizer_or_token_counter (Union[str, Callable]): The tokenizer or token counter to use.
            chunk_size (int): The target size of the chunks.
            recipe (str): The recipe to use.
            lang (str): The language to use.
            candidate_size (int): The size of the candidate splits that the chunker will consider.
            min_characters_per_chunk (int): The minimum number of characters per chunk.
            api_key (Optional[str]): The Chonkie API key.

        """
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + " or pass an API key to the SlumberChunker constructor."
            )

        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if candidate_size <= 0:
            raise ValueError("Candidate size must be greater than 0.")
        if min_characters_per_chunk < 1:
            raise ValueError("Minimum characters per chunk must be greater than 0.")

        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.recipe = recipe
        self.lang = lang
        self.candidate_size = candidate_size
        self.min_characters_per_chunk = min_characters_per_chunk

        # Check if the API is up
        try:
            response = requests.get(f"{self.BASE_URL}/")
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        except requests.exceptions.RequestException as error:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down or unreachable."
                + " Please try again in a short while."
                + " If the issue persists, please contact support at support@chonkie.ai."
            ) from error

        # Initialize the file manager to upload files if needed
        self.file_manager = FileManager(api_key=self.api_key)


    def chunk(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Chunk the text or file into a list of chunks using the Slumber strategy via API.

        Args:
            text (Union[str, List[str]]): The text or list of texts to chunk.
            file (Optional[str]): The path to a file to chunk.

        Returns:
            List[Dict]: A list of dictionaries representing the chunks or texts.

        """
        payload: Dict[str, Any]
        if text is not None:
            payload = {
                "text": text,
                "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
                "chunk_size": self.chunk_size,
                "recipe": self.recipe,
                "lang": self.lang,
                "candidate_size": self.candidate_size,
                "min_characters_per_chunk": self.min_characters_per_chunk,
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
                "recipe": self.recipe,
                "lang": self.lang,
                "candidate_size": self.candidate_size,
                "min_characters_per_chunk": self.min_characters_per_chunk,
            }
        else:
            raise ValueError("No text or file provided. Please provide either text or a file path.")

        try:
            response = requests.post(
                f"{self.BASE_URL}/{self.VERSION}/chunk/slumber",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.exceptions.RequestException as e:
            # More specific error message including potential response text for debugging
            error_message = (
                "Oh no! The Chonkie API returned an error while trying to chunk with Slumber."
                + " Please try again in a short while."
            )
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f" Details: {error_detail}"
                except ValueError: # if response is not JSON
                    error_message += f" Status Code: {e.response.status_code}. Response: {e.response.text}"
            error_message += " If the issue persists, please contact support at support@chonkie.ai."
            raise ValueError(error_message) from e


        try:
            # Assuming the API always returns a list of dictionaries.
            if isinstance(text, list):
                batch_result: List[List[Dict]] = cast(List[List[Dict]], response.json())
                batch_chunks: List[List[Chunk]] = []
                for chunk_list in batch_result:
                    curr_chunks: List[Chunk] = []
                    for chunk in chunk_list:
                        curr_chunks.append(Chunk.from_dict(chunk))
                    batch_chunks.append(curr_chunks)
                return batch_chunks
            else:
                single_result: List[Dict] = cast(List[Dict], response.json())
                single_chunks: List[Chunk] = [Chunk.from_dict(chunk) for chunk in single_result]
                return single_chunks
        except ValueError as error: # JSONDecodeError inherits from ValueError
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid JSON response for Slumber chunking."
                + " Please try again in a short while."
                + f" Response text: {response.text}"
                + " If the issue persists, please contact support at support@chonkie.ai."
            ) from error
        except Exception as error: # Catch any other parsing/validation errors
            raise ValueError(
                "Oh no! Failed to parse the response from Chonkie API for Slumber chunking."
                + " Please try again in a short while."
                + f" Details: {str(error)}"
                + " If the issue persists, please contact support at support@chonkie.ai."
            ) from error


    def __call__(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Call the SlumberChunker."""
        return self.chunk(text=text, file=file)

    def __repr__(self) -> str:
        """Return a string representation of the SlumberChunker."""
        return (
            f"SlumberChunker(api_key={'********' if self.api_key else None}, "
            f"tokenizer_or_token_counter='{self.tokenizer_or_token_counter}', "
            f"chunk_size={self.chunk_size}, "
            f"recipe='{self.recipe}', "
            f"lang='{self.lang}', "
            f"candidate_size={self.candidate_size}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk})"
        )
