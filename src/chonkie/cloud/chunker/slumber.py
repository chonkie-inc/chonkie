"""Slumber Chunking for Chonkie API."""

import os
from typing import Callable, Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.types import RecursiveRules

from .base import CloudChunker


class SlumberChunker(CloudChunker):
    """Slumber Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable] = "gpt2",
        chunk_size: int = 1024,
        rules: RecursiveRules = RecursiveRules(),
        candidate_size: int = 128,
        min_characters_per_chunk: int = 24,
        return_type: Literal["texts", "chunks"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the SlumberChunker.

        Args:
            tokenizer_or_token_counter (Union[str, Callable]): The tokenizer or token counter to use.
            chunk_size (int): The target size of the chunks.
            rules (RecursiveRules): The rules to use to split the candidate chunks.
            candidate_size (int): The size of the candidate splits that the chunker will consider.
            min_characters_per_chunk (int): The minimum number of characters per chunk.
            return_type (Literal["texts", "chunks"]): The type of output to return.
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
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.rules = rules
        self.candidate_size = candidate_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.return_type = return_type

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


    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of chunks using the Slumber strategy via API.

        Args:
            text (Union[str, List[str]]): The text or list of texts to chunk.

        Returns:
            List[Dict]: A list of dictionaries representing the chunks or texts.

        """
        payload = {
            "text": text,
            "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
            "chunk_size": self.chunk_size,
            "rules": self.rules.to_dict(), # Assuming RecursiveRules has a to_dict method
            "candidate_size": self.candidate_size,
            "min_characters_per_chunk": self.min_characters_per_chunk,
            "return_type": self.return_type,
        }

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
            # The content of these dictionaries depends on the 'return_type' specified in the payload.
            result: List[Dict] = cast(List[Dict], response.json())
            return result
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


    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the SlumberChunker."""
        return self.chunk(text)

    def __repr__(self) -> str:
        """Return a string representation of the SlumberChunker."""
        return (
            f"SlumberChunker(api_key={'********' if self.api_key else None}, "
            f"tokenizer_or_token_counter='{self.tokenizer_or_token_counter}', "
            f"chunk_size={self.chunk_size}, "
            f"rules={self.rules}, "
            f"candidate_size={self.candidate_size}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk}, "
            f"return_type='{self.return_type}')"
        )
