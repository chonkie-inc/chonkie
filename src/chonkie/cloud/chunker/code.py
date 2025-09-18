"""Code Chunking for Chonkie API."""

import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.cloud.file import FileManager
from chonkie.types import Chunk

from .base import CloudChunker


class CodeChunker(CloudChunker):
    """Code Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        tokenizer_or_token_counter: str = "gpt2",
        chunk_size: int = 512,
        language: Union[Literal["auto"], str] = "auto",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Cloud CodeChunker.
        
        Args:
            tokenizer_or_token_counter: The tokenizer or token counter to use.
            chunk_size: The size of the chunks to create.
            language: The language of the code to parse. Accepts any of the languages supported by tree-sitter-language-pack.
            api_key: The API key for the Chonkie API.
            
        Raises:
            ValueError: If the API key is not provided or if parameters are invalid.

        """
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + " or pass an API key to the CodeChunker constructor."
            )

        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")

        # Assign all the attributes to the instance
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.language = language

        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now."
                + " Please try again in a short while."
                + " If the issue persists, please contact support at support@chonkie.ai or raise an issue on GitHub."
            )

        # Initialize the file manager to upload files if needed
        self.file_manager = FileManager(api_key=self.api_key)

    def chunk(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Chunk the code into a list of chunks.

        Args:
            text: The code text(s) to chunk.
            file: The path to a file to chunk.

        Returns:
            A list of Chunk objects containing the chunked code.

        Raises:
            ValueError: If the API request fails or returns invalid data.

        """
        # Define the payload for the request
        payload: Dict[str, Any]
        if text is not None:
            payload = {
                "text": text,
                "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
                "chunk_size": self.chunk_size,
                "language": self.language,
                "lang": self.language, # For backward compatibility
                "include_nodes": False,  # API doesn't support tree-sitter nodes
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
                "language": self.language,
                "lang": self.language, # For backward compatibility
                "include_nodes": False,  # API doesn't support tree-sitter nodes
            }
        else:
            raise ValueError("No text or file provided. Please provide either text or a file path.")
        
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/code",
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
            raise ValueError(f"Error parsing the response: {error}") from error

    def __call__(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Call the chunker."""
        return self.chunk(text=text, file=file)