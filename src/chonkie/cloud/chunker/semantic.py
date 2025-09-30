"""Semantic Chunking for Chonkie API."""

import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.cloud.file import FileManager
from chonkie.types import Chunk

from .base import CloudChunker


class SemanticChunker(CloudChunker):
    """Semantic Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    def __init__(
        self,
        embedding_model: str = "minishlab/potion-base-32M",
        threshold: float = 0.8,
        chunk_size: int = 512,
        similarity_window: int = 1,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the Chonkie Cloud Semantic Chunker."""
        super().__init__()
        
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
        if threshold <= 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1.")

        # Check if the similarity window is valid
        if similarity_window <= 0:
            raise ValueError("Similarity window must be greater than 0.")

        # Check if the minimum sentences is valid
        if min_sentences_per_chunk <= 0:
            raise ValueError("Minimum sentences must be greater than 0.")

        # Check if the minimum characters per sentence is valid
        if min_characters_per_sentence <= 0:
            raise ValueError("Minimum characters per sentence must be greater than 0.")

        # Check if the skip window is valid
        if skip_window < 0:
            raise ValueError("Skip window must be greater than or equal to 0.")

        # Check if the filter window is valid
        if filter_window <= 0:
            raise ValueError("Filter window must be greater than 0.")

        # Check if the filter polyorder is valid
        if filter_polyorder < 0 or filter_polyorder >= filter_window:
            raise ValueError(
                "Filter polyorder must be greater than 0 and less than or equal to filter window."
            )

        # Check if the filter tolerance is valid
        if filter_tolerance <= 0 or filter_tolerance >= 1:
            raise ValueError("Filter tolerance must be between 0 and 1.")

        # Check if the delim is valid
        if not isinstance(delim, (list, str)):
            raise ValueError("Delim must be a list or a string.")

        # Check if the include delim is valid
        if include_delim not in ["prev", "next", None]:
            raise ValueError("Include delim must be either 'prev', 'next', or None.")


        # Add all the attributes
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.skip_window = skip_window
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.filter_tolerance = filter_tolerance
        self.delim = delim
        self.include_delim = include_delim

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

    def chunk(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Chunk the text or file into a list of chunks."""
        # Make the payload
        payload: Dict[str, Any]
        if text is not None:
            payload = {
                "text": text,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "threshold": self.threshold,
                "similarity_window": self.similarity_window,
                "min_sentences_per_chunk": self.min_sentences_per_chunk,
                "min_characters_per_sentence": self.min_characters_per_sentence,
                "skip_window": self.skip_window,
                "filter_window": self.filter_window,
                "filter_polyorder": self.filter_polyorder,
                "filter_tolerance": self.filter_tolerance,
                "delim": self.delim,
                "include_delim": self.include_delim,
            }
        elif file is not None:
            file_response = self.file_manager.upload(file)
            payload = {
                "file": {
                    "type": "document",
                    "content": file_response.name,
                },
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "threshold": self.threshold,
                "similarity_window": self.similarity_window,
                "min_sentences_per_chunk": self.min_sentences_per_chunk,
                "min_characters_per_sentence": self.min_characters_per_sentence,
                "skip_window": self.skip_window,
                "filter_window": self.filter_window,
                "filter_polyorder": self.filter_polyorder,
                "filter_tolerance": self.filter_tolerance,
                "delim": self.delim,
                "include_delim": self.include_delim,
            }
        else:
            raise ValueError("No text or file provided. Please provide either text or a file path.")

        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/semantic",
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

    def __call__(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Call the chunker."""
        return self.chunk(text=text, file=file)
