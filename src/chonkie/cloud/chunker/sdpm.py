"""Semantic Double-Pass Merging for Chonkie API."""

import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.types import SemanticChunk

from .base import CloudChunker


class SDPMChunker(CloudChunker):
    """Semantic Double-Pass Merging for Chonkie API.
    
    This chunker uses the Semantic Double-Pass Merging algorithm to chunk text.

    Args:
        embedding_model: The embedding model to use.
        mode: The mode to use.
        threshold: The threshold to use.
        chunk_size: The chunk size to use.
        similarity_window: The similarity window to use.
        min_sentences: The minimum number of sentences to use.
        min_chunk_size: The minimum chunk size to use.
        min_characters_per_sentence: The minimum number of characters per sentence to use.
        threshold_step: The threshold step to use.
        delim: The delimiters to use.
        include_delim: Whether to include delimiters in chunks.
        skip_window: The skip window to use.
        return_type: The return type to use.
        api_key: The API key to use.
        **kwargs: Additional keyword arguments.
        
    """

    def __init__(self,
                 embedding_model: str = "minishlab/potion-base-8M",
                 mode: str = "window",
                 threshold: Union[str, float, int] = "auto",
                 chunk_size: int = 512,
                 similarity_window: int = 1,
                 min_sentences: int = 1,
                 min_chunk_size: int = 2,
                 min_characters_per_sentence: int = 12,
                 threshold_step: float = 0.01,
                 delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
                 include_delim: Optional[Literal["prev", "next"]] = "prev",
                 skip_window: int = 1,
                 api_key: Optional[str] = None, 
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the SemanticDoublePassMerger.
        
        Args:
            embedding_model: The embedding model to use.
            mode: The mode to use.
            threshold: The threshold to use.
            chunk_size: The chunk size to use.
            similarity_window: The similarity window to use.
            min_sentences: The minimum number of sentences to use.
            min_chunk_size: The minimum chunk size to use.
            min_characters_per_sentence: The minimum number of characters per sentence to use.
            threshold_step: The threshold step to use.
            delim: The delimiters to use.
            include_delim: Whether to include delimiters in chunks.
            skip_window: The skip window to use.
            api_key: The API key to use.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()

        # Get the API key
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the SemanticChunker constructor."
            )

        # Validate all the parameters
        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")

        # Check if the threshold is valid
        if isinstance(threshold, str) and threshold != "auto":
            raise ValueError("Threshold must be either 'auto' or a number between 0 and 1.")

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
        if threshold_step <= 0:
            raise ValueError("Threshold step must be greater than 0.")
        
        # Check if the delimiters are valid
        if not isinstance(delim, list):
            raise ValueError("Delimiters must be a list.")
        
        # Check if the include_delim is valid
        if include_delim not in ["prev", "next"]:
            raise ValueError("Include delim must be either 'prev' or 'next'.")

        # Check if the skip_window is valid
        if skip_window <= 0:
            raise ValueError("Skip window must be greater than 0.")

        # Check if the embedding model is a string
        if not isinstance(embedding_model, str):
            raise ValueError("Embedding model must be a string.")
        
        # Initialize the chunker
        self.embedding_model = embedding_model
        self.mode = mode
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences = min_sentences
        self.min_chunk_size = min_chunk_size
        self.min_characters_per_sentence = min_characters_per_sentence
        self.threshold_step = threshold_step
        self.delim = delim
        self.include_delim = include_delim
        self.skip_window = skip_window


    def chunk(self, text: Union[str, List[str]]) -> Union[List[SemanticChunk], List[List[SemanticChunk]]]:
        """Chunk the text into a list of chunks.
        
        Args:
            text: The text to chunk.
            
        Returns:
            A list of chunks.

        """
        # Make the payload
        payload = {
            "text": text,
            "embedding_model": self.embedding_model,
            "mode": self.mode,
            "threshold": self.threshold,
            "chunk_size": self.chunk_size,
            "similarity_window": self.similarity_window,
            "min_sentences": self.min_sentences,
            "min_chunk_size": self.min_chunk_size,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "threshold_step": self.threshold_step,
            "delim": self.delim,
            "include_delim": self.include_delim,
            "skip_window": self.skip_window,
        }

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
                batch_chunks: List[List[SemanticChunk]] = []
                for chunk_list in batch_result:
                    curr_chunks = []
                    for chunk in chunk_list:
                        curr_chunks.append(SemanticChunk.from_dict(chunk))
                    batch_chunks.append(curr_chunks)
                return batch_chunks
            else:
                single_result: List[Dict] = cast(List[Dict], response.json())
                single_chunks: List[SemanticChunk] = [SemanticChunk.from_dict(chunk) for chunk in single_result]
                return single_chunks
        except Exception as error:
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid response."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai." 
            ) from error

    def __call__(self, text: Union[str, List[str]]) -> Union[List[SemanticChunk], List[List[SemanticChunk]]]:
        """Call the chunker.
        
        Args:
            text: The text to chunk.
            
        Returns:
            A list of chunks.

        """
        return self.chunk(text)
