"""Neural Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from .base import CloudChunker


class NeuralChunker(CloudChunker):
    """Neural Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    DEFAULT_MODEL = "mirth/chonky_distilbert_base_uncased_1"

    SUPPORTED_MODELS = [
        "mirth/chonky_distilbert_base_uncased_1",
        "mirth/chonky_modernbert_base_1",
        "mirth/chonky_modernbert_large_1",
    ]

    SUPPORTED_MODEL_STRIDES = {
        "mirth/chonky_distilbert_base_uncased_1": 256,
        "mirth/chonky_modernbert_base_1": 512,
        "mirth/chonky_modernbert_large_1": 512,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        min_characters_per_chunk: int = 10,
        stride: Optional[int] = None,
        return_type: Literal["texts", "chunks"] = "chunks",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the NeuralChunker."""
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable "
                + "or pass an API key to the NeuralChunker constructor."
            )

        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model} is not supported. Please choose from one of the following: {self.SUPPORTED_MODELS}"
            )
        if min_characters_per_chunk < 1:
            raise ValueError("Minimum characters per chunk must be greater than 0.")
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        self.model = model
        self.min_characters_per_chunk = min_characters_per_chunk
        self.stride = stride if stride is not None else self.SUPPORTED_MODEL_STRIDES[model]
        self.return_type = return_type

        # Check if the Chonkie API is reachable
        try:
            response = requests.get(f"{self.BASE_URL}/")
            if response.status_code != 200:
                raise ValueError(
                    "Oh no! You caught Chonkie at a bad time. It seems to be down right now. Please try again in a short while."
                    + "If the issue persists, please contact support at support@chonkie.ai."
                )
        except Exception as error:
            raise ValueError(
                "Oh no! You caught Chonkie at a bad time. It seems to be down right now. Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from error

    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of chunks."""
        # Create the payload
        payload = {
            "text": text,
            "model": self.model,
            "min_characters_per_chunk": self.min_characters_per_chunk,
            "stride": self.stride,
            "return_type": self.return_type,
        }
        
        # Send the request to the Chonkie API
        try:
            response = requests.post(
                f"{self.BASE_URL}/{self.VERSION}/chunk/neural",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            result: List[Dict] = cast(List[Dict], response.json())
        except Exception as error:
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid response. Please ensure your input is correct and try again. "
                + "If the problem continues, contact support at support@chonkie.ai."
            ) from error
        return result

    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the NeuralChunker."""
        return self.chunk(text)
