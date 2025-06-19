"""Overlap Refinery for Chonkie Cloud."""
import os
from typing import Any, Dict, List, Literal, Optional, Union, cast

import requests

from .base import BaseRefinery


class OverlapRefinery(BaseRefinery):
    """Overlap Refinery for Chonkie Cloud."""

    def __init__(self,         
        tokenizer_or_token_counter: str = "gpt2",
        context_size: Union[int, float] = 0.25,
        mode: Literal["token", "recursive"] = "token",
        method: Literal["suffix", "prefix"] = "suffix",
        recipe: str = "default",
        lang: str = "en",
        merge: bool = True,
        api_key: Optional[str] = None,
    ):
        """Initialize the OverlapRefinery.

        Args:
            tokenizer_or_token_counter: The tokenizer or token counter to use.
            context_size: The context size to use. Must be a value between 0 and 1 for token mode and an integer for recursive mode.
            mode: The mode to use.
            method: The method to use.
            recipe: The name of the recursive rules recipe to use. Find all available recipes at https://hf.co/datasets/chonkie-ai/recipes
            lang: The language of the recipe. Please make sure a valide recipe with the given `recipe` value and `lang` values exists on https://hf.co/datasets/chonkie-ai/recipes
            merge: Whether to merge the chunks.
            api_key: Your Chonkie Cloud API Key.

        """
        super().__init__()

        # Get the API key
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.context_size = context_size
        self.mode = mode
        self.method = method
        self.recipe = recipe
        self.lang = lang
        self.merge = merge
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please set the CHONKIE_API_KEY environment variable"
                + "or pass an API key to the OverlapRefinery constructor."
            )

    def refine(self, chunks: List[Any]) -> List[Any]:
        """Refine the chunks.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            Chunks with overlap.

        Raises:
            ValueError: If all chunks are not of the same type.

        """
        # Define the payload for the request
        if any(type(chunk) != type(chunks[0]) for chunk in chunks):
            raise ValueError("All chunks must be of the same type.")
        og_type = type(chunks[0])
        payload = {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
            "context_size": self.context_size,
            "mode": self.mode,
            "method": self.method,
            "recipe": self.recipe,
            "lang": self.lang,
            "merge": self.merge,
        }

        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/refine/overlap",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Parse the response
        result: List[Dict] = cast(List[Dict], response.json())
        result_chunks = [og_type.from_dict(chunk) for chunk in result]
        return result_chunks

    def __call__(self, chunks: List[Any]) -> List[Any]:
        """Call the OverlapRefinery."""
        return self.refine(chunks)
