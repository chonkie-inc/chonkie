"""Late Chunking for Chonkie API."""

from typing import Any, Dict, List, Optional, Union, cast

import requests

from chonkie.types import Chunk

from .recursive import RecursiveChunker


class LateChunker(RecursiveChunker):
    """Late Chunking for Chonkie API.

    This class sends requests to the Chonkie API's late chunking endpoint.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        min_characters_per_chunk: int = 24,
        recipe: str = "default",
        lang: str = "en",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the LateChunker for the Chonkie Cloud API.

        Args:
            embedding_model: The name or identifier of the embedding model to be used by the API.
            chunk_size: The target maximum size of each chunk (in tokens, as defined by the embedding model's tokenizer).
            min_characters_per_chunk: The minimum number of characters a chunk should have.
            recipe: The name of the recursive rules recipe to use. Find all available recipes at https://hf.co/datasets/chonkie-ai/recipes
            lang: The language of the recipe. Please make sure a valid recipe with the given `recipe` value and `lang` values exists on https://hf.co/datasets/chonkie-ai/recipes
            api_key: The Chonkie API key. If None, it's read from the CHONKIE_API_KEY environment variable.

        """
        self.embedding_model = embedding_model
        super().__init__(
            api_key=api_key,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
            recipe=recipe,
            lang=lang,
        )

    def chunk(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Chunk the text or file into a list of late-interaction chunks via the Chonkie API.

        Args:
            text: The text or list of texts to chunk.
            file: The path to a file to chunk.

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and is expected to contain text, start/end indices, token count, and embedding.

        Raises:
            ValueError: If the API returns an error or an invalid response.

        """
        # Make the payload
        payload: Dict[str, Any]
        if text is not None:
            payload = {
                "text": text,
                "embedding_model": self.embedding_model,
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
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "min_characters_per_chunk": self.min_characters_per_chunk,
                "recipe": self.recipe,
                "lang": self.lang,
            }
        else:
            raise ValueError("No text or file provided. Please provide either text or a file path.")

        # Make the request to the Chonkie API's late chunking endpoint
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/late",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Try to parse the response
        try:
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
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
        except requests.exceptions.HTTPError as http_error:
            # Attempt to get more detailed error from API response if possible
            error_detail = ""
            try:
                error_detail = response.json().get("detail", "")
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text
            raise ValueError(
                f"Oh no! Chonkie API returned an error for late chunking: {http_error}. "
                f"Details: {error_detail}"
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from http_error
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(
                "Oh no! The Chonkie API returned an invalid JSON response for late chunking."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from error
        except Exception as error: # Catch any other unexpected errors
            raise ValueError(
                "An unexpected error occurred while processing the response from Chonkie API for late chunking."
                + "Please try again in a short while."
                + "If the issue persists, please contact support at support@chonkie.ai."
            ) from error
    
    def __call__(self, text: Optional[Union[str, List[str]]] = None, file: Optional[str] = None) -> Union[List[Chunk], List[List[Chunk]]]:
        """Call the LateChunker to chunk text."""
        return self.chunk(text=text, file=file)
