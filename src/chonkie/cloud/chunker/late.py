"""Late Chunking for Chonkie API."""

from typing import Dict, List, Optional, Union, cast

import requests

from chonkie.types import RecursiveRules

from .recursive import RecursiveChunker


class LateChunker(RecursiveChunker):
    """Late Chunking for Chonkie API.

    This class sends requests to the Chonkie API's late chunking endpoint.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        min_characters_per_chunk: int = 24,  # Default from local LateChunker
        rules: RecursiveRules = RecursiveRules(),
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the LateChunker for the Chonkie Cloud API.

        Args:
            embedding_model: The name or identifier of the embedding model to be used by the API.
            chunk_size: The target maximum size of each chunk (in tokens, as defined by the embedding model's tokenizer).
            min_characters_per_chunk: The minimum number of characters a chunk should have.
            rules: The recursive splitting rules to apply before late interaction.
            api_key: The Chonkie API key. If None, it's read from the CHONKIE_API_KEY environment variable.

        """
        self.embedding_model = embedding_model
        # The API key, base URL, version, and API health check are handled by the superclass.
        # Parameters like chunk_size, min_characters_per_chunk, and rules are also set by super().__init__.
        super().__init__(
            api_key=api_key,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
            rules=rules,
            # tokenizer_or_token_counter is required by RecursiveChunker's __init__,
            # but its specific value is less critical here as the late chunking endpoint
            # will primarily use the embedding_model. A generic default is provided.
            tokenizer_or_token_counter="gpt2",
            return_type="chunks",  # Assuming the API returns structured chunk data
        )

    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of late-interaction chunks via the Chonkie API.

        Args:
            text: The text or list of texts to chunk.

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and is expected to contain text, start/end indices, token count, and embedding.

        Raises:
            ValueError: If the API returns an error or an invalid response.

        """
        # Make the payload
        payload = {
            "text": text,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "min_characters_per_chunk": self.min_characters_per_chunk,
            "rules": self.rules.to_dict(),
            # "return_type": self.return_type, # Implicitly "late_chunks" by endpoint
        }

        # Make the request to the Chonkie API's late chunking endpoint
        # Assuming the endpoint is /chunk/late, similar to /chunk/recursive
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/late",  # Assumed new endpoint
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        # Try to parse the response
        try:
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            result: List[Dict] = cast(List[Dict], response.json())
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
            
        return result

    @classmethod
    def from_recipe( # type: ignore
        cls,
        name: Optional[str] = "default",
        lang: Optional[str] = "en",
        path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        min_characters_per_chunk: int = 24,
        api_key: Optional[str] = None,
    ) -> "LateChunker":
        """Create a LateChunker from a recipe.

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The name or identifier of the embedding model to be used by the API.
            chunk_size: The chunk size to use.
            min_characters_per_chunk: The minimum number of characters per chunk.
            api_key: The Chonkie API key.

        Returns:
            LateChunker: The created LateChunker.

        Raises:
            ValueError: If the recipe is invalid or if the recipe is not found.
            
        """
        rules = RecursiveRules.from_recipe(name=name, lang=lang, path=path)
        return cls(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
            api_key=api_key,
        )

    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the LateChunker to chunk text."""
        return self.chunk(text)
