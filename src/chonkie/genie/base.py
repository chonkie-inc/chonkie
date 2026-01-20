"""BaseGenie is the base class for all genies."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseGenie(ABC):
    """Abstract Base Class for Genies.

    This class defines the common interface for all Genie implementations. Genies
    are responsible for generating text or structured JSON responses based on
    input prompts.

    Subclasses must implement the `generate` method. If structured JSON output
    is required, subclasses should also implement the `generate_json` method.
    The batch generation methods (`generate_batch`, `generate_json_batch`) are
    provided for convenience and typically do not need to be overridden.

    Methods:
        generate(prompt: str) -> str:
            Generates a text response for a single prompt. Must be implemented by subclasses.
        generate_async(prompt: str) -> str:
            Generates a text response for a single prompt asynchronously. Uses `generate`.
        generate_batch(prompts: list[str]) -> list[str]:
            Generates text responses for a batch of prompts. Uses `generate`.
        generate_batch_async(prompts: list[str]) -> list[str]:
            Generates text responses for a batch of prompts asynchronously. Uses `generate_batch`.
        generate_json(prompt: str, schema: Any) -> Any:
            Generates a structured JSON response conforming to the provided schema
            for a single prompt. Should be implemented by subclasses if JSON output
            is needed.
        generate_json_async(prompt: str, schema: Any) -> Any:
            Generates a structured JSON response conforming to the provided schema
            for a single prompt asynchronously. Uses `generate_json`.
        generate_json_batch(prompts: list[str], schema: Any) -> list[Any]:
            Generates structured JSON responses for a batch of prompts. Uses `generate_json`.
        generate_json_batch_async(prompts: list[str], schema: Any) -> list[Any]:
            Generates structured JSON responses for a batch of prompts asynchronously. Uses `generate_json_batch`.

    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        raise NotImplementedError

    async def generate_async(self, prompt: str) -> str:
        """Generate a response asynchronously."""
        return await asyncio.to_thread(self.generate, prompt)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate a batch of responses based on the given prompts."""
        return [self.generate(prompt) for prompt in prompts]

    async def generate_batch_async(
        self, prompts: list[str], max_concurrency: int = 10
    ) -> list[str]:
        """Generate a batch of responses asynchronously.
        
        Args:
            prompts: List of prompts to generate responses for.
            max_concurrency: Maximum number of concurrent requests. Defaults to 10.
            
        Returns:
            list[str]: List of generated responses.

        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(prompt)

        return await asyncio.gather(*[_bounded_generate(prompt) for prompt in prompts])

    def generate_json(self, prompt: str, schema: Any) -> Any:
        """Generate a JSON response based on the given prompt and BaseModel schema."""
        raise NotImplementedError

    async def generate_json_async(self, prompt: str, schema: Any) -> Any:
        """Generate a JSON response asynchronously."""
        return await asyncio.to_thread(self.generate_json, prompt, schema)

    def generate_json_batch(self, prompts: list[str], schema: Any) -> list[Any]:
        """Generate a batch of JSON responses based on the given prompts and BaseModel schema."""
        return [self.generate_json(prompt, schema) for prompt in prompts]

    async def generate_json_batch_async(
        self, prompts: list[str], schema: Any, max_concurrency: int = 10
    ) -> list[Any]:
        """Generate a batch of JSON responses asynchronously.
        
        Args:
            prompts: List of prompts to generate responses for.
            schema: The schema for the JSON response.
            max_concurrency: Maximum number of concurrent requests. Defaults to 10.
            
        Returns:
            list[Any]: List of generated JSON responses.

        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate_json(prompt: str) -> Any:
            async with semaphore:
                return await self.generate_json_async(prompt, schema)

        return await asyncio.gather(*[
            _bounded_generate_json(prompt) for prompt in prompts
        ])
