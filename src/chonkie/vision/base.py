"""Base class for vision components."""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Union


class BaseVision(ABC):
    """Base class for vision pipeline components.

    Vision components extract text from visual documents (images, PDFs)
    and return the extracted text as strings. They sit between the fetcher
    and chef stages in the CHOMP pipeline.

    """

    @abstractmethod
    def process(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from a single file.

        Args:
            file_path: Path to the image or PDF file.

        Returns:
            Extracted text content.

        """
        raise NotImplementedError("Subclasses must implement process()")

    async def aprocess(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from a single file asynchronously."""
        return await asyncio.to_thread(self.process, file_path)

    def process_batch(self, file_paths: list[Union[str, os.PathLike]]) -> list[str]:
        """Extract text from multiple files.

        Args:
            file_paths: List of file paths to process.

        Returns:
            List of extracted text strings.

        """
        return [self.process(fp) for fp in file_paths]

    async def aprocess_batch(
        self, file_paths: list[Union[str, os.PathLike]], max_concurrency: int = 5
    ) -> list[str]:
        """Extract text from multiple files asynchronously."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(fp: Union[str, os.PathLike]) -> str:
            async with semaphore:
                return await self.aprocess(fp)

        return await asyncio.gather(*[_bounded(fp) for fp in file_paths])

    def __call__(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from a file."""
        return self.process(file_path)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}()"
