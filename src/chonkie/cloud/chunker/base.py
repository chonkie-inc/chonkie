"""Base class for all cloud chunking algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Union


class CloudChunker(ABC):
    """Base class for all cloud chunking algorithms."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    @abstractmethod
    def chunk(self, text: Union[str, list[str]]) -> Any:
        """Chunk the text into a list of chunks."""
        pass

    def __call__(self, text: Union[str, list[str]]) -> Any:
        """Call the chunker."""
        return self.chunk(text)
