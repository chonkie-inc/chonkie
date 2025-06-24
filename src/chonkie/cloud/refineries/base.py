"""Base class for all refinery classes."""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseRefinery(ABC):
    """Base class for all cloud refinery classes."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"

    @abstractmethod
    def refine(self, chunks: List[Any]) -> List[Any]:
        """Refine the chunks."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __call__(self, chunks: List[Any]) -> List[Any]:
        """Call the refinery.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.
            
        """
        return self.refine(chunks)