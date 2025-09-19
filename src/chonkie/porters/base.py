"""Module for Chonkie's Porters.

Porters allow the user to _export_ data from chonkie into a variety of formats for saving on disk or cloud blob storage. Porters make the implicit assumption that the data is not being used for querying, but rather for saving.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from chonkie.types import Chunk


class BasePorter(ABC):
    """Abstract base class for Chonkie's Porters.

    Porters are responsible for exporting Chonkie's Chunks into a variety of formats.
    The main method to implement is `export`, which should take in a list of Chunks
    and any other arguments, and export them to the desired format.
    """

    @abstractmethod
    def export(self, chunks: List[Chunk], **kwargs: Dict[str, Any]) -> None:
        """Export the chunks to the desired format.
        
        Args:
            chunks: The chunks to export.
            **kwargs: Additional keyword arguments.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
            
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, chunks: List[Chunk], **kwargs: Dict[str, Any]) -> None:
        """Export the chunks to the desired format."""
        return self.export(chunks)