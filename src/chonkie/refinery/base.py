"""Base class for all refinery classes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from chonkie.types import Document

from chonkie.types import Chunk


class BaseRefinery(ABC):
    """Base class for all refinery classes."""

    @abstractmethod
    def _is_available(self) -> bool:
        """Check if the refinery is available."""
        pass

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the chunks.

        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def refine_document(self, document: "Document") -> "Document":
        """Refine the chunks within a document.

        Args:
            document: The document whose chunks should be refined.

        Returns:
            The document with refined chunks.

        """
        document.chunks = self.refine(document.chunks)
        return document

    def __repr__(self) -> str:
        """Return the string representation of the refinery."""
        return f"{self.__class__.__name__}()"

    def __call__(self, chunks: List[Chunk]) -> List[Chunk]:
        """Call the refinery.

        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        return self.refine(chunks)