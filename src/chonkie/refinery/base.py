"""Base class for all refinery classes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonkie.types import Document

from chonkie.logger import get_logger
from chonkie.types import Chunk

logger = get_logger(__name__)


class BaseRefinery(ABC):
    """Base class for all refinery classes."""

    @abstractmethod
    def refine(self, chunks: list[Chunk]) -> list[Chunk]:
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

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Call the refinery.

        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        logger.info(f"Refining {len(chunks)} chunks with {self.__class__.__name__}")
        try:
            refined_chunks = self.refine(chunks)
            logger.info(f"Refinement complete: {len(refined_chunks)} chunks output")
            return refined_chunks
        except Exception as e:
            logger.error(f"Refinement failed: {str(e)}", error_type=type(e).__name__)
            raise
