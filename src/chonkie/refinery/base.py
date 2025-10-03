"""Base class for all refinery classes."""

from abc import ABC, abstractmethod
from typing import List

from chonkie.logger import get_logger
from chonkie.types import Chunk

logger = get_logger(__name__)


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
        """Refine the chunk."""
        raise NotImplementedError("Subclasses must implement this method.")
    
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
        logger.info(f"Refining {len(chunks)} chunks with {self.__class__.__name__}")
        try:
            refined_chunks = self.refine(chunks)
            logger.info(f"Refinement complete: {len(refined_chunks)} chunks output")
            return refined_chunks
        except Exception as e:
            logger.error(f"Refinement failed: {str(e)}", error_type=type(e).__name__)
            raise