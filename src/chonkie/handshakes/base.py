"""Base class for Handshakes."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    List,
    Sequence,
    Union,
)

from chonkie.logger import get_logger
from chonkie.types import Chunk

logger = get_logger(__name__)

# TODO: Move this to inside the BaseHandshake class
# Why is this even outside the class?
# def _generate_default_id(*args: Any) -> str:
#     """Generate a default UUID."""
#     return str(uuid.uuid4())


class BaseHandshake(ABC):
    """Abstract base class for Handshakes."""

    @abstractmethod
    def write(self, chunk: Union[Chunk, List[Chunk]]) -> Any:
        """Write a single chunk to the vector database.

        Args:
            chunk (Union[Chunk, List[Chunk]]): The chunk to write.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError

    def __call__(self, chunks: Union[Chunk, List[Chunk]]) -> Any:
        """Write chunks using the default batch method when the instance is called.

        Args:
            chunks (Union[Chunk, List[Chunk]]): A single chunk or a sequence of chunks.

        Returns:
            Any: The result from the database write operation.

        """
        if isinstance(chunks, Chunk) or isinstance(chunks, Sequence):
            chunk_count = 1 if isinstance(chunks, Chunk) else len(chunks)
            logger.info(f"Writing {chunk_count} chunk(s) to database with {self.__class__.__name__}")
            try:
                result = self.write(chunks)
                logger.debug(f"Successfully wrote {chunk_count} chunk(s)")
                return result
            except Exception as e:
                logger.error(
                    f"Failed to write {chunk_count} chunk(s) to database",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        else:
            raise TypeError("Input must be a Chunk or a sequence of Chunks.")