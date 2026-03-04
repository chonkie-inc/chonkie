"""Base class for Handshakes."""

import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Sequence,
    Union,
)

from chonkie.logger import get_logger
from chonkie.types import Chunk

logger = get_logger(__name__)


class BaseHandshake(ABC):
    """Abstract base class for Handshakes."""

    @staticmethod
    def _generate_default_id(*args: Any) -> str:
        """Generate a default UUID."""
        return str(uuid.uuid4())

    @abstractmethod
    def write(self, chunks: Union[Chunk, list[Chunk]]) -> Any:
        """Write chunk(s) to the vector database.

        Args:
            chunks (Union[Chunk, list[Chunk]]): The chunk(s) to write.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError

    def __call__(self, chunks: Union[Chunk, list[Chunk]]) -> Any:
        """Write chunks using the default batch method when the instance is called.

        Args:
            chunks (Union[Chunk, list[Chunk]]): A single chunk or a sequence of chunks.

        Returns:
            Any: The result from the database write operation.

        """
        if isinstance(chunks, Chunk) or isinstance(chunks, Sequence):
            chunk_count = 1 if isinstance(chunks, Chunk) else len(chunks)
            logger.info(
                f"Writing {chunk_count} chunk(s) to database with {self.__class__.__name__}",
            )
            try:
                result = self.write(chunks)
                logger.debug(f"Successfully wrote {chunk_count} chunk(s)")
                return result
            except Exception as e:
                logger.error(
                    f"Failed to write {chunk_count} chunk(s) to database: {e}",
                    exc_info=True,
                )
                raise
        else:
            raise TypeError("Input must be a Chunk or a sequence of Chunks.")
