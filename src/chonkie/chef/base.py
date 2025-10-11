"""Base class for chefs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from chonkie.logger import get_logger
from chonkie.types import Document

logger = get_logger(__name__)


class BaseChef(ABC):
    """Base class for chefs."""

    @abstractmethod
    def process(self, path: Union[str, Path]) -> Document:
        """Process the data from a file path.

        Args:
            path: Path to the file to process.

        Returns:
            Document created from the file.

        """
        raise NotImplementedError("Subclasses must implement process()")

    @abstractmethod
    def parse(self, text: str) -> Document:
        """Parse raw text into a Document.

        Args:
            text: Raw text to parse.

        Returns:
            Document created from the text.

        """
        raise NotImplementedError("Subclasses must implement parse()")

    def process_batch(self, paths: Union[List[str], List[Path]]) -> List[Document]:
        """Process multiple files in a batch.

        Args:
            paths: List of file paths to process.

        Returns:
            List of Documents created from the files.

        """
        logger.info(f"Processing batch of {len(paths)} files")
        results = [self.process(path) for path in paths]
        logger.info(f"Completed batch processing of {len(paths)} files")
        return results

    def read(self, path: Union[str, Path]) -> str:
        """Read the file content.

        Args:
            path: Path to the file to read.

        Returns:
            File content as string.

        """
        try:
            logger.debug(f"Reading file: {path}")
            with open(path, "r", encoding="utf-8") as file:
                content = str(file.read())
                logger.debug(f"Successfully read file: {path}", size=len(content))
                return content
        except Exception as e:
            logger.warning(f"Failed to read file: {path}", error=str(e))
            raise

    def __call__(self, path: Union[str, Path]) -> Document:
        """Call the chef to process the data.

        Args:
            path: Path to the file to process.

        Returns:
            Document created from the file.

        """
        logger.debug(f"Processing file with {self.__class__.__name__}: {path}")
        return self.process(path)

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
