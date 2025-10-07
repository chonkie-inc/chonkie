"""Base class for chefs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from chonkie.types import Document


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
        return [self.process(path) for path in paths]

    def read(self, path: Union[str, Path]) -> str:
        """Read the file content.

        Args:
            path: Path to the file to read.

        Returns:
            File content as string.

        """
        with open(path, "r", encoding="utf-8") as file:
            return str(file.read())

    def __call__(self, path: Union[str, Path]) -> Document:
        """Call the chef to process the data.

        Args:
            path: Path to the file to process.

        Returns:
            Document created from the file.

        """
        return self.process(path)

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
