"""Base class for chefs."""

from abc import ABC, abstractmethod
from typing import Any

from chonkie.types import Document


class BaseChef(ABC):
    """Base class for chefs."""

    @abstractmethod
    def process(self, path: Any) -> Document:
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

    def process_batch(self, paths: Any) -> Any:
        """Process the data in a batch."""
        return [self.process(path) for path in paths]

    def read(self, path: Any) -> Any:
        """Read the file content."""
        with open(path, "r", encoding="utf-8") as file:
            return str(file.read())

    def __call__(self, path: Any) -> Any:
        """Call the chef to process the data."""
        return self.process(path)

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
