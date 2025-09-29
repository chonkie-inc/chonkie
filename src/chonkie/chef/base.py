"""Base class for chefs."""

from abc import ABC, abstractmethod
from typing import Any


class BaseChef(ABC):
    """Base class for chefs."""

    @abstractmethod
    def process(self, path: Any) -> Any:
        """Process the data."""
        raise NotImplementedError("Subclasses must implement process()")
    
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
