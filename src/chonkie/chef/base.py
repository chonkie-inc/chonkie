"""Base class for chefs."""

from abc import ABC, abstractmethod
from typing import Any

from chonkie.logger import get_logger

logger = get_logger(__name__)


class BaseChef(ABC):
    """Base class for chefs."""

    @abstractmethod
    def process(self, path: Any) -> Any:
        """Process the data."""
        raise NotImplementedError("Subclasses must implement process()")
    
    def process_batch(self, paths: Any) -> Any:
        """Process the data in a batch."""
        logger.info(f"Processing batch of {len(paths)} files")
        results = [self.process(path) for path in paths]
        logger.info(f"Completed batch processing of {len(paths)} files")
        return results

    def read(self, path: Any) -> Any:
        """Read the file content."""
        try:
            logger.debug(f"Reading file: {path}")
            with open(path, "r", encoding="utf-8") as file:
                content = str(file.read())
                logger.debug(f"Successfully read file: {path}", size=len(content))
                return content
        except Exception as e:
            logger.warning(f"Failed to read file: {path}", error=str(e))
            raise

    def __call__(self, path: Any) -> Any:
        """Call the chef to process the data."""
        logger.debug(f"Processing file with {self.__class__.__name__}: {path}")
        return self.process(path)

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
