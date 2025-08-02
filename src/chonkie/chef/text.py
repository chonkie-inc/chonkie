"""TextChef is a chef that processes text data."""

from pathlib import Path
from typing import List, Union

from .base import BaseChef


class TextChef(BaseChef):
    """TextChef is a chef that processes text data."""

    def process(self, path: Union[str, Path]) -> str:
        """Process the text data from given file(s).

        Args:
            path (Union[str, Path]): Path to the file(s) to process.

        Returns:
            str: Processed text data.

        """
        with open(path, "r") as file:
            return str(file.read())

    def process_batch(self, paths: Union[List[str], List[Path]]) -> List[str]:
        """Process the text data in a batch.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the files to process.

        Returns:
            List[str]: Processed text data.

        """
        return [self.process(path) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]]
    ) -> Union[str, List[str]]:
        """Process the text data from given file(s)."""
        if isinstance(path, (list, tuple)):
            return self.process_batch(path)
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
