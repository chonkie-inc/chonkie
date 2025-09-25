"""TextChef is a chef that processes text data."""

from pathlib import Path
from typing import List, Union

from chonkie.types import Document

from .base import BaseChef


class TextChef(BaseChef):
    """TextChef is a chef that processes text data."""

    def process(self, path: Union[str, Path]) -> Document:
        """Process the text data from given file(s).

        Args:
            path (Union[str, Path]): Path to the file(s) to process.

        Returns:
            Document: Processed text data.

        """
        return Document(content=self.read(path))

    def process_batch(self, paths: Union[List[str], List[Path]]) -> List[Document]:
        """Process the text data in a batch.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the files to process.

        Returns:
            List[Document]: Processed text data.

        """
        return [self.process(path) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]]
    ) -> Union[Document, List[Document]]:
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
