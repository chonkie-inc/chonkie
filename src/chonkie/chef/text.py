"""TextChef is a chef that processes text data."""

from pathlib import Path
from typing import Union

from chonkie.logger import get_logger
from chonkie.pipeline import chef
from chonkie.types import Document

from .base import BaseChef

logger = get_logger(__name__)


@chef("text")
class TextChef(BaseChef):
    """TextChef is a chef that processes text data."""

    def process(self, path: Union[str, Path]) -> Document:
        """Process the text data from given file(s).

        Args:
            path (Union[str, Path]): Path to the file(s) to process.

        Returns:
            Document: Processed text data.

        """
        logger.debug(f"Processing text file: {path}")
        content = self.read(path)
        logger.info(f"Text processing complete: read {len(content)} characters from {path}")
        return Document(content=content)

    def parse(self, text: str) -> Document:
        """Parse raw text into a Document.

        Args:
            text: Raw text to parse.

        Returns:
            Document: Document created from the text.

        """
        return Document(content=text)

    def process_batch(self, paths: Union[list[str], list[Path]]) -> list[Document]:
        """Process the text data in a batch.

        Args:
            paths (Union[list[str], list[Path]]): Paths to the files to process.

        Returns:
            list[Document]: Processed text data.

        """
        return [self.process(path) for path in paths]

    def __call__(  # type: ignore[override]
        self,
        path: Union[str, Path, list[str], list[Path]],
    ) -> Union[Document, list[Document]]:
        """Process the text data from given file(s).

        Args:
            path: Path to file(s) to process. Can be single path or list of paths.

        Returns:
            Document or list[Document] created from the file(s).

        """
        if isinstance(path, (list, tuple)):
            return self.process_batch(path)
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
