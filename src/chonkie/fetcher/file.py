"""FileFetcher is a fetcher that fetches paths of files from local directories."""

from pathlib import Path
from typing import List, Optional

from .base import BaseFetcher


class FileFetcher(BaseFetcher):
    """FileFetcher is a fetcher that fetches paths of files from local directories."""

    def __init__(self) -> None:
        """Initialize the FileFetcher."""
        super().__init__()

    def fetch(self, dir: str, ext: Optional[List[str]] = None) -> List[Path]:
        """Fetch files from a directory.

        Args:
            dir (str): The directory to fetch files from.
            ext (Optional[List[str]]): The file extensions to fetch.

        Returns:
            List[Path]: The list of files fetched from the directory.

        """
        # Reads the entire directory and returns a list of files with the specified extension
        return [
            file
            for file in Path(dir).iterdir()
            if file.is_file() and (ext is None or file.suffix in ext)
        ]

    def fetch_file(self, dir: str, name: str) -> Path:  # type: ignore[override]
        """Given a directory and a file name, return the path to the file.

        NOTE: This method is mostly for uniformity across fetchers since one may require to
        get a file from an online database.
        """
        # We should search the directory for the file
        for file in Path(dir).iterdir():
            if file.is_file() and file.name == name:
                return file
        raise FileNotFoundError(f"File {name} not found in directory {dir}")

    def __call__(self, dir: str, ext: Optional[List[str]] = None) -> List[Path]:  # type: ignore[override]
        """Fetch files from a directory.

        Args:
            dir (str): The directory to fetch files from.
            ext (Optional[List[str]]): The file extensions to fetch.

        Returns:
            List[Path]: The list of files fetched from the directory.

        """
        return self.fetch(dir, ext)
