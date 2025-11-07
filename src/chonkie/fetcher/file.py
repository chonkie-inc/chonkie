"""FileFetcher is a fetcher that fetches paths of files from local directories."""

import os
from pathlib import Path
from typing import List, Optional, Union

from chonkie.pipeline import fetcher

from .base import BaseFetcher


@fetcher("file")
class FileFetcher(BaseFetcher):
    """FileFetcher fetches a single file or multiple files from a directory.

    Supports two modes:
    - Single file mode: provide 'path' parameter
    - Directory mode: provide 'dir' parameter (optionally with 'ext' filter)
    """

    def __init__(self) -> None:
        """Initialize the FileFetcher."""
        super().__init__()

    def fetch(
        self,
        path: Optional[str] = None,
        dir: Optional[str] = None,
        ext: Optional[List[str]] = None,
    ) -> Union[Path, List[Path]]:
        """Fetch a single file or files from a directory.

        Args:
            path: Path to a single file
            dir: Directory to fetch files from
            ext: File extensions to filter (only used with dir parameter)

        Returns:
            Union[Path, List[Path]]: Single Path for file mode, List[Path] for directory mode

        Raises:
            ValueError: If neither or both path and dir are provided
            FileNotFoundError: If the specified file or directory doesn't exist

        Examples:
            ```python
            # Single file mode
            fetcher.fetch(path="document.txt")

            # Directory mode
            fetcher.fetch(dir="./docs", ext=[".txt", ".md"])
            ```

        """
        if path is not None and dir is not None:
            raise ValueError("Provide either 'path' or 'dir', not both")

        if path is not None:
            # Single file mode
            file_path = Path(path)
            if not file_path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            return file_path

        elif dir is not None:
            dir_path = Path(dir)
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Directory not found: {dir}")

            # Use os.walk for a safe recursive walk, avoiding symlink loops.
            all_files: List[Path] = []
            for root, _, filenames in os.walk(dir_path, followlinks=False):
                for filename in filenames:
                    # Check extension if a filter is provided
                    if ext is None or os.path.splitext(filename)[1] in ext:
                        # Construct the full path and add it to the list
                        full_path = Path(os.path.join(root, filename))
                        all_files.append(full_path)
            return all_files
        else:
            raise ValueError("Must provide either 'path' or 'dir'")

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

    def __call__(
        self,
        path: Optional[str] = None,
        dir: Optional[str] = None,
        ext: Optional[List[str]] = None,
    ) -> Union[Path, List[Path]]:  # type: ignore[override]
        """Fetch a single file or files from a directory.

        Args:
            path: Path to a single file
            dir: Directory to fetch files from
            ext: File extensions to filter (only used with dir parameter)

        Returns:
            Union[Path, List[Path]]: Single Path for file mode, List[Path] for directory mode

        """
        return self.fetch(path=path, dir=dir, ext=ext)
