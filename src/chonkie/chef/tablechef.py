"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import re
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from .base import BaseChef

if TYPE_CHECKING:
    import pandas as pd


class TableChef(BaseChef):
    """TableChef processes CSV files and returns pandas DataFrames."""

    table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")

    def _lazy_import_pandas(self) -> None:
        global pd
        import pandas as pd

    def process(self, path: Union[str, Path]) -> Union[str, List[str], None]:
        """Process a CSV file and return a pandas DataFrame.

        Args:
            path (Union[str, Path]): Path to the CSV file.

        Returns:
            Union[str, List[str], None]: Markdown string of the table or list of markdown tables.

        """
        self._lazy_import_pandas()
        # if file exists
        if Path(path).is_file():
            str_path = str(path)
            if str_path.endswith(".csv"):
                df = pd.read_csv(str_path)
                return df.to_markdown(index=False)
            elif str_path.endswith(".xls") or str_path.endswith(".xlsx"):
                df = pd.read_excel(str_path)
                return df.to_markdown(index=False)
        # else string is a markedown table
        return self.extract_tables_from_markdown(str(path))

    def process_batch(
        self, paths: Union[List[str], List[Path]]
    ) -> List["pd.DataFrame"]:
        """Process multiple CSV files and return a list of DataFrames.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the CSV files.

        Returns:
            List[pd.DataFrame]: List of DataFrames.

        """
        return [self.process(path) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]]
    ) -> Union[
        "pd.DataFrame",
        List["pd.DataFrame"],
        None,
        List[Union["pd.DataFrame", List["pd.DataFrame"], None]],
    ]:
        """Process one or more CSV files and return DataFrame(s)."""
        if isinstance(path, (list, tuple)):
            return self.process_batch(path)
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def extract_tables_from_markdown(self, markdown: str) -> List[str]:
        """Extract markdown tables from a markdown string.

        Args:
            markdown (str): The markdown text containing tables.

        Returns:
            List[str]: A list of strings, each representing a markdown table found in the input.

        """
        tables: List[str] = []
        for match in self.table_pattern.finditer(markdown):
            tables.append(match.group(0))
        return tables

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
