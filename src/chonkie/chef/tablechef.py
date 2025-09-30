"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from chonkie.chef.base import BaseChef
from chonkie.types import MarkdownTable
from chonkie.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pandas as pd


class TableChef(BaseChef):
    """TableChef processes CSV files and returns pandas DataFrames."""

    def __init__(self) -> None:
        """Initialize TableChef with a regex pattern for markdown tables."""
        self.table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")

    def _lazy_import_pandas(self) -> None:
        try: 
            global pd
            import pandas as pd
        except ImportError as e:
            raise ImportError("Pandas is required to use TableChef. Please install it with `pip install chonkie[table]`.") from e

    def process(self, path: Union[str, Path]) -> Union[str, List[MarkdownTable], None]:
        """Process a CSV file and return a pandas DataFrame.

        Args:
            path (Union[str, Path]): Path to the CSV file.

        Returns:
            Union[str, List[MarkdownTable], None]: Markdown string of the table or list of markdown tables.

        """
        logger.debug(f"Processing table file/string: {path}")
        self._lazy_import_pandas()
        # if file exists
        if Path(path).is_file():
            str_path = str(path)
            if str_path.endswith(".csv"):
                logger.debug("Processing CSV file")
                df = pd.read_csv(str_path)
                markdown = df.to_markdown(index=False)
                logger.info(f"CSV processing complete: converted {len(df)} rows to markdown")
                return markdown
            elif str_path.endswith(".xls") or str_path.endswith(".xlsx"):
                logger.debug("Processing Excel file")
                df = pd.read_excel(str_path)
                markdown = df.to_markdown(index=False)
                logger.info(f"Excel processing complete: converted {len(df)} rows to markdown")
                return markdown
        # else string is a markedown table
        logger.debug("Extracting tables from markdown string")
        tables = self.extract_tables_from_markdown(str(path))
        logger.info(f"Markdown table extraction complete: found {len(tables)} tables")
        return tables

    def process_batch(
        self, paths: Union[List[str], List[Path]]
    ) -> List[Union[str, List[MarkdownTable], None]]:
        """Process multiple CSV files and return a list of DataFrames.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the CSV files.

        Returns:
            List[pd.DataFrame]: List of DataFrames.

        """
        return [self.process(path) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]]
    ) -> Union[str, List[MarkdownTable], None, List[Union[str, List[MarkdownTable], None]]]:
        """Process one or more CSV files and return DataFrame(s)."""
        if isinstance(path, (list, tuple)):
            return self.process_batch(path)
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def extract_tables_from_markdown(self, markdown: str) -> List[MarkdownTable]:
        """Extract markdown tables from a markdown string.

        Args:
            markdown (str): The markdown text containing tables.

        Returns:
            List[MarkdownTable]: A list of MarkdownTable objects, each representing a markdown table found in the input.

        """
        tables: List[MarkdownTable] = []
        for match in self.table_pattern.finditer(markdown):
            table_content = match.group(0)
            start_index = match.start()
            end_index = match.end()
            tables.append(MarkdownTable(content=table_content, start_index=start_index, end_index=end_index))
        return tables

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
