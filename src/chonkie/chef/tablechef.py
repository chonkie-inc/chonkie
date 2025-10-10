"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from chonkie.chef.base import BaseChef
from chonkie.pipeline import chef
from chonkie.types import Document, MarkdownDocument, MarkdownTable

if TYPE_CHECKING:
    import pandas as pd


@chef("table")
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

    def parse(self, text: str) -> Union[MarkdownTable, List[MarkdownTable], None]:
        """Parse raw markdown text into a MarkdownDocument with extracted tables.

        Args:
            text: Raw markdown text.

        Returns:
            MarkdownDocument: Document with extracted tables.

        """
        tables = self.extract_tables_from_markdown(text)
        return tables

    def process(self, path: Union[str, Path]) -> MarkdownTable:
        """Process a CSV/Excel file or markdown file into a MarkdownDocument.

        Args:
            path (Union[str, Path]): Path to the CSV/Excel/markdown file.

        Returns:
            MarkdownDocument: Document with extracted tables.

        """
        self._lazy_import_pandas()
        # if file exists
        if Path(path).is_file():
            str_path = str(path)
            if str_path.endswith(".csv"):
                df = pd.read_csv(str_path)
                markdown = df.to_markdown(index=False)
                return self.parse(markdown)
            elif str_path.endswith(".xls") or str_path.endswith(".xlsx"):
                all_df = pd.read_excel(str_path,sheet_name=None)
                if len(all_df.keys()) > 1:
                    out: List[MarkdownTable] = []
                    for df in all_df.values():
                        text = df.to_markdown(index=False)
                        out.append(MarkdownTable(content=text, start_index=0, end_index=len(text)))
                    return out
                else:
                    df = list(all_df.values())[0]
                    text = df.to_markdown(index=False)
                    return MarkdownTable(content=text, start_index=0, end_index=len(text))
        return self.extract_tables_from_markdown(str(path))

    def process_batch(
        self, paths: Union[List[str], List[Path]]
    ) -> List[Union[MarkdownTable, List[MarkdownTable], None]]:
        """Process multiple CSV files and return a list of DataFrames.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the CSV files.

        Returns:
            List[Union[MarkdownTable, List[MarkdownTable], None]]: List of DataFrames or None for each file.

        """
        return [self.process(path) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]]
    ) -> Union[MarkdownTable, List[MarkdownTable], None, List[Union[MarkdownTable, List[MarkdownTable], None]]]:
        """Process a single file or a batch of files."""
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
