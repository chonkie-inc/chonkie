"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import re
from pathlib import Path
from typing import Union

from chonkie.chef.base import BaseChef
from chonkie.logger import get_logger
from chonkie.pipeline import chef
from chonkie.types import Document, MarkdownDocument, MarkdownTable

logger = get_logger(__name__)


@chef("table")
class TableChef(BaseChef):
    """TableChef processes CSV files and returns pandas DataFrames."""

    def __init__(self) -> None:
        """Initialize TableChef with a regex pattern for markdown tables."""
        self.table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")

    def parse(self, text: str) -> Document:
        """Parse raw markdown text and extract tables into a MarkdownDocument.

        Args:
            text: Raw markdown text.

        Returns:
            Document: MarkdownDocument with extracted tables.

        """
        logger.debug("Parsing markdown text for tables")
        tables = self.extract_tables_from_markdown(text)
        logger.info(f"Markdown table extraction complete: found {len(tables)} tables")
        return MarkdownDocument(content=text, tables=tables)

    def process(self, path: Union[str, Path]) -> Document:
        """Process a CSV/Excel file or markdown text into a MarkdownDocument.

        Args:
            path (Union[str, Path]): Path to the CSV/Excel file, or markdown text string.

        Returns:
            Document: MarkdownDocument with extracted tables.

        """
        logger.debug(f"Processing table file/string: {path}")
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Pandas is required to use TableChef. Please install it with `pip install chonkie[table]`.",
            ) from e
        # if file exists
        path_obj = Path(path)
        if path_obj.is_file():
            str_path = str(path)
            if str_path.endswith(".csv"):
                logger.debug("Processing CSV file")
                df = pd.read_csv(str_path)
                markdown = df.to_markdown(index=False)
                logger.info(f"CSV processing complete: converted {len(df)} rows to markdown")
                # CSV always produces a single table
                table = MarkdownTable(content=markdown, start_index=0, end_index=len(markdown))
                return MarkdownDocument(content=markdown, tables=[table])
            elif str_path.endswith(".xls") or str_path.endswith(".xlsx"):
                logger.debug("Processing Excel file")
                all_df = pd.read_excel(str_path, sheet_name=None)
                tables: list[MarkdownTable] = []
                all_content = []
                for df in all_df.values():
                    text = df.to_markdown(index=False)
                    all_content.append(text)
                    tables.append(MarkdownTable(content=text, start_index=0, end_index=len(text)))
                # Join all sheets with double newline
                content = "\n\n".join(all_content)
                logger.info(
                    f"Excel processing complete: converted {len(all_df)} sheets to markdown",
                )
                return MarkdownDocument(content=content, tables=tables)
        # Not a file, treat as markdown string and extract tables
        logger.debug("Extracting tables from markdown string")
        return self.parse(str(path))

    def process_batch(self, paths: Union[list[str], list[Path]]) -> list[Document]:
        """Process multiple CSV/Excel files or markdown texts.

        Args:
            paths (Union[list[str], list[Path]]): Paths to files or markdown text strings.

        Returns:
            list[Document]: List of MarkdownDocuments with extracted tables.

        """
        logger.debug(f"Processing batch of {len(paths)} files/strings")
        results = [self.process(path) for path in paths]
        logger.info(f"Completed batch processing of {len(paths)} files/strings")
        return results

    def __call__(  # type: ignore[override]
        self,
        path: Union[str, Path, list[str], list[Path]],
    ) -> Union[Document, list[Document]]:
        """Process a single file/text or a batch of files/texts.

        Args:
            path: Single file path, markdown text string, or list of paths/texts.

        Returns:
            Union[Document, list[Document]]: MarkdownDocument or list of MarkdownDocuments.

        """
        if isinstance(path, (list, tuple)):
            return self.process_batch(path)
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def extract_tables_from_markdown(self, markdown: str) -> list[MarkdownTable]:
        """Extract markdown tables from a markdown string.

        Args:
            markdown (str): The markdown text containing tables.

        Returns:
            list[MarkdownTable]: A list of MarkdownTable objects, each representing a markdown table found in the input.

        """
        tables: list[MarkdownTable] = []
        for match in self.table_pattern.finditer(markdown):
            table_content = match.group(0)
            start_index = match.start()
            end_index = match.end()
            tables.append(
                MarkdownTable(content=table_content, start_index=start_index, end_index=end_index),
            )
        return tables

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
