"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import os
import re
from pathlib import Path
from typing import Union, cast

from chonkie.chef.base import BaseChef
from chonkie.logger import get_logger
from chonkie.pipeline import chef
from chonkie.types import Document, MarkdownDocument, MarkdownTable

logger = get_logger(__name__)


@chef("table")
class TableChef(BaseChef):
    """TableChef processes CSV files and returns pandas DataFrames."""

    def __init__(self) -> None:
        """Initialize TableChef with regex patterns for markdown and HTML tables."""
        self.table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")
        # Matches individual opening/closing <table> tags only — no DOTALL, no .*?
        # Depth tracking in _find_html_table_spans handles nesting and avoids ReDoS.
        self._html_table_tag_pattern = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)

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

    def process(self, path: str | os.PathLike) -> Document:
        """Process a CSV/Excel file or markdown text into a MarkdownDocument.

        Args:
            path: Path to the CSV/Excel file, or markdown text string.

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
                markdown = df.to_markdown(index=False) or ""
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
                    text = df.to_markdown(index=False) or ""
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

    def process_batch(self, paths: list[str | os.PathLike]) -> list[Document]:
        """Process multiple CSV/Excel files or markdown texts.

        Args:
            paths: Paths to files or markdown text strings.

        Returns:
            list[Document]: List of MarkdownDocuments with extracted tables.

        """
        logger.debug(f"Processing batch of {len(paths)} files/strings")
        results = [self.process(path) for path in paths]
        logger.info(f"Completed batch processing of {len(paths)} files/strings")
        return results

    def __call__(  # type: ignore[override]
        self,
        path: str | os.PathLike | list[str | os.PathLike],
    ) -> Union[Document, list[Document]]:
        """Process a single file/text or a batch of files/texts.

        Args:
            path: Single file path, markdown text string, or list of paths/texts.

        Returns:
            Union[Document, list[Document]]: MarkdownDocument or list of MarkdownDocuments.

        """
        if isinstance(path, (list, tuple)):
            return self.process_batch(cast("list[str | os.PathLike]", list(path)))
        elif isinstance(path, (str, Path)):
            return self.process(path)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def _find_html_table_spans(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) character spans for each top-level HTML table.

        Uses depth tracking so nested <table> elements are included in their
        enclosing table's span rather than being returned as separate tables.
        The tag pattern [^>]* has a hard stop character and cannot backtrack
        catastrophically regardless of input length.
        """
        spans: list[tuple[int, int]] = []
        depth = 0
        start = -1
        for m in self._html_table_tag_pattern.finditer(text):
            if m.group(1):  # closing </table>
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        spans.append((start, m.end()))
            else:  # opening <table …>
                if depth == 0:
                    start = m.start()
                depth += 1
        return spans

    def extract_tables_from_markdown(self, markdown: str) -> list[MarkdownTable]:
        """Extract markdown and HTML tables from a markdown string.

        Args:
            markdown (str): The markdown text containing tables.

        Returns:
            list[MarkdownTable]: A list of MarkdownTable objects, each representing a table found in the input.

        """
        tables: list[MarkdownTable] = []

        # Find HTML tables first; they take priority when spans overlap
        html_tables: list[MarkdownTable] = []
        for start, end in self._find_html_table_spans(markdown):
            html_tables.append(
                MarkdownTable(
                    content=markdown[start:end],
                    start_index=start,
                    end_index=end,
                )
            )

        # Find markdown pipe tables, skipping any that overlap with an HTML match
        for match in self.table_pattern.finditer(markdown):
            start_index = match.start()
            end_index = match.end()
            if any(
                start_index < ht.end_index and ht.start_index < end_index for ht in html_tables
            ):
                continue
            tables.append(
                MarkdownTable(content=match.group(0), start_index=start_index, end_index=end_index)
            )

        tables.extend(html_tables)
        # Sort tables by their appearance in the text
        tables.sort(key=lambda x: x.start_index)
        return tables

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
