"""Table converter utilities for transforming tables to different formats."""

import re
from html.parser import HTMLParser
from typing import Any

NUMERIC_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


class HTMLTableParser(HTMLParser):
    """Simple HTML table parser to extract headers and rows."""

    def __init__(self) -> None:
        """Initialize the HTML table parser."""
        super().__init__()
        self.tables: list[dict[str, Any]] = []
        self._current_table: dict[str, Any] = {"headers": [], "rows": []}
        self._in_thead = False
        self._in_tbody = False
        self._in_th = False
        self._in_td = False
        self._current_headers: list[str] = []
        self._current_row: list[str] = []
        self._current_cell_data = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening HTML tags."""
        match tag:
            case "table":
                self._current_table = {"headers": [], "rows": []}
            case "thead":
                self._in_thead = True
                self._current_headers = []
            case "tbody":
                self._in_tbody = True
            case "th":
                self._in_th = True
                self._current_cell_data = ""
            case "td":
                self._in_td = True
                self._current_cell_data = ""
            case "tr":
                self._current_row = []

    def handle_endtag(self, tag: str) -> None:
        """Handle closing HTML tags."""
        match tag:
            case "table":
                if self._current_headers:
                    self._current_table["headers"] = self._current_headers
                self.tables.append(self._current_table)
            case "thead":
                self._in_thead = False
                if self._current_headers:
                    self._current_table["headers"] = self._current_headers
            case "tbody":
                self._in_tbody = False
            case "th":
                self._in_th = False
                self._current_headers.append(self._current_cell_data.strip())
            case "td":
                self._in_td = False
                self._current_row.append(self._current_cell_data.strip())
            case "tr":
                if self._current_row:
                    self._current_table["rows"].append(self._current_row)

    def handle_data(self, data: str) -> None:
        """Handle text data within HTML tags."""
        if self._in_th or self._in_td:
            self._current_cell_data += data


def _infer_type(value: str) -> int | float | str:
    """Infer the type of a cell value."""
    if not value:
        return ""
    if NUMERIC_PATTERN.match(value):
        return float(value) if "." in value else int(value)
    return value


def _rows_to_json(headers: list[str], rows: list[list[str]]) -> list[dict[str, int | float | str]]:
    """Convert parsed headers and rows to JSON format."""
    result: list[dict[str, int | float | str]] = []
    for row in rows:
        row_dict: dict[str, int | float | str] = {}
        for i, header in enumerate(headers):
            row_dict[header] = _infer_type(row[i]) if i < len(row) else ""
        result.append(row_dict)
    return result


def _parse_markdown_table(table_content: str) -> tuple[list[str], list[list[str]]]:
    """Parse markdown table into headers and rows."""
    lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return [], []

    headers = [cell.strip() for cell in lines[0].split("|") if cell.strip()]
    rows = [[c for c in [cell.strip() for cell in line.split("|")] if c] for line in lines[2:]]
    return headers, [r for r in rows if r]


def markdown_table_to_json(table_content: str) -> list[dict[str, int | float | str]]:
    """Convert a markdown table to a JSON-serializable list of dictionaries."""
    headers, rows = _parse_markdown_table(table_content)
    if not headers or not rows:
        return []
    return _rows_to_json(headers, rows)


def html_table_to_json(table_content: str) -> list[dict[str, int | float | str]] | None:
    """Convert an HTML table to a JSON-serializable list of dictionaries."""
    parser = HTMLTableParser()
    try:
        parser.feed(table_content)
    except Exception:
        return None

    if not parser.tables:
        return None

    table = parser.tables[0]
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if not headers:
        return None

    return _rows_to_json(headers, rows)
