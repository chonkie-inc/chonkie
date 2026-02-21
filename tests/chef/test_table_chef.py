"""Tests for the TableChef class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chonkie import TableChef
from chonkie.types import MarkdownDocument


class TestTableChef:
    """Test suite for TableChef class."""

    @pytest.fixture
    def table_chef(self) -> TableChef:
        """Fixture that returns a TableChef instance."""
        return TableChef()

    @pytest.fixture
    def csv_content(self) -> str:
        """Fixture that returns a simple CSV string."""
        return "col1,col2\n1,2\n3,4"

    @pytest.fixture
    def excel_df(self) -> pd.DataFrame:
        """Fixture that returns a simple DataFrame for Excel tests."""
        return pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

    @pytest.fixture
    def markdown_table(self) -> str:
        """Fixture that returns a markdown table string."""
        return """
| col1 | col2 |
|------|------|
| 1    | 2    |
| 3    | 4    |
""".strip()

    def test_process_csv_file(
        self: "TestTableChef",
        table_chef: TableChef,
        csv_content: str,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        """Test processing a CSV file with TableChef."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        called = {}
        import pandas as pd

        orig_read_csv = pd.read_csv

        def fake_read_csv(path, *args, **kwargs):  # type: ignore
            called["called"] = True
            return orig_read_csv(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)
        result = table_chef.process(csv_file)
        assert hasattr(result, "content")
        # Check for column names and values, not exact formatting
        content_str = str(result.content)
        assert "col1" in content_str and "col2" in content_str
        assert (
            "1" in content_str and "2" in content_str and "3" in content_str and "4" in content_str
        )
        assert called["called"]

    def test_process_excel_file(
        self: "TestTableChef",
        table_chef: TableChef,
        excel_df: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        """Test processing an Excel file with TableChef."""
        excel_file = tmp_path / "test.xlsx"
        excel_df.to_excel(excel_file, index=False)
        called = {}
        import pandas as pd

        orig_read_excel = pd.read_excel

        def fake_read_excel(path: str, *args: object, **kwargs: object) -> pd.DataFrame:
            called["called"] = True
            return orig_read_excel(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_excel", fake_read_excel)
        result = table_chef.process(excel_file)
        assert hasattr(result, "content")
        content_str = str(result.content)
        assert "col1" in content_str and "col2" in content_str
        assert (
            "1" in content_str and "2" in content_str and "3" in content_str and "4" in content_str
        )
        assert called["called"]

    def test_process_markdown_table_string(
        self: "TestTableChef",
        table_chef: TableChef,
        markdown_table: str,
    ) -> None:
        """Test processing a markdown table string."""
        result = table_chef.process(markdown_table)
        assert isinstance(result, MarkdownDocument)
        assert len(result.tables) == 1
        assert hasattr(result.tables[0], "content")

    def test_process_batch(
        self: "TestTableChef",
        table_chef: TableChef,
        csv_content: str,
        tmp_path: Path,
    ) -> None:
        """Test batch processing of multiple CSV files."""
        file1 = tmp_path / "a.csv"
        file2 = tmp_path / "b.csv"
        file1.write_text(csv_content)
        file2.write_text(csv_content)
        results = table_chef.process_batch([file1, file2])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            if hasattr(r, "content"):
                content_str = str(r.content)
                assert "col1" in content_str and "col2" in content_str
                assert (
                    "1" in content_str
                    and "2" in content_str
                    and "3" in content_str
                    and "4" in content_str
                )
            elif isinstance(r, list):
                assert all(hasattr(t, "content") for t in r)
            elif r is None:
                continue
            else:
                assert False, f"Unexpected result type: {type(r)}"

    def test_call_with_list(
        self: "TestTableChef",
        table_chef: TableChef,
        csv_content: str,
        tmp_path: Path,
    ) -> None:
        """Test calling TableChef with a list of file paths."""
        file1 = tmp_path / "a.csv"
        file2 = tmp_path / "b.csv"
        file1.write_text(csv_content)
        file2.write_text(csv_content)
        results = table_chef([file1, file2])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            if hasattr(r, "content"):
                content_str = str(r.content)
                assert "col1" in content_str and "col2" in content_str
                assert (
                    "1" in content_str
                    and "2" in content_str
                    and "3" in content_str
                    and "4" in content_str
                )
            elif isinstance(r, list):
                assert all(hasattr(t, "content") for t in r)
            elif r is None:
                continue
            else:
                assert False, f"Unexpected result type: {type(r)}"

    def test_call_with_single(
        self: "TestTableChef",
        table_chef: TableChef,
        csv_content: str,
        tmp_path: Path,
    ) -> None:
        """Test calling TableChef with a single file path."""
        file1 = tmp_path / "a.csv"
        file1.write_text(csv_content)
        result = table_chef(file1)
        assert hasattr(result, "content")
        content_str = str(result.content)
        assert "col1" in content_str and "col2" in content_str
        assert (
            "1" in content_str and "2" in content_str and "3" in content_str and "4" in content_str
        )

    def test_call_invalid_type(self: "TestTableChef", table_chef: TableChef) -> None:
        """Test that TableChef raises TypeError on invalid input type."""
        with pytest.raises(TypeError, match="Unsupported type"):
            table_chef(123)  # type: ignore

    def test_extract_tables_from_markdown_multiple(
        self: "TestTableChef",
        table_chef: TableChef,
    ) -> None:
        """Test extracting multiple tables from markdown text."""
        md = """
| a | b |
|---|---|
| 1 | 2 |

Some text

| c | d |
|---|---|
| 3 | 4 |
"""
        tables = table_chef.extract_tables_from_markdown(md)
        assert isinstance(tables, list)
        assert len(tables) == 2
        assert all(hasattr(t, "content") and "|" in t.content for t in tables)

    def test_repr(self: "TestTableChef", table_chef: TableChef) -> None:
        """Test the __repr__ method of TableChef."""
        assert repr(table_chef) == "TableChef()"
