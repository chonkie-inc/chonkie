"""Tests for the chef module."""

from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from chonkie.chef import BaseChef, TableChef, TextChef
from chonkie.types import Document


class ConcreteChef(BaseChef):
    """Concrete implementation of BaseChef for testing."""

    def process(self, path: str) -> str:
        """Test implementation that returns the path."""
        return f"processed: {path}"


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
        self: "TestTableChef", table_chef: TableChef, csv_content: str, tmp_path: Path, monkeypatch: Any
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
        result = table_chef.process(str(csv_file))
        assert isinstance(result, str)
        # Check for column names and values, not exact formatting
        assert "col1" in result and "col2" in result
        assert "1" in result and "2" in result and "3" in result and "4" in result
        assert called["called"]

    def test_process_excel_file(
        self: "TestTableChef", table_chef: TableChef, excel_df: pd.DataFrame, tmp_path: Path, monkeypatch: Any
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
        result = table_chef.process(str(excel_file))
        assert isinstance(result, str)
        assert "col1" in result and "col2" in result
        assert "1" in result and "2" in result and "3" in result and "4" in result
        assert called["called"]

    def test_process_markdown_table_string(
        self: "TestTableChef", table_chef: TableChef, markdown_table: str
    ) -> None:
        """Test processing a markdown table string."""
        tables = table_chef.process(markdown_table)
        assert isinstance(tables, list)
        assert len(tables) == 1
        assert hasattr(tables[0], "content")

    def test_process_batch(
        self: "TestTableChef", table_chef: TableChef, csv_content: str, tmp_path: Path
    ) -> None:
        """Test batch processing of multiple CSV files."""
        file1 = tmp_path / "a.csv"
        file2 = tmp_path / "b.csv"
        file1.write_text(csv_content)
        file2.write_text(csv_content)
        results = table_chef.process_batch([str(file1), str(file2)])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            if isinstance(r, str):
                assert "col1" in r and "col2" in r
                assert "1" in r and "2" in r and "3" in r and "4" in r
            elif isinstance(r, list):
                assert all(hasattr(t, "content") for t in r)
            else:
                assert False, f"Unexpected result type: {type(r)}"

    def test_call_with_list(
        self: "TestTableChef", table_chef: TableChef, csv_content: str, tmp_path: Path
    ) -> None:
        """Test calling TableChef with a list of file paths."""
        file1 = tmp_path / "a.csv"
        file2 = tmp_path / "b.csv"
        file1.write_text(csv_content)
        file2.write_text(csv_content)
        results = table_chef([str(file1), str(file2)])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            if isinstance(r, str):
                assert "col1" in r and "col2" in r
                assert "1" in r and "2" in r and "3" in r and "4" in r
            elif isinstance(r, list):
                assert all(hasattr(t, "content") for t in r)
            else:
                assert False, f"Unexpected result type: {type(r)}"

    def test_call_with_single(
        self: "TestTableChef", table_chef: TableChef, csv_content: str, tmp_path: Path
    ) -> None:
        """Test calling TableChef with a single file path."""
        file1 = tmp_path / "a.csv"
        file1.write_text(csv_content)
        result = table_chef(str(file1))
        assert isinstance(result, str)
        assert "col1" in result and "col2" in result
        assert "1" in result and "2" in result and "3" in result and "4" in result

    def test_call_invalid_type(self: "TestTableChef", table_chef: TableChef) -> None:
        """Test that TableChef raises TypeError on invalid input type."""
        with pytest.raises(TypeError, match="Unsupported type"):
            table_chef(123)  # type: ignore

    def test_extract_tables_from_markdown_multiple(self: "TestTableChef", table_chef: TableChef) -> None:
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


class TestBaseChef:
    """Test cases for BaseChef abstract class."""

    def test_cannot_instantiate_abstract_class(self: "TestBaseChef") -> None:
        """Test that BaseChef cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChef()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self: "TestBaseChef") -> None:
        """Test that concrete subclass can be instantiated."""
        chef = ConcreteChef()
        assert isinstance(chef, BaseChef)

    def test_call_delegates_to_process(self: "TestBaseChef") -> None:
        """Test that __call__ method delegates to process method."""
        chef = ConcreteChef()
        result = chef("test_path")
        assert result == "processed: test_path"

    def test_repr_method(self: "TestBaseChef") -> None:
        """Test __repr__ method returns correct string."""
        chef = ConcreteChef()
        assert repr(chef) == "ConcreteChef()"


class TestTextChef:
    """Test cases for TextChef class."""

    @pytest.fixture
    def text_chef(self) -> TextChef:
        """Fixture that returns a TextChef instance."""
        return TextChef()

    @pytest.fixture
    def sample_text(self) -> str:
        """Fixture that returns sample text content."""
        return "This is a sample text file content.\nWith multiple lines.\nFor testing purposes."

    def test_initialization(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test TextChef can be instantiated."""
        assert isinstance(text_chef, TextChef)
        assert isinstance(text_chef, BaseChef)

    def test_process_single_file_string_path(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test processing a single file with string path."""
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef.process("test_file.txt")
            if isinstance(result, list):
                assert all(r.content == sample_text for r in result)
            else:
                if isinstance(result, list):
                    assert all(r.content == sample_text for r in result)
                else:
                    assert result.content == sample_text

    def test_process_single_file_path_object(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test processing a single file with Path object."""
        path_obj = Path("test_file.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef.process(path_obj)
            if isinstance(result, list):
                assert all(r.content == sample_text for r in result)
            else:
                assert result.content == sample_text

    def test_process_batch_string_paths(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test processing multiple files with string paths."""
        paths = ["file1.txt", "file2.txt", "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_process_batch_path_objects(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test processing multiple files with Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt"), Path("file3.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_process_batch_mixed_path_types(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test processing multiple files with mixed path types."""
        paths = ["file1.txt", Path("file2.txt"), "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch([str(p) for p in paths])
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_call_single_string_path(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test __call__ method with single string path."""
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef("test_file.txt")
            if isinstance(result, list):
                assert all(r.content == sample_text for r in result)
                assert all(isinstance(r, Document) for r in result)
            else:
                assert result.content == sample_text
                assert isinstance(result, Document)

    def test_call_single_path_object(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test __call__ method with single Path object."""
        path_obj = Path("test_file.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef(path_obj)
            assert result.content == sample_text
            assert isinstance(result, Document)

    def test_call_list_of_strings(self: "TestTextChef", text_chef: TextChef, sample_text: str) -> None:
        """Test __call__ method with list of string paths."""
        paths = ["file1.txt", "file2.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_list_of_path_objects(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test __call__ method with list of Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_tuple_of_paths(self: "TestTextChef", text_chef: TextChef, sample_text: str) -> None:
        """Test __call__ method with tuple of paths."""
        paths = ("file1.txt", "file2.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_invalid_type_raises_error(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test __call__ method with invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(123)

    def test_call_invalid_type_none_raises_error(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test __call__ method with None raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(None)

    def test_file_not_found_error(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test handling of FileNotFoundError."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                text_chef.process("nonexistent_file.txt")

    def test_permission_error(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test handling of PermissionError."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                text_chef.process("restricted_file.txt")

    def test_empty_file_content(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test processing empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = text_chef.process("empty_file.txt")
            assert result.content == ""

    def test_file_with_unicode_content(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test processing file with unicode content."""
        unicode_text = "Hello 世界! 🌍 Café naïve résumé"
        with patch("builtins.open", mock_open(read_data=unicode_text)):
            result = text_chef.process("unicode_file.txt")
            assert result.content == unicode_text

    def test_repr_method(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test __repr__ method returns correct string."""
        assert repr(text_chef) == "TextChef()"

    def test_file_opened_with_correct_mode(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test that files are opened in read mode."""
        mock_file = mock_open(read_data=sample_text)
        with patch("builtins.open", mock_file):
            text_chef.process("test_file.txt")
            mock_file.assert_called_once_with("test_file.txt", "r", encoding="utf-8")

    def test_batch_processing_calls_process_for_each_file(
        self: "TestTextChef", text_chef: TextChef, sample_text: str
    ) -> None:
        """Test that batch processing calls process method for each file."""
        paths = ["file1.txt", "file2.txt"]
        with patch.object(
            text_chef, "process", return_value=sample_text
        ) as mock_process:
            results = text_chef.process_batch(paths)
            assert mock_process.call_count == 2
            assert len(results) == 2
            mock_process.assert_any_call("file1.txt")
            mock_process.assert_any_call("file2.txt")
