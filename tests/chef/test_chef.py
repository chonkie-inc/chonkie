
"""Tests for the chef module."""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from chonkie.chef import BaseChef, TableChef, TextChef


class ConcreteChef(BaseChef):
    """Concrete implementation of BaseChef for testing."""
    
    def process(self, path: str) -> str:
        """Test implementation that returns the path."""
        return f"processed: {path}"


class TestBaseChef:
    """Test cases for BaseChef abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseChef cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChef()
    
    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclass can be instantiated."""
        chef = ConcreteChef()
        assert isinstance(chef, BaseChef)
    
    def test_call_delegates_to_process(self):
        """Test that __call__ method delegates to process method."""
        chef = ConcreteChef()
        result = chef("test_path")
        assert result == "processed: test_path"
    
    def test_repr_method(self):
        """Test __repr__ method returns correct string."""
        chef = ConcreteChef()
        assert repr(chef) == "ConcreteChef()"


class TestTableChef:
    """Test cases for TableChef class."""

    @pytest.fixture
    def table_chef(self):
        return TableChef()

    @pytest.fixture
    def csv_content(self):
        return """Name,Age,City\nAlice,25,New York\nBob,30,Paris\n"""

    @pytest.fixture
    def excel_content(self, tmp_path):
        # Create a temporary Excel file
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        excel_path = tmp_path / "test.xlsx"
        df.to_excel(excel_path, index=False)
        return excel_path

    @pytest.fixture
    def markdown_tables(self):
        return """
        | Name | Age | City |
        |------|-----|------|
        | Alice | 25 | New York |
        | Bob | 30 | Paris |
        """

    def test_process_csv_file(self, table_chef, csv_content, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        df = table_chef.process(csv_path)
        assert not df.empty
        assert list(df.columns) == ["Name", "Age", "City"]
        assert df.iloc[0]["Name"] == "Alice"

    def test_process_excel_file(self, table_chef, excel_content):
        df = table_chef.process(excel_content)
        assert not df.empty
        assert list(df.columns) == ["A", "B"]
        assert df.iloc[0]["A"] == 1

    def test_process_markdown_table(self, table_chef, markdown_tables):
        df = table_chef.process(markdown_tables)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Note: column names and values will have surrounding spaces due to the markdown format
        assert list(df.columns) == [' Name ', ' Age ', ' City ']
        assert df.shape == (2, 3)
        assert df.iloc[0][' Name '] == ' Alice '

    def test_process_batch_csv(self, table_chef, csv_content, tmp_path):
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        csv1.write_text(csv_content)
        csv2.write_text(csv_content)
        dfs = table_chef.process_batch([csv1, csv2])
        assert len(dfs) == 2
        assert all(isinstance(df, pd.DataFrame) for df in dfs)

    def test_call_with_csv_path(self, table_chef, csv_content, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        df = table_chef(csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_call_with_list_of_csv_paths(self, table_chef, csv_content, tmp_path):
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        csv1.write_text(csv_content)
        csv2.write_text(csv_content)
        dfs = table_chef([csv1, csv2])
        assert isinstance(dfs, list)
        assert all(isinstance(df, pd.DataFrame) for df in dfs)

    def test_call_with_invalid_type(self, table_chef):
        with pytest.raises(TypeError):
            table_chef(123)

    def test_extract_tables_from_markdown(self, table_chef, markdown_tables):
        tables = table_chef.extract_tables_from_markdown(markdown_tables)
        assert isinstance(tables, list)
        assert len(tables) >= 1

    def test_process_markdown_with_no_table(self, table_chef):
        result = table_chef.process("no tables here!")
        assert result is None

    def test_repr_method(self, table_chef):
        assert repr(table_chef) == "TableChef()"
class TestTextChef:
    """Test cases for TextChef class."""
    
    @pytest.fixture
    def text_chef(self):
        """Fixture that returns a TextChef instance."""
        return TextChef()
    
    @pytest.fixture
    def sample_text(self):
        """Fixture that returns sample text content."""
        return "This is a sample text file content.\nWith multiple lines.\nFor testing purposes."
    
    def test_initialization(self, text_chef):
        """Test TextChef can be instantiated."""
        assert isinstance(text_chef, TextChef)
        assert isinstance(text_chef, BaseChef)
    
    def test_process_single_file_string_path(self, text_chef, sample_text):
        """Test processing a single file with string path."""
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef.process("test_file.txt")
            assert result == sample_text
    
    def test_process_single_file_path_object(self, text_chef, sample_text):
        """Test processing a single file with Path object."""
        path_obj = Path("test_file.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef.process(path_obj)
            assert result == sample_text
    
    def test_process_batch_string_paths(self, text_chef, sample_text):
        """Test processing multiple files with string paths."""
        paths = ["file1.txt", "file2.txt", "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result == sample_text for result in results)
    
    def test_process_batch_path_objects(self, text_chef, sample_text):
        """Test processing multiple files with Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt"), Path("file3.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result == sample_text for result in results)
    
    def test_process_batch_mixed_path_types(self, text_chef, sample_text):
        """Test processing multiple files with mixed path types."""
        paths = ["file1.txt", Path("file2.txt"), "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result == sample_text for result in results)
    
    def test_call_single_string_path(self, text_chef, sample_text):
        """Test __call__ method with single string path."""
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef("test_file.txt")
            assert result == sample_text
            assert isinstance(result, str)
    
    def test_call_single_path_object(self, text_chef, sample_text):
        """Test __call__ method with single Path object."""
        path_obj = Path("test_file.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef(path_obj)
            assert result == sample_text
            assert isinstance(result, str)
    
    def test_call_list_of_strings(self, text_chef, sample_text):
        """Test __call__ method with list of string paths."""
        paths = ["file1.txt", "file2.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result == sample_text for result in results)
    
    def test_call_list_of_path_objects(self, text_chef, sample_text):
        """Test __call__ method with list of Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result == sample_text for result in results)
    
    def test_call_tuple_of_paths(self, text_chef, sample_text):
        """Test __call__ method with tuple of paths."""
        paths = ("file1.txt", "file2.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result == sample_text for result in results)
    
    def test_call_invalid_type_raises_error(self, text_chef):
        """Test __call__ method with invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(123)
    
    def test_call_invalid_type_none_raises_error(self, text_chef):
        """Test __call__ method with None raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(None)
    
    def test_file_not_found_error(self, text_chef):
        """Test handling of FileNotFoundError."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                text_chef.process("nonexistent_file.txt")
    
    def test_permission_error(self, text_chef):
        """Test handling of PermissionError."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                text_chef.process("restricted_file.txt")
    
    def test_empty_file_content(self, text_chef):
        """Test processing empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = text_chef.process("empty_file.txt")
            assert result == ""
    
    def test_file_with_unicode_content(self, text_chef):
        """Test processing file with unicode content."""
        unicode_text = "Hello 世界! 🌍 Café naïve résumé"
        with patch("builtins.open", mock_open(read_data=unicode_text)):
            result = text_chef.process("unicode_file.txt")
            assert result == unicode_text
    
    def test_repr_method(self, text_chef):
        """Test __repr__ method returns correct string."""
        assert repr(text_chef) == "TextChef()"
    
    def test_file_opened_with_correct_mode(self, text_chef, sample_text):
        """Test that files are opened in read mode."""
        mock_file = mock_open(read_data=sample_text)
        with patch("builtins.open", mock_file):
            text_chef.process("test_file.txt")
            mock_file.assert_called_once_with("test_file.txt", "r")
    


    def test_batch_processing_calls_process_for_each_file(self, text_chef, sample_text):
        """Test that batch processing calls process method for each file."""
        paths = ["file1.txt", "file2.txt"]
        with patch.object(text_chef, 'process', return_value=sample_text) as mock_process:
            results = text_chef.process_batch(paths)
            assert mock_process.call_count == 2
            assert len(results) == 2
            mock_process.assert_any_call("file1.txt")
            mock_process.assert_any_call("file2.txt")

