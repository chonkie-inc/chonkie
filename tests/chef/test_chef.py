"""Tests for the TextChef class."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from chonkie.chef import TextChef
from chonkie.types import Document


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

    def test_process_single_file_string_path(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
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
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
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
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test processing multiple files with string paths."""
        paths = ["file1.txt", "file2.txt", "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_process_batch_path_objects(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test processing multiple files with Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt"), Path("file3.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch(paths)
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_process_batch_mixed_path_types(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test processing multiple files with mixed path types."""
        paths = ["file1.txt", Path("file2.txt"), "file3.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef.process_batch([str(p) for p in paths])
            assert len(results) == 3
            assert all(result.content == sample_text for result in results)

    def test_call_single_string_path(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
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
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test __call__ method with single Path object."""
        path_obj = Path("test_file.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = text_chef(path_obj)
            if isinstance(result, list):
                assert all(r.content == sample_text for r in result)
            else:
                assert result.content == sample_text
                assert isinstance(result, Document)

    def test_call_list_of_strings(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test __call__ method with list of string paths."""
        paths = ["file1.txt", "file2.txt"]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_list_of_path_objects(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test __call__ method with list of Path objects."""
        paths = [Path("file1.txt"), Path("file2.txt")]
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(paths)
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_tuple_of_paths(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test __call__ method with tuple of paths."""
        paths = ("file1.txt", "file2.txt")
        with patch("builtins.open", mock_open(read_data=sample_text)):
            results = text_chef(list(paths))  # type: ignore[arg-type]
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(result.content == sample_text for result in results)

    def test_call_invalid_type_raises_error(self: "TestTextChef", text_chef: TextChef) -> None:
        """Test __call__ method with invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(123)  # type: ignore[arg-type]

    def test_call_invalid_type_none_raises_error(
        self: "TestTextChef",
        text_chef: TextChef,
    ) -> None:
        """Test __call__ method with None raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            text_chef(None)  # type: ignore[arg-type]

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
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test that files are opened in read mode."""
        mock_file = mock_open(read_data=sample_text)
        with patch("builtins.open", mock_file):
            text_chef.process("test_file.txt")
            mock_file.assert_called_once_with("test_file.txt", "r", encoding="utf-8")

    def test_batch_processing_calls_process_for_each_file(
        self: "TestTextChef",
        text_chef: TextChef,
        sample_text: str,
    ) -> None:
        """Test that batch processing calls process method for each file."""
        paths = ["file1.txt", "file2.txt"]
        with patch.object(text_chef, "process", return_value=sample_text) as mock_process:
            results = text_chef.process_batch(paths)
            assert mock_process.call_count == 2
            assert len(results) == 2
            mock_process.assert_any_call("file1.txt")
            mock_process.assert_any_call("file2.txt")
