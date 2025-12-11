"""Tests for the fetcher module."""

import tempfile
from pathlib import Path

import pytest

from chonkie.fetcher import BaseFetcher, FileFetcher


class ConcreteFetcher(BaseFetcher):
    """Concrete implementation of BaseFetcher for testing."""
    
    def fetch(self):
        """Test implementation that returns a test value."""
        return "fetched_data"


class TestBaseFetcher:
    """Test cases for BaseFetcher abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseFetcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFetcher()
    
    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclass can be instantiated."""
        fetcher = ConcreteFetcher()
        assert isinstance(fetcher, BaseFetcher)
    
    def test_concrete_subclass_init(self):
        """Test that concrete subclass initialization works."""
        fetcher = ConcreteFetcher()
        assert fetcher is not None
    
    def test_concrete_subclass_fetch_method(self):
        """Test that concrete subclass fetch method works."""
        fetcher = ConcreteFetcher()
        result = fetcher.fetch()
        assert result == "fetched_data"


class TestFileFetcher:
    """Test cases for FileFetcher class."""
    
    @pytest.fixture
    def file_fetcher(self):
        """Fixture that returns a FileFetcher instance."""
        return FileFetcher()
    
    @pytest.fixture
    def temp_dir_with_files(self):
        """Fixture that creates a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files with different extensions
            (temp_path / "file1.txt").write_text("Content of file1")
            (temp_path / "file2.py").write_text("print('Hello World')")
            (temp_path / "file3.md").write_text("# Markdown File")
            (temp_path / "file4.txt").write_text("Content of file4")
            (temp_path / "no_extension_file").write_text("File without extension")
            
            # Create a subdirectory with a file (should now be included by fetch)
            subdir = temp_path / "subdirectory"
            subdir.mkdir()
            (subdir / "nested_file.txt").write_text("Nested file content")
            
            yield temp_dir
    
    @pytest.fixture
    def empty_temp_dir(self):
        """Fixture that creates an empty temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_initialization(self, file_fetcher):
        """Test FileFetcher can be instantiated."""
        assert isinstance(file_fetcher, FileFetcher)
        assert isinstance(file_fetcher, BaseFetcher)
    
    def test_fetch_all_files_no_filter(self, file_fetcher, temp_dir_with_files):
        """Test fetching all files, now including nested files."""
        files = file_fetcher.fetch(dir=temp_dir_with_files)

        # Should return all files, including the nested one.
        assert len(files) == 6  # file1.txt, file2.py, file3.md, file4.txt, no_extension_file, nested_file.txt
        
        # Verify all returned items are Path objects and files
        for file_path in files:
            assert isinstance(file_path, Path)
            assert file_path.is_file()
        
        # Check that specific files are present
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.py" in file_names
        assert "file3.md" in file_names
        assert "file4.txt" in file_names
        assert "no_extension_file" in file_names
        assert "nested_file.txt" in file_names # Verify the nested file is found
        
        # Verify subdirectory is not included
        assert "subdirectory" not in file_names
    
    def test_fetch_with_single_extension_filter(self, file_fetcher, temp_dir_with_files):
        """Test fetching files with single extension filter."""
        files = file_fetcher.fetch(dir=temp_dir_with_files, ext=[".txt"])
        
        # NOTE: This test now implicitly tests recursion. It should find 3 .txt files.
        assert len(files) == 3  # file1.txt, file4.txt, nested_file.txt
        
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file4.txt" in file_names
        assert "nested_file.txt" in file_names # Check for nested .txt file
        assert "file2.py" not in file_names
        assert "file3.md" not in file_names
    
    def test_fetch_with_multiple_extension_filters(self, file_fetcher, temp_dir_with_files):
        """Test fetching files with multiple extension filters."""
        files = file_fetcher.fetch(dir=temp_dir_with_files, ext=[".txt", ".py"])
        
        # NOTE: This test now implicitly tests recursion. It should find 4 files.
        assert len(files) == 4  # file1.txt, file2.py, file4.txt, nested_file.txt
        
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.py" in file_names
        assert "file4.txt" in file_names
        assert "nested_file.txt" in file_names
        assert "file3.md" not in file_names
        assert "no_extension_file" not in file_names
    
    def test_fetch_with_non_matching_extension(self, file_fetcher, temp_dir_with_files):
        """Test fetching files with extension that doesn't match any files."""
        files = file_fetcher.fetch(dir=temp_dir_with_files, ext=[".xyz"])
        
        # Should return empty list
        assert len(files) == 0
        assert files == []
    
    def test_fetch_empty_directory(self, file_fetcher, empty_temp_dir):
        """Test fetching files from empty directory."""
        files = file_fetcher.fetch(dir=empty_temp_dir)
        
        # Should return empty list
        assert len(files) == 0
        assert files == []
    
    def test_fetch_non_existent_directory(self, file_fetcher):
        """Test fetching files from non-existent directory."""
        with pytest.raises((FileNotFoundError, ValueError)):
            file_fetcher.fetch(dir="/non/existent/directory")
    
    def test_fetch_file_existing_file(self, file_fetcher, temp_dir_with_files):
        """Test fetch_file method with existing file."""
        file_path = file_fetcher.fetch_file(temp_dir_with_files, "file1.txt")
        
        assert isinstance(file_path, Path)
        assert file_path.name == "file1.txt"
        assert file_path.is_file()
        assert file_path.read_text() == "Content of file1"
    
    def test_fetch_file_non_existent_file(self, file_fetcher, temp_dir_with_files):
        """Test fetch_file method with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File nonexistent.txt not found in directory"):
            file_fetcher.fetch_file(temp_dir_with_files, "nonexistent.txt")
    
    def test_fetch_file_non_existent_directory(self, file_fetcher):
        """Test fetch_file method with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            file_fetcher.fetch_file("/non/existent/directory", "any_file.txt")
    
    def test_fetch_file_with_no_extension_file(self, file_fetcher, temp_dir_with_files):
        """Test fetch_file method with file that has no extension."""
        file_path = file_fetcher.fetch_file(temp_dir_with_files, "no_extension_file")
        
        assert isinstance(file_path, Path)
        assert file_path.name == "no_extension_file"
        assert file_path.is_file()
        assert file_path.read_text() == "File without extension"
    
    def test_call_method_delegates_to_fetch(self, file_fetcher, temp_dir_with_files):
        """Test __call__ method delegates to fetch method."""
        # Test without extension filter
        files1 = file_fetcher(dir=temp_dir_with_files)
        files2 = file_fetcher.fetch(dir=temp_dir_with_files)
        
        assert len(files1) == len(files2)
        assert set(f.name for f in files1) == set(f.name for f in files2)
    
    def test_call_method_with_extension_filter(self, file_fetcher, temp_dir_with_files):
        """Test __call__ method with extension filter."""
        files1 = file_fetcher(dir=temp_dir_with_files, ext=[".txt"])
        files2 = file_fetcher.fetch(dir=temp_dir_with_files, ext=[".txt"])
        
        assert len(files1) == len(files2)
        assert set(f.name for f in files1) == set(f.name for f in files2)
        assert len(files1) == 3  # Should have 3 .txt files now
    
    def test_fetch_ignores_subdirectories(self, file_fetcher, temp_dir_with_files):
        """Test that fetch method returns files, not the directories themselves."""
        files = file_fetcher.fetch(dir=temp_dir_with_files)
        
        # Verify no directories are returned
        for file_path in files:
            assert file_path.is_file()
            assert not file_path.is_dir()
        
        # Verify subdirectory name is not in the results as a file
        file_names = [f.name for f in files]
        assert "subdirectory" not in file_names
    
    def test_fetch_with_mixed_case_extensions(self, file_fetcher):
        """Test fetch with mixed case extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with mixed case extensions
            (temp_path / "file1.TXT").write_text("Content 1")
            (temp_path / "file2.txt").write_text("Content 2")
            (temp_path / "file3.Txt").write_text("Content 3")
            
            # Test exact case matching
            files_lower = file_fetcher.fetch(dir=temp_dir, ext=[".txt"])
            files_upper = file_fetcher.fetch(dir=temp_dir, ext=[".TXT"])
            files_mixed = file_fetcher.fetch(dir=temp_dir, ext=[".Txt"])
            
            assert len(files_lower) == 1  # Only file2.txt
            assert len(files_upper) == 1  # Only file1.TXT
            assert len(files_mixed) == 1  # Only file3.Txt
            
            assert files_lower[0].name == "file2.txt"
            assert files_upper[0].name == "file1.TXT"
            assert files_mixed[0].name == "file3.Txt"

    def test_fetch_single_file_mode(self, file_fetcher, temp_dir_with_files):
        """Test fetching a single file using path parameter."""
        temp_path = Path(temp_dir_with_files)
        file_path = temp_path / "file1.txt"

        result = file_fetcher.fetch(path=str(file_path))

        assert isinstance(result, Path)
        assert result.name == "file1.txt"
        assert result.is_file()
        assert result.read_text() == "Content of file1"

    def test_fetch_single_file_not_found(self, file_fetcher, temp_dir_with_files):
        """Test fetching a non-existent file in single file mode."""
        temp_path = Path(temp_dir_with_files)
        file_path = temp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="File not found"):
            file_fetcher.fetch(path=str(file_path))

    def test_fetch_both_path_and_dir_raises_error(self, file_fetcher, temp_dir_with_files):
        """Test that providing both path and dir raises ValueError."""
        with pytest.raises(ValueError, match="Provide either 'path' or 'dir', not both"):
            file_fetcher.fetch(path="file.txt", dir=temp_dir_with_files)

    def test_fetch_neither_path_nor_dir_raises_error(self, file_fetcher):
        """Test that providing neither path nor dir raises ValueError."""
        with pytest.raises(ValueError, match="Must provide either 'path' or 'dir'"):
            file_fetcher.fetch()

    def test_fetch_single_file_with_ext_ignored(self, file_fetcher, temp_dir_with_files):
        """Test that ext parameter is ignored in single file mode."""
        temp_path = Path(temp_dir_with_files)
        file_path = temp_path / "file2.py"

        # ext should be ignored when path is provided
        result = file_fetcher.fetch(path=str(file_path), ext=[".txt"])

        assert isinstance(result, Path)
        assert result.name == "file2.py"
        assert result.suffix == ".py"  # Not .txt

    def test_call_single_file_mode(self, file_fetcher, temp_dir_with_files):
        """Test __call__ method with single file."""
        temp_path = Path(temp_dir_with_files)
        file_path = temp_path / "file1.txt"

        result = file_fetcher(path=str(file_path))

        assert isinstance(result, Path)
        assert result.name == "file1.txt"