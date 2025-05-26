"""Comprehensive tests for the chonkie viz module."""

import html
import os
import tempfile
import warnings
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from chonkie.types import Chunk
from chonkie.utils.viz import (
    BODY_BACKGROUND_COLOR_DARK,
    BODY_BACKGROUND_COLOR_LIGHT,
    CONTENT_BACKGROUND_COLOR_DARK,
    CONTENT_BACKGROUND_COLOR_LIGHT,
    DARK_THEMES,
    FOOTER_TEMPLATE,
    HTML_TEMPLATE,
    LIGHT_THEMES,
    MAIN_TEMPLATE,
    TEXT_COLOR_DARK,
    TEXT_COLOR_LIGHT,
    Visualizer,
)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(text="Hello ", start_index=0, end_index=6, token_count=2),
        Chunk(text="world! ", start_index=6, end_index=13, token_count=2),
        Chunk(text="This is ", start_index=13, end_index=21, token_count=3),
        Chunk(text="a test.", start_index=21, end_index=28, token_count=3),
    ]


@pytest.fixture
def sample_text() -> str:
    """Sample text that corresponds to the sample chunks."""
    return "Hello world! This is a test."


@pytest.fixture
def overlapping_chunks() -> List[Chunk]:
    """Create overlapping chunks for testing."""
    return [
        Chunk(text="Hello world", start_index=0, end_index=11, token_count=2),
        Chunk(text="world! This", start_index=6, end_index=17, token_count=3),
        Chunk(text="This is a test", start_index=13, end_index=27, token_count=4),
    ]


@pytest.fixture
def empty_chunks() -> List[Chunk]:
    """Create empty chunks list for testing."""
    return []


class TestVisualizerInitialization:
    """Test Visualizer initialization and theme handling."""

    def test_init_default_theme(self) -> None:
        """Test initialization with default theme."""
        with patch('rich.console.Console') as mock_console:
            viz = Visualizer()
            assert viz.theme_name == "pastel"
            assert viz.theme == LIGHT_THEMES["pastel"]
            assert viz.text_color == TEXT_COLOR_LIGHT
            mock_console.assert_called_once()

    def test_init_custom_list_theme(self) -> None:
        """Test initialization with custom list theme."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        with patch('rich.console.Console'):
            viz = Visualizer(theme=custom_colors)
            assert viz.theme_name == "custom"
            assert viz.theme == custom_colors
            assert viz.text_color == ""

    def test_init_light_theme(self) -> None:
        """Test initialization with light theme."""
        with patch('rich.console.Console'):
            viz = Visualizer(theme="tiktokenizer")
            assert viz.theme_name == "tiktokenizer"
            assert viz.theme == LIGHT_THEMES["tiktokenizer"]
            assert viz.text_color == TEXT_COLOR_LIGHT

    def test_init_dark_theme(self) -> None:
        """Test initialization with dark theme."""
        with patch('rich.console.Console'):
            viz = Visualizer(theme="tiktokenizer_dark")
            assert viz.theme_name == "tiktokenizer_dark"
            assert viz.theme == DARK_THEMES["tiktokenizer_dark"]
            assert viz.text_color == TEXT_COLOR_DARK

    def test_init_invalid_theme(self) -> None:
        """Test initialization with invalid theme."""
        with patch('rich.console.Console'):
            with pytest.raises(ValueError, match="Invalid theme"):
                Visualizer(theme="invalid_theme")

    def test_import_dependencies_success(self) -> None:
        """Test successful import of dependencies."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            viz._import_dependencies()
            # Should not raise any exception

    def test_import_dependencies_failure(self) -> None:
        """Test import failure handling."""
        with patch('builtins.__import__', side_effect=ImportError("Missing rich")):
            with pytest.raises(ImportError, match="Could not import dependencies"):
                Visualizer()


class TestVisualizerThemeManagement:
    """Test theme management methods."""

    def test_get_theme_light(self) -> None:
        """Test getting light theme."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            theme, text_color = viz._get_theme("pastel")
            assert theme == LIGHT_THEMES["pastel"]
            assert text_color == TEXT_COLOR_LIGHT

    def test_get_theme_dark(self) -> None:
        """Test getting dark theme."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            theme, text_color = viz._get_theme("pastel_dark")
            assert theme == DARK_THEMES["pastel_dark"]
            assert text_color == TEXT_COLOR_DARK

    def test_get_color_cycling(self) -> None:
        """Test color cycling behavior."""
        with patch('rich.console.Console'):
            viz = Visualizer(theme="pastel")
            # Test normal indexing
            assert viz._get_color(0) == LIGHT_THEMES["pastel"][0]
            assert viz._get_color(1) == LIGHT_THEMES["pastel"][1]
            # Test cycling behavior
            theme_length = len(LIGHT_THEMES["pastel"])
            assert viz._get_color(theme_length) == LIGHT_THEMES["pastel"][0]
            assert viz._get_color(theme_length + 1) == LIGHT_THEMES["pastel"][1]

    def test_darken_color_valid_hex(self) -> None:
        """Test color darkening with valid hex colors."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            # Test 6-digit hex
            darkened = viz._darken_color("#FF0000", 0.5)
            assert darkened == "#7f0000"
            # Test 3-digit hex
            darkened = viz._darken_color("#F00", 0.5)
            assert darkened == "#7f0000"

    def test_darken_color_invalid_hex(self) -> None:
        """Test color darkening with invalid hex colors."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            with patch('builtins.print') as mock_print:
                result = viz._darken_color("invalid", 0.5)
                assert result == "#808080"
                mock_print.assert_called_once()

    def test_darken_color_edge_cases(self) -> None:
        """Test color darkening edge cases."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            # Test with amount 0 (should be black)
            darkened = viz._darken_color("#FFFFFF", 0.0)
            assert darkened == "#000000"
            # Test with amount 1 (should be unchanged)
            darkened = viz._darken_color("#FFFFFF", 1.0)
            assert darkened == "#ffffff"


class TestVisualizerPrintMethod:
    """Test the print method functionality."""

    def test_print_empty_chunks(self, sample_text: str) -> None:
        """Test printing empty chunks."""
        with patch('rich.console.Console') as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            
            viz = Visualizer()
            viz.print([])
            
            mock_console.print.assert_called_once_with("No chunks to visualize.")

    def test_print_with_full_text(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test printing with provided full text."""
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                viz.print(sample_chunks, sample_text)
                
                mock_text_class.assert_called_once_with(sample_text)
                mock_console.print.assert_called_once_with(mock_text)

    def test_print_without_full_text(self, sample_chunks: List[Chunk]) -> None:
        """Test printing without provided full text (reconstruction)."""
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                viz.print(sample_chunks)
                
                expected_text = "Hello world! This is a test."
                mock_text_class.assert_called_once_with(expected_text)

    def test_print_invalid_chunks_no_text_attr(self) -> None:
        """Test printing with chunks missing text attribute."""
        invalid_chunks = [MagicMock(spec=[])]  # No text attribute
        
        with patch('rich.console.Console'):
            viz = Visualizer()
            with pytest.raises(ValueError, match="Chunks must have 'text'"):
                viz.print(invalid_chunks)

    def test_print_invalid_chunk_indices(self, sample_text: str) -> None:
        """Test printing with chunks having invalid indices."""
        # Create chunks with invalid indices
        invalid_chunks = [
            MagicMock(text="Hello ", start_index=None, end_index=6),
            MagicMock(text="world!", start_index=6, end_index="invalid"),
        ]
        
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    viz.print(invalid_chunks, sample_text)
                    assert len(w) == 2
                    assert "invalid start/end index" in str(w[0].message)

    def test_print_stylize_error_handling(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test error handling during text stylization."""
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_text.stylize.side_effect = Exception("Stylize error")
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    viz.print(sample_chunks, sample_text)
                    assert len(w) >= 1
                    assert "Could not apply style" in str(w[0].message)


class TestVisualizerSaveMethod:
    """Test the save method functionality."""

    def test_save_empty_chunks(self) -> None:
        """Test saving with empty chunks."""
        with patch('rich.console.Console'):
            with patch('builtins.print') as mock_print:
                viz = Visualizer()
                viz.save("test.html", [])
                mock_print.assert_called_with("No chunks to visualize. HTML file not saved.")

    def test_save_with_full_text(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test saving with provided full text."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                with patch('builtins.print') as mock_print:
                    viz = Visualizer()
                    viz.save(filepath, sample_chunks, sample_text)
                    
                    # Check file was created
                    assert os.path.exists(filepath)
                    
                    # Check content
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check that the text content is present (it gets split into spans)
                    assert "Hello" in content
                    assert "world!" in content
                    assert "This is" in content
                    assert "a test." in content
                    assert 'Chunk Visualization' in content
                    assert 'background-color:' in content
                    mock_print.assert_called_once()
                    assert f"file://{os.path.abspath(filepath)}" in mock_print.call_args[0][0]

    def test_save_without_full_text(self, sample_chunks: List[Chunk]) -> None:
        """Test saving without provided full text (reconstruction)."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, sample_chunks)
                
                assert os.path.exists(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check that the text content is present (it gets split into spans)
                assert "Hello" in content
                assert "world!" in content
                assert "This is" in content
                assert "a test." in content

    def test_save_auto_add_html_extension(self, sample_chunks: List[Chunk]) -> None:
        """Test automatic addition of .html extension."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test")  # No extension
                
                viz = Visualizer()
                viz.save(filepath, sample_chunks)
                
                # Should create test.html
                expected_path = filepath + ".html"
                assert os.path.exists(expected_path)

    def test_save_custom_title(self, sample_chunks: List[Chunk]) -> None:
        """Test saving with custom title."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                custom_title = "My Custom Title"
                
                viz = Visualizer()
                viz.save(filepath, sample_chunks, title=custom_title)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                assert f"<title>{html.escape(custom_title)}</title>" in content

    def test_save_dark_theme_styling(self, sample_chunks: List[Chunk]) -> None:
        """Test saving with dark theme styling."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer(theme="pastel_dark")
                viz.save(filepath, sample_chunks)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                assert BODY_BACKGROUND_COLOR_DARK in content
                assert CONTENT_BACKGROUND_COLOR_DARK in content
                assert TEXT_COLOR_DARK in content

    def test_save_light_theme_styling(self, sample_chunks: List[Chunk]) -> None:
        """Test saving with light theme styling."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer(theme="pastel")
                viz.save(filepath, sample_chunks)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                assert BODY_BACKGROUND_COLOR_LIGHT in content
                assert CONTENT_BACKGROUND_COLOR_LIGHT in content
                assert TEXT_COLOR_LIGHT in content

    def test_save_overlapping_chunks(self, overlapping_chunks: List[Chunk]) -> None:
        """Test saving with overlapping chunks."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                full_text = "Hello world! This is a test."
                
                viz = Visualizer()
                viz.save(filepath, overlapping_chunks, full_text)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Should contain overlap indicators
                assert "(Overlap)" in content
                assert "title=" in content

    def test_save_hippo_favicon_embedding(self, sample_chunks: List[Chunk]) -> None:
        """Test that hippo favicon is properly embedded."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, sample_chunks)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for favicon link tag
                assert 'rel="icon"' in content
                assert 'type="image/svg+xml"' in content
                assert 'data:image/svg+xml;base64,' in content

    def test_save_favicon_encoding_error(self, sample_chunks: List[Chunk]) -> None:
        """Test handling of favicon encoding errors."""
        with patch('rich.console.Console'):
            with patch('base64.b64encode', side_effect=Exception("Encoding error")):
                with patch('builtins.print') as mock_print:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        filepath = os.path.join(tmpdir, "test.html")
                        
                        viz = Visualizer()
                        viz.save(filepath, sample_chunks)
                        
                        # Should print warning
                        mock_print.assert_any_call("Warning: Could not encode embedded hippo favicon: Encoding error")

    def test_save_invalid_chunks(self) -> None:
        """Test saving with invalid chunks."""
        invalid_chunks = [MagicMock(spec=[])]  # No required attributes
        
        with patch('rich.console.Console'):
            viz = Visualizer()
            with pytest.raises(AttributeError, match="Chunks must have 'text'"):
                viz.save("test.html", invalid_chunks)

    def test_save_file_write_error(self, sample_chunks: List[Chunk]) -> None:
        """Test handling of file write errors."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            # Try to write to a directory that doesn't exist
            with pytest.raises(IOError, match="Could not write file"):
                viz.save("/nonexistent/directory/test.html", sample_chunks)

    def test_save_text_escaping(self, sample_text: str) -> None:
        """Test proper HTML escaping of text content."""
        # Create chunks with HTML characters
        html_chunks = [
            Chunk(text="<script>", start_index=0, end_index=8, token_count=1),
            Chunk(text="alert('hi')", start_index=8, end_index=19, token_count=2),
            Chunk(text="</script>", start_index=19, end_index=28, token_count=1),
        ]
        html_text = "<script>alert('hi')</script>"
        
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, html_chunks, html_text)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # HTML should be escaped
                assert "&lt;script&gt;" in content
                assert "&lt;/script&gt;" in content
                assert "alert(&#x27;hi&#x27;)" in content

    def test_save_newline_handling(self) -> None:
        """Test proper handling of newlines in text."""
        newline_chunks = [
            Chunk(text="Line 1\n", start_index=0, end_index=7, token_count=2),
            Chunk(text="Line 2\n", start_index=7, end_index=14, token_count=2),
            Chunk(text="Line 3", start_index=14, end_index=20, token_count=2),
        ]
        newline_text = "Line 1\nLine 2\nLine 3"
        
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, newline_chunks, newline_text)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Newlines should be converted to <br>
                assert "Line 1<br>" in content
                assert "Line 2<br>" in content


class TestVisualizerCallMethod:
    """Test the __call__ method functionality."""

    def test_call_delegates_to_print(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test that __call__ delegates to print method."""
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                viz(sample_chunks, sample_text)
                
                mock_text_class.assert_called_once_with(sample_text)
                mock_console.print.assert_called_once_with(mock_text)

    def test_call_without_full_text(self, sample_chunks: List[Chunk]) -> None:
        """Test __call__ without provided full text."""
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                viz(sample_chunks)
                
                expected_text = "Hello world! This is a test."
                mock_text_class.assert_called_once_with(expected_text)


class TestVisualizerUtilityMethods:
    """Test utility methods."""

    def test_repr(self) -> None:
        """Test __repr__ method."""
        with patch('rich.console.Console'):
            viz = Visualizer(theme="pastel")
            repr_str = repr(viz)
            assert "Visualizer" in repr_str
            assert str(viz.theme) in repr_str


class TestVisualizerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text_chunks(self) -> None:
        """Test chunks with empty text."""
        empty_text_chunks = [
            Chunk(text="", start_index=0, end_index=0, token_count=0),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, empty_text_chunks, "Hello")
                
                assert os.path.exists(filepath)

    def test_chunks_beyond_text_length(self, sample_text: str) -> None:
        """Test chunks with indices beyond text length."""
        beyond_chunks = [
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
            Chunk(text="Beyond", start_index=100, end_index=106, token_count=1),  # Beyond text length
        ]
        
        with patch('rich.console.Console') as mock_console_class:
            with patch('rich.text.Text') as mock_text_class:
                mock_console = MagicMock()
                mock_text = MagicMock()
                mock_console_class.return_value = mock_console
                mock_text_class.return_value = mock_text
                
                viz = Visualizer()
                viz.print(beyond_chunks, sample_text)
                
                # Should handle gracefully without errors
                mock_text_class.assert_called_once_with(sample_text)

    def test_negative_chunk_indices(self, sample_text: str) -> None:
        """Test chunks with negative indices."""
        negative_chunks = [
            Chunk(text="Invalid", start_index=-5, end_index=5, token_count=1),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.html")
                
                viz = Visualizer()
                viz.save(filepath, negative_chunks, sample_text)
                
                # Should handle gracefully
                assert os.path.exists(filepath)

    def test_chunk_start_equals_end(self, sample_text: str) -> None:
        """Test chunks where start_index equals end_index."""
        equal_indices_chunks = [
            Chunk(text="", start_index=5, end_index=5, token_count=0),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        
        with patch('rich.console.Console'):
            viz = Visualizer()
            
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                viz.print(equal_indices_chunks, sample_text)
                # Should not produce warnings for valid zero-length chunks


class TestVisualizerConstants:
    """Test module constants and templates."""

    def test_theme_constants(self) -> None:
        """Test that theme constants are properly defined."""
        assert isinstance(LIGHT_THEMES, dict)
        assert isinstance(DARK_THEMES, dict)
        assert "pastel" in LIGHT_THEMES
        assert "tiktokenizer" in LIGHT_THEMES
        assert "pastel_dark" in DARK_THEMES
        assert "tiktokenizer_dark" in DARK_THEMES

    def test_color_constants(self) -> None:
        """Test that color constants are properly defined."""
        assert BODY_BACKGROUND_COLOR_LIGHT.startswith("#")
        assert CONTENT_BACKGROUND_COLOR_LIGHT.startswith("#")
        assert TEXT_COLOR_LIGHT.startswith("#")
        assert BODY_BACKGROUND_COLOR_DARK.startswith("#")
        assert CONTENT_BACKGROUND_COLOR_DARK.startswith("#")
        assert TEXT_COLOR_DARK.startswith("#")

    def test_template_constants(self) -> None:
        """Test that template constants are properly defined."""
        assert isinstance(HTML_TEMPLATE, str)
        assert isinstance(MAIN_TEMPLATE, str)
        assert isinstance(FOOTER_TEMPLATE, str)
        assert "{title}" in HTML_TEMPLATE
        assert "{html_parts}" in MAIN_TEMPLATE
        assert "Chonkie" in FOOTER_TEMPLATE

    def test_hippo_svg_content(self) -> None:
        """Test that hippo SVG content is properly defined."""
        with patch('rich.console.Console'):
            viz = Visualizer()
            assert "🦛" in viz.HIPPO_SVG_CONTENT
            assert "<svg" in viz.HIPPO_SVG_CONTENT
            assert "</svg>" in viz.HIPPO_SVG_CONTENT


class TestVisualizerIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_light_theme(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test complete workflow with light theme."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "integration_test.html")
                
                viz = Visualizer(theme="pastel")
                viz.save(filepath, sample_chunks, sample_text, title="Integration Test")
                
                # Verify file creation and content
                assert os.path.exists(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check all major components
                assert "Integration Test" in content
                assert "Hello" in content and "world!" in content  # Text is split into spans
                assert "background-color:" in content
                assert "🦛" in content or "data:image/svg+xml" in content
                assert BODY_BACKGROUND_COLOR_LIGHT in content

    def test_full_workflow_dark_theme(self, sample_chunks: List[Chunk], sample_text: str) -> None:
        """Test complete workflow with dark theme."""
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "integration_test_dark.html")
                
                viz = Visualizer(theme="pastel_dark")
                viz.save(filepath, sample_chunks, sample_text, title="Dark Theme Test")
                
                # Verify file creation and content
                assert os.path.exists(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check dark theme specific elements
                assert BODY_BACKGROUND_COLOR_DARK in content
                assert CONTENT_BACKGROUND_COLOR_DARK in content
                assert TEXT_COLOR_DARK in content

    def test_complex_overlapping_scenario(self) -> None:
        """Test complex overlapping chunks scenario."""
        complex_chunks = [
            Chunk(text="The quick", start_index=0, end_index=9, token_count=2),
            Chunk(text="quick brown", start_index=4, end_index=15, token_count=2),
            Chunk(text="brown fox", start_index=10, end_index=19, token_count=2),
            Chunk(text="fox jumps", start_index=16, end_index=25, token_count=2),
        ]
        complex_text = "The quick brown fox jumps"
        
        with patch('rich.console.Console'):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "complex_test.html")
                
                viz = Visualizer()
                viz.save(filepath, complex_chunks, complex_text)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Should handle multiple overlaps
                assert "(Overlap)" in content
                # Check that the text content is present (it gets split into spans)
                assert "The" in content
                assert "quick" in content
                assert "brown" in content
                assert "fox" in content
                assert "jumps" in content