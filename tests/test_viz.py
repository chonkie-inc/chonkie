"""Comprehensive tests for the chonkie viz module."""

import html
import logging
import os
from pathlib import Path
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
def sample_chunks() -> list[Chunk]:
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
def overlapping_chunks() -> list[Chunk]:
    """Create overlapping chunks for testing."""
    return [
        Chunk(text="Hello world", start_index=0, end_index=11, token_count=2),
        Chunk(text="world! This", start_index=6, end_index=17, token_count=3),
        Chunk(text="This is a test", start_index=13, end_index=27, token_count=4),
    ]


@pytest.fixture
def empty_chunks() -> list[Chunk]:
    """Create empty chunks list for testing."""
    return []


@pytest.fixture()
def mock_console_instance() -> MagicMock:
    from rich.console import Console

    return MagicMock(spec=Console)


@pytest.fixture(autouse=True)
def mock_console_class(mock_console_instance) -> MagicMock:
    with patch("rich.console.Console") as mock_console_class:
        mock_console_class.return_value = mock_console_instance
        yield mock_console_class


@pytest.fixture()
def mock_text_instance() -> MagicMock:
    from rich.text import Text

    return MagicMock(spec=Text)


@pytest.fixture()
def mock_text_class(mock_text_instance) -> MagicMock:
    with patch("rich.text.Text") as mock_text_class:
        mock_text_class.return_value = mock_text_instance
        yield mock_text_class


class TestVisualizerInitialization:
    """Test Visualizer initialization and theme handling."""

    def test_init_default_theme(self, mock_console_class) -> None:
        """Test initialization with default theme."""
        viz = Visualizer()
        assert viz.theme_name == "pastel"
        assert viz.theme == LIGHT_THEMES["pastel"]
        assert viz.text_color == TEXT_COLOR_LIGHT
        mock_console_class.assert_called_once()

    def test_init_custom_list_theme(self) -> None:
        """Test initialization with custom list theme."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        viz = Visualizer(theme=custom_colors)
        assert viz.theme_name == "custom"
        assert viz.theme == custom_colors
        assert viz.text_color == ""

    def test_init_light_theme(self) -> None:
        """Test initialization with light theme."""
        viz = Visualizer(theme="tiktokenizer")
        assert viz.theme_name == "tiktokenizer"
        assert viz.theme == LIGHT_THEMES["tiktokenizer"]
        assert viz.text_color == TEXT_COLOR_LIGHT

    def test_init_dark_theme(self) -> None:
        """Test initialization with dark theme."""
        viz = Visualizer(theme="tiktokenizer_dark")
        assert viz.theme_name == "tiktokenizer_dark"
        assert viz.theme == DARK_THEMES["tiktokenizer_dark"]
        assert viz.text_color == TEXT_COLOR_DARK

    def test_init_ocean_breeze_theme(self) -> None:
        """Test initialization with the new 'ocean_breeze' light theme."""
        viz = Visualizer(theme="ocean_breeze")
        assert viz.theme_name == "ocean_breeze"
        assert viz.theme == LIGHT_THEMES["ocean_breeze"]
        assert viz.text_color == TEXT_COLOR_LIGHT

    def test_init_midnight_theme(self) -> None:
        """Test initialization with the new 'midnight' dark theme."""
        viz = Visualizer(theme="midnight")
        assert viz.theme_name == "midnight"
        assert viz.theme == DARK_THEMES["midnight"]
        assert viz.text_color == TEXT_COLOR_DARK

    def test_init_invalid_theme(self) -> None:
        """Test initialization with invalid theme."""
        with pytest.raises(ValueError, match="Invalid theme"):
            Visualizer(theme="invalid_theme")


class TestVisualizerThemeManagement:
    """Test theme management methods."""

    def test_get_theme_light(self) -> None:
        """Test getting light theme."""
        viz = Visualizer()
        theme, text_color = viz._get_theme("pastel")
        assert theme == LIGHT_THEMES["pastel"]
        assert text_color == TEXT_COLOR_LIGHT

    def test_get_theme_dark(self) -> None:
        """Test getting dark theme."""
        viz = Visualizer()
        theme, text_color = viz._get_theme("pastel_dark")
        assert theme == DARK_THEMES["pastel_dark"]
        assert text_color == TEXT_COLOR_DARK

    def test_get_color_cycling(self) -> None:
        """Test color cycling behavior."""
        viz = Visualizer(theme="pastel")
        assert viz._get_color(0) == LIGHT_THEMES["pastel"][0]
        assert viz._get_color(1) == LIGHT_THEMES["pastel"][1]
        theme_length = len(LIGHT_THEMES["pastel"])
        assert viz._get_color(theme_length) == LIGHT_THEMES["pastel"][0]
        assert viz._get_color(theme_length + 1) == LIGHT_THEMES["pastel"][1]

    def test_darken_color_valid_hex(self) -> None:
        """Test color darkening with valid hex colors."""
        viz = Visualizer()
        darkened = viz._darken_color("#FF0000", 0.5)
        assert darkened == "#7f0000"
        darkened = viz._darken_color("#F00", 0.5)
        assert darkened == "#7f0000"

    def test_darken_color_invalid_hex(self, caplog) -> None:
        """Test color darkening with invalid hex colors."""
        viz = Visualizer()
        result = viz._darken_color("invalid", 0.5)
        assert result == "#808080"
        assert "Could not darken" in caplog.text

    def test_darken_color_edge_cases(self) -> None:
        """Test color darkening edge cases."""
        viz = Visualizer()
        darkened = viz._darken_color("#FFFFFF", 0.0)
        assert darkened == "#000000"
        darkened = viz._darken_color("#FFFFFF", 1.0)
        assert darkened == "#ffffff"


class TestVisualizerPrintMethod:
    """Test the print method functionality."""

    def test_print_empty_chunks(self, sample_text: str, mock_console_instance) -> None:
        """Test printing empty chunks."""
        Visualizer().print([])
        mock_console_instance.print.assert_called_once_with("No chunks to visualize.")

    @pytest.mark.parametrize("method", ["print", "__call__"])
    def test_print_with_full_text(
        self,
        sample_chunks: list[Chunk],
        sample_text: str,
        mock_console_instance,
        mock_text_class,
        mock_text_instance,
        method: str,
    ) -> None:
        """Test printing with provided full text."""
        viz = Visualizer()
        getattr(viz, method)(sample_chunks, sample_text)
        mock_console_instance.print.assert_called_once_with(mock_text_instance)

    @pytest.mark.parametrize("method", ["print", "__call__"])
    def test_print_without_full_text(
        self,
        sample_chunks: list[Chunk],
        mock_console_instance,
        mock_text_class,
        method: str,
    ) -> None:
        """Test printing without provided full text (reconstruction)."""
        viz = Visualizer()
        getattr(viz, method)(sample_chunks)
        expected_text = "Hello world! This is a test."
        mock_text_class.assert_called_once_with(expected_text)

    def test_print_invalid_chunks_no_text_attr(self) -> None:
        """Test printing with chunks missing text attribute."""
        with pytest.raises(ValueError, match="Chunks must have 'text'"):
            Visualizer().print([[]])

    def test_print_invalid_chunk_indices(self, sample_text: str, caplog) -> None:
        """Test printing with chunks having invalid indices."""
        invalid_chunks = [
            MagicMock(text="Hello ", start_index=None, end_index=6),
            MagicMock(text="world!", start_index=6, end_index="invalid"),
        ]
        Visualizer().print(invalid_chunks, sample_text)
        assert "invalid start/end index" in caplog.text

    def test_print_stylize_error_handling(
        self,
        sample_chunks: list[Chunk],
        sample_text: str,
        mock_text_class,
        mock_text_instance,
        caplog,
    ) -> None:
        """Test error handling during text stylization."""
        mock_text_instance.stylize.side_effect = Exception("Stylize error")
        Visualizer().print(sample_chunks, sample_text)
        assert "Could not apply style" in caplog.text


class TestVisualizerSaveMethod:
    """Test the save method functionality."""

    def test_save_empty_chunks(self, caplog) -> None:
        """Test saving with empty chunks."""
        caplog.set_level(logging.INFO)
        Visualizer().save("test.html", [])
        assert "No chunks to visualize. HTML file not saved." in caplog.text

    def test_save_with_full_text(
        self, sample_chunks: list[Chunk], sample_text: str, tmp_path, caplog
    ) -> None:
        """Test saving with provided full text."""
        caplog.set_level(logging.INFO)
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, sample_chunks, sample_text)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Hello" in content
        assert "world!" in content
        assert "This is" in content
        assert "a test." in content
        assert "Chunk Visualization" in content
        assert "background-color:" in content
        # Check that logger.info was called with the success message
        assert f"file://{os.path.abspath(filepath)}" in caplog.text

    def test_save_without_full_text(self, sample_chunks: list[Chunk], tmp_path: Path) -> None:
        """Test saving without provided full text (reconstruction)."""
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, sample_chunks)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Hello" in content
        assert "world!" in content
        assert "This is" in content
        assert "a test." in content

    def test_save_auto_add_html_extension(
        self, sample_chunks: list[Chunk], tmp_path: Path
    ) -> None:
        """Test automatic addition of .html extension."""
        filepath = os.path.join(tmp_path, "test")
        Visualizer().save(filepath, sample_chunks)
        expected_path = filepath + ".html"
        assert os.path.exists(expected_path)

    def test_save_custom_title(self, sample_chunks: list[Chunk], tmp_path: Path) -> None:
        """Test saving with custom title."""
        filepath = os.path.join(tmp_path, "test.html")
        custom_title = "My Custom Title"
        Visualizer().save(filepath, sample_chunks, title=custom_title)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert f"<title>{html.escape(custom_title)}</title>" in content

    def test_save_dark_theme_styling(self, sample_chunks: list[Chunk], tmp_path: Path) -> None:
        """Test saving with dark theme styling."""
        filepath = os.path.join(tmp_path, "test.html")
        viz = Visualizer(theme="pastel_dark")
        viz.save(filepath, sample_chunks)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert BODY_BACKGROUND_COLOR_DARK in content
        assert CONTENT_BACKGROUND_COLOR_DARK in content
        assert TEXT_COLOR_DARK in content

    def test_save_light_theme_styling(self, sample_chunks: list[Chunk], tmp_path: Path) -> None:
        """Test saving with light theme styling."""
        filepath = os.path.join(tmp_path, "test.html")
        viz = Visualizer(theme="pastel")
        viz.save(filepath, sample_chunks)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert BODY_BACKGROUND_COLOR_LIGHT in content
        assert CONTENT_BACKGROUND_COLOR_LIGHT in content
        assert TEXT_COLOR_LIGHT in content

    def test_save_overlapping_chunks(
        self, overlapping_chunks: list[Chunk], tmp_path: Path
    ) -> None:
        """Test saving with overlapping chunks."""
        filepath = os.path.join(tmp_path, "test.html")
        full_text = "Hello world! This is a test."
        Visualizer().save(filepath, overlapping_chunks, full_text)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "(Overlap)" in content
        assert "title=" in content

    def test_save_hippo_favicon_embedding(
        self, sample_chunks: list[Chunk], tmp_path: Path
    ) -> None:
        """Test that hippo favicon is properly embedded."""
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, sample_chunks)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert 'rel="icon"' in content
        assert 'type="image/svg+xml"' in content
        assert "data:image/svg+xml;base64," in content

    def test_save_favicon_encoding_error(
        self, sample_chunks: list[Chunk], tmp_path, caplog
    ) -> None:
        """Test handling of favicon encoding errors."""
        with patch("base64.b64encode", side_effect=Exception("Encoding error")):
            filepath = os.path.join(tmp_path, "test.html")
            Visualizer().save(filepath, sample_chunks)
            assert "Could not encode embedded hippo favicon: Encoding error" in caplog.text

    def test_save_invalid_chunks(self) -> None:
        """Test saving with invalid chunks."""
        with pytest.raises(AttributeError, match="Chunks must have 'text'"):
            Visualizer().save("test.html", [[]])

    def test_save_file_write_error(self, sample_chunks: list[Chunk]) -> None:
        """Test handling of file write errors."""
        with pytest.raises(IOError, match="Could not write file"):
            Visualizer().save("/nonexistent/directory/test.html", sample_chunks)

    def test_save_text_escaping(self, tmp_path: Path) -> None:
        """Test proper HTML escaping of text content."""
        html_chunks = [
            Chunk(text="<script>", start_index=0, end_index=8, token_count=1),
            Chunk(text="alert('hi')", start_index=8, end_index=19, token_count=2),
            Chunk(text="</script>", start_index=19, end_index=28, token_count=1),
        ]
        html_text = "<script>alert('hi')</script>"
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, html_chunks, html_text)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "&lt;script&gt;" in content
        assert "&lt;/script&gt;" in content
        assert "alert(&#x27;hi&#x27;)" in content

    def test_save_newline_handling(self, tmp_path: Path) -> None:
        """Test proper handling of newlines in text."""
        newline_chunks = [
            Chunk(text="Line 1\n", start_index=0, end_index=7, token_count=2),
            Chunk(text="Line 2\n", start_index=7, end_index=14, token_count=2),
            Chunk(text="Line 3", start_index=14, end_index=20, token_count=2),
        ]
        newline_text = "Line 1\nLine 2\nLine 3"
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, newline_chunks, newline_text)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Line 1<br>" in content
        assert "Line 2<br>" in content


class TestVisualizerUtilityMethods:
    """Test utility methods."""

    def test_repr(self) -> None:
        """Test __repr__ method."""
        viz = Visualizer(theme="pastel")
        repr_str = repr(viz)
        assert "Visualizer" in repr_str
        assert str(viz.theme) in repr_str


class TestVisualizerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text_chunks(self, tmp_path: Path) -> None:
        """Test chunks with empty text."""
        empty_text_chunks = [
            Chunk(text="", start_index=0, end_index=0, token_count=0),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, empty_text_chunks, "Hello")
        assert os.path.exists(filepath)

    def test_chunks_beyond_text_length(self, sample_text: str, mock_text_class) -> None:
        """Test chunks with indices beyond text length."""
        beyond_chunks = [
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
            Chunk(text="Beyond", start_index=100, end_index=106, token_count=1),
        ]
        Visualizer().print(beyond_chunks, sample_text)
        mock_text_class.assert_called_once_with(sample_text)

    def test_negative_chunk_indices(self, sample_text: str, tmp_path: Path) -> None:
        """Test chunks with negative indices."""
        negative_chunks = [
            Chunk(text="Invalid", start_index=-5, end_index=5, token_count=1),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        filepath = os.path.join(tmp_path, "test.html")
        Visualizer().save(filepath, negative_chunks, sample_text)
        assert os.path.exists(filepath)

    def test_chunk_start_equals_end(self, sample_text: str) -> None:
        """Test chunks where start_index equals end_index."""
        equal_indices_chunks = [
            Chunk(text="", start_index=5, end_index=5, token_count=0),
            Chunk(text="Hello", start_index=0, end_index=5, token_count=1),
        ]
        Visualizer().print(equal_indices_chunks, sample_text)


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
        assert "ocean_breeze" in LIGHT_THEMES
        assert "midnight" in DARK_THEMES

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
        assert "🦛" in Visualizer.HIPPO_SVG_CONTENT
        assert "<svg" in Visualizer.HIPPO_SVG_CONTENT
        assert "</svg>" in Visualizer.HIPPO_SVG_CONTENT


class TestVisualizerIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_light_theme(
        self,
        sample_chunks: list[Chunk],
        sample_text: str,
        tmp_path: Path,
    ) -> None:
        """Test complete workflow with light theme."""
        filepath = os.path.join(tmp_path, "integration_test.html")
        viz = Visualizer(theme="pastel")
        viz.save(filepath, sample_chunks, sample_text, title="Integration Test")
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Integration Test" in content
        assert "Hello" in content and "world!" in content
        assert "background-color:" in content
        assert "🦛" in content or "data:image/svg+xml" in content
        assert BODY_BACKGROUND_COLOR_LIGHT in content

    def test_full_workflow_dark_theme(
        self,
        sample_chunks: list[Chunk],
        sample_text: str,
        tmp_path: Path,
    ) -> None:
        """Test complete workflow with dark theme."""
        filepath = os.path.join(tmp_path, "integration_test_dark.html")
        viz = Visualizer(theme="pastel_dark")
        viz.save(filepath, sample_chunks, sample_text, title="Dark Theme Test")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert BODY_BACKGROUND_COLOR_DARK in content
        assert CONTENT_BACKGROUND_COLOR_DARK in content
        assert TEXT_COLOR_DARK in content

    def test_complex_overlapping_scenario(self, tmp_path: Path) -> None:
        """Test complex overlapping chunks scenario."""
        complex_chunks = [
            Chunk(text="The quick", start_index=0, end_index=9, token_count=2),
            Chunk(text="quick brown", start_index=4, end_index=15, token_count=2),
            Chunk(text="brown fox", start_index=10, end_index=19, token_count=2),
            Chunk(text="fox jumps", start_index=16, end_index=25, token_count=2),
        ]
        complex_text = "The quick brown fox jumps"
        filepath = os.path.join(tmp_path, "complex_test.html")
        Visualizer().save(filepath, complex_chunks, complex_text)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "(Overlap)" in content
        assert "The" in content
        assert "quick" in content
        assert "brown" in content
        assert "fox" in content
        assert "jumps" in content


class TestVisualizerStringInput:
    """Test the list[str] input support for Visualizer."""

    def test_print_list_of_strings(
        self,
        mock_console_instance,
        mock_text_class,
        mock_text_instance,
    ) -> None:
        """Test printing a list of strings auto-constructs chunks and full_text."""
        Visualizer().print(["hello", "world"])
        mock_text_class.assert_called_once_with("helloworld")
        mock_console_instance.print.assert_called_once_with(mock_text_instance)

    def test_call_list_of_strings(self, mock_text_class, mock_console_class) -> None:
        """Test __call__ with a list of strings delegates to print correctly."""
        Visualizer()(["hello", "world"])
        mock_text_class.assert_called_once_with("helloworld")

    def test_print_strings_with_explicit_full_text(self, mock_text_class) -> None:
        """Test that explicit full_text is not overwritten when strings are passed."""
        # full_text has a space between the chunks
        Visualizer().print(["hello", "world"], full_text="hello world")
        mock_text_class.assert_called_once_with("hello world")

    def test_print_strings_indices_with_gaps(self, mock_text_class, mock_text_instance) -> None:
        """Test that string chunks get correct indices when full_text has gaps."""
        Visualizer().print(["a", "b"], full_text="a-b")
        # "b" should be highlighted at index 2, not index 1
        assert mock_text_instance.stylize.call_count == 2
        # First call: "a" at (0, 1)
        first_call_args = mock_text_instance.stylize.call_args_list[0]
        assert first_call_args[0][1] == 0  # start
        assert first_call_args[0][2] == 1  # end
        # Second call: "b" at (2, 3)
        second_call_args = mock_text_instance.stylize.call_args_list[1]
        assert second_call_args[0][1] == 2  # start
        assert second_call_args[0][2] == 3  # end

    def test_print_mixed_chunks_and_strings(self, mock_text_class, mock_text_instance) -> None:
        """Test printing a mix of Chunk objects and strings."""
        mixed = [
            Chunk(text="Hello ", start_index=0, end_index=6, token_count=1),
            "world",
        ]
        Visualizer().print(mixed, full_text="Hello world")
        mock_text_class.assert_called_once_with("Hello world")
        # Should have 2 stylize calls for 2 chunks
        assert mock_text_instance.stylize.call_count == 2
