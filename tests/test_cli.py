"""Tests for the CLI utilities in chonkie."""

from typer.testing import CliRunner

from chonkie.cli.cli_utils import app

runner = CliRunner()


def test_chunk_text_semantic():
    """Test chunking text using the semantic chunker."""
    result = runner.invoke(app, ["chunk", "Hello world. This is a test.", "--chunker", "semantic"])
    assert result.exit_code == 0
    assert "Chunking with semantic..." in result.stdout


def test_chunk_text_recursive():
    """Test chunking text using the recursive chunker."""
    result = runner.invoke(
        app, ["chunk", "Hello world. This is a test.", "--chunker", "recursive"]
    )
    assert result.exit_code == 0
    assert "Chunking with recursive..." in result.stdout


def test_chunk_invalid_chunker():
    """Test handling of an invalid chunker argument."""
    result = runner.invoke(app, ["chunk", "text", "--chunker", "invalid"])
    assert result.exit_code == 1
    assert "Error: Unknown chunker 'invalid'" in result.stdout


def test_chunk_file(tmp_path):
    """Test chunking a file using the sentence chunker."""
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("Hello world. This is a file test.")

    result = runner.invoke(app, ["chunk", str(p), "--chunker", "sentence"])
    assert result.exit_code == 0
    assert "Chunking with sentence..." in result.stdout


def test_chunk_file_not_found():
    """Test chunking when the specified file does not exist."""
    result = runner.invoke(app, ["chunk", "nonexistent.txt"])
    # It treats it as text if file not found, but since we check os.path.isfile inside,
    # if it's not a file, it treats it as raw text.
    # So "nonexistent.txt" is treated as the text content "nonexistent.txt".
    # This is intended behavior for the CLI to support both.
    assert result.exit_code == 0

    assert "nonexistent.txt" in result.stdout


def test_pipeline_text() -> None:
    """Test running the pipeline command with text input."""
    result = runner.invoke(
        app, ["pipeline", "Hello world. This is a pipeline test.", "--chunker", "sentence"]
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_file(tmp_path) -> None:
    """Test running the pipeline command with a file input."""
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "pipeline.txt"
    p.write_text("Pipeline file test.")

    result = runner.invoke(app, ["pipeline", str(p), "--chunker", "token"])
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_invalid_args() -> None:
    """Test running the pipeline command with missing required arguments."""
    # Pipeline requires at least text
    result = runner.invoke(app, ["pipeline"])
    assert result.exit_code != 0
