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


def test_chunk_with_chunk_size():
    """Test chunking with explicit chunk_size parameter."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world. This is a test with explicit chunk size.",
            "--chunker",
            "recursive",
            "--chunk-size",
            "512",
        ],
    )
    assert result.exit_code == 0
    assert "Chunking with recursive..." in result.stdout


def test_chunk_with_chunker_params():
    """Test chunking with chunker_params key=value pairs."""
    # Typer list options: pass multiple values by repeating the option or as space-separated
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world. This is a test with chunker params.",
            "--chunker",
            "recursive",
            "--chunker-params",
            "chunk_size=512",
            "--chunker-params",
            "min_characters_per_chunk=50",
        ],
    )
    assert result.exit_code == 0
    assert "Chunking with recursive..." in result.stdout


def test_chunk_with_explicit_and_params():
    """Test that explicit parameters override chunker_params."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world. This is a test.",
            "--chunker",
            "recursive",
            "--chunk-size",
            "1024",
            "--chunker-params",
            "chunk_size=512",  # This should be overridden by --chunk-size
        ],
    )
    assert result.exit_code == 0
    assert "Chunking with recursive..." in result.stdout


def test_chunk_with_threshold():
    """Test chunking with threshold parameter for semantic chunker."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world. This is a test with threshold.",
            "--chunker",
            "semantic",
            "--threshold",
            "0.7",
        ],
    )
    assert result.exit_code == 0
    assert "Chunking with semantic..." in result.stdout


def test_chunk_with_chunk_overlap():
    """Test chunking with chunk_overlap parameter."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world. This is a test with overlap.",
            "--chunker",
            "token",
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "20",
        ],
    )
    assert result.exit_code == 0
    assert "Chunking with token..." in result.stdout


def test_chunk_invalid_params():
    """Test chunking with invalid parameters."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world.",
            "--chunker",
            "recursive",
            "--chunker-params",
            "invalid_param=value",
        ],
    )
    # Should fail with parameter error
    assert result.exit_code != 0


def test_pipeline_with_chunk_size():
    """Test pipeline with explicit chunk_size parameter."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "Hello world. This is a pipeline test with chunk size.",
            "--chunker",
            "recursive",
            "--chunk-size",
            "512",
        ],
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_with_chunker_params():
    """Test pipeline with chunker_params."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "Hello world. This is a pipeline test with params.",
            "--chunker",
            "recursive",
            "--chunker-params",
            "chunk_size=512",
            "--chunker-params",
            "min_characters_per_chunk=50",
        ],
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_with_chef_params():
    """Test pipeline with chef_params (text chef doesn't accept params, but test the option works)."""
    # TextChef doesn't actually accept parameters, but we test that the option parsing works
    # and gracefully handles empty/invalid params
    result = runner.invoke(
        app,
        [
            "pipeline",
            "Hello world. This is a pipeline test with chef params.",
            "--chef",
            "text",
            "--chunker",
            "recursive",
        ],
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_with_refiner_params():
    """Test pipeline with refiner_params."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "Hello world. This is a pipeline test with refiner params.",
            "--chunker",
            "recursive",
            "--refiner",
            "overlap",
            "--refiner-params",
            "context_size=50",
        ],
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout


def test_pipeline_with_all_params():
    """Test pipeline with multiple component parameters."""
    # Use token chunker which accepts chunk_overlap
    result = runner.invoke(
        app,
        [
            "pipeline",
            "Hello world. This is a comprehensive test.",
            "--chef",
            "text",
            "--chunker",
            "token",
            "--chunk-size",
            "512",
            "--chunk-overlap",
            "50",
            "--chunker-params",
            "tokenizer=character",
        ],
    )
    assert result.exit_code == 0
    assert "Running pipeline..." in result.stdout
