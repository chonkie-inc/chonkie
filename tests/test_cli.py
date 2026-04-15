"""Tests for the CLI utilities in chonkie."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from chonkie.cli.cli_utils import app, merge_params, parse_params

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


# --- parse_params / merge_params (pure helpers) ---


def test_parse_params_empty() -> None:
    """Empty or missing param list returns an empty dict."""
    assert parse_params(None) == {}
    assert parse_params([]) == {}


def test_parse_params_boolean_flag_and_literals() -> None:
    """Flags without '=' are True; common literals are coerced."""
    out = parse_params(
        ["verbose", "enabled=true", "disabled=false", "empty=none", "nulled=null"],
    )
    assert out == {
        "verbose": True,
        "enabled": True,
        "disabled": False,
        "empty": None,
        "nulled": None,
    }


def test_parse_params_numeric_and_scientific() -> None:
    """Integers, floats, and scientific notation are parsed as numbers."""
    assert parse_params(["n=42"]) == {"n": 42}
    assert parse_params(["x=3.14"]) == {"x": 3.14}
    assert parse_params(["sci=1e2"]) == {"sci": 100.0}
    assert parse_params(["frac=2.5e0"])["frac"] == 2.5


def test_parse_params_whitespace_around_key_value() -> None:
    out = parse_params(["  spaced_key  =  hello  "])
    assert out == {"spaced_key": "hello"}


def test_parse_params_non_numeric_string() -> None:
    """Unparseable numbers stay as strings."""
    assert parse_params(["s=not-a-number"]) == {"s": "not-a-number"}


def test_merge_params_explicit_non_none_override() -> None:
    """Explicit non-None values override parsed; None is omitted."""
    explicit = {"chunk_size": 1024, "chunk_overlap": None, "threshold": 0.5}
    parsed = {"chunk_size": 512, "tokenizer": "character"}
    merged = merge_params(explicit, parsed)
    assert merged["chunk_size"] == 1024
    assert merged["tokenizer"] == "character"
    assert "chunk_overlap" not in merged
    assert merged["threshold"] == 0.5


# --- chunk / pipeline / serve edge cases ---


def test_chunk_unknown_handshaker() -> None:
    """Unknown --handshaker prints available list and exits 1."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello world.",
            "--chunker",
            "token",
            "--chunk-size",
            "50",
            "--handshaker",
            "definitely_not_a_real_handshake_alias",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Unknown handshaker" in result.stdout


def test_chunk_file_read_error_invalid_utf8(tmp_path) -> None:
    """Invalid UTF-8 in a file yields a read error and exit 1."""
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xff\xff")

    result = runner.invoke(
        app,
        [
            "chunk",
            str(bad),
            "--chunker",
            "token",
            "--chunk-size",
            "20",
        ],
    )
    assert result.exit_code == 1
    assert "Error reading file" in result.stdout


def test_chunk_file_open_oserror(tmp_path) -> None:
    """OSError from open() is surfaced."""
    p = tmp_path / "x.txt"
    p.write_text("ok")

    with patch("builtins.open", side_effect=OSError("permission denied")):
        result = runner.invoke(
            app,
            ["chunk", str(p), "--chunker", "token", "--chunk-size", "5"],
        )

    assert result.exit_code == 1
    assert "Error reading file" in result.stdout


def test_chunk_chunker_init_invalid_kwargs() -> None:
    """Invalid chunker constructor kwargs surface as exit 1."""
    result = runner.invoke(
        app,
        [
            "chunk",
            "Hello.",
            "--chunker",
            "token",
            "--chunker-params",
            "not_a_valid_token_chunker_kwarg=1",
        ],
    )
    assert result.exit_code == 1
    assert "Error initializing chunker" in result.stdout


def test_chunk_handshake_write_failure() -> None:
    """Handshake write errors are reported and exit 1."""

    class FailingHandshake:
        def __init__(self) -> None:
            pass

        def write(self, chunks) -> None:  # noqa: ANN001
            raise RuntimeError("persist failed")

    fake_entry = MagicMock()
    fake_entry.component_class = FailingHandshake

    with patch("chonkie.cli.cli_utils.ComponentRegistry.get_handshake", return_value=fake_entry):
        result = runner.invoke(
            app,
            [
                "chunk",
                "Short text here.",
                "--chunker",
                "token",
                "--chunk-size",
                "20",
                "--chunker-params",
                "tokenizer=character",
                "--handshaker",
                "broken-store",
            ],
        )

    assert result.exit_code == 1
    assert "Error storing chunks" in result.stdout


def test_chunk_stores_chunks_with_mock_handshake() -> None:
    """Successful handshake write path prints storage confirmation."""
    mock_hs = MagicMock()

    class FakeHandshake:
        def __init__(self) -> None:
            pass

        def write(self, chunks) -> None:  # noqa: ANN001
            mock_hs.written = chunks

    fake_entry = MagicMock()
    fake_entry.component_class = FakeHandshake

    with patch("chonkie.cli.cli_utils.ComponentRegistry.get_handshake", return_value=fake_entry):
        result = runner.invoke(
            app,
            [
                "chunk",
                "One two three four.",
                "--chunker",
                "token",
                "--chunk-size",
                "10",
                "--chunker-params",
                "tokenizer=character",
                "--handshaker",
                "fake-store",
            ],
        )

    assert result.exit_code == 0
    assert "Storing chunks" in result.stdout
    assert "Chunks stored successfully" in result.stdout
    assert mock_hs.written is not None


def test_pipeline_requires_input() -> None:
    """Pipeline with no text and no --d fails."""
    result = runner.invoke(app, ["pipeline"])
    assert result.exit_code == 1
    assert "Must provide either text" in result.stdout


def test_pipeline_directory_not_found(tmp_path) -> None:
    """Pipeline --d with a missing directory fails."""
    missing = tmp_path / "no_such_subdir_chonkie_cli"
    result = runner.invoke(
        app,
        ["pipeline", "--d", str(missing), "--chunker", "sentence"],
    )
    assert result.exit_code == 1
    assert "Directory" in result.stdout
    assert "not found" in result.stdout


def test_pipeline_run_failure() -> None:
    """Exceptions from pipe.run are reported and exit 1."""
    mock_pipe = MagicMock()
    mock_pipe.run.side_effect = RuntimeError("simulated pipeline failure")

    with patch("chonkie.cli.cli_utils.Pipeline", return_value=mock_pipe):
        result = runner.invoke(
            app,
            ["pipeline", "Hello world.", "--chunker", "sentence"],
        )

    assert result.exit_code == 1
    assert "Error running pipeline" in result.stdout
    assert "simulated pipeline failure" in result.stdout


def test_pipeline_no_output_when_empty_doc() -> None:
    """When run returns a falsy document, CLI prints no output message."""
    mock_pipe = MagicMock()
    mock_pipe.run.return_value = None

    with patch("chonkie.cli.cli_utils.Pipeline", return_value=mock_pipe):
        result = runner.invoke(
            app,
            ["pipeline", "Hello.", "--chunker", "sentence"],
        )

    assert result.exit_code == 0
    assert "No output generated." in result.stdout


def test_pipeline_completes_with_handshaker_only() -> None:
    """Handshaker set skips visualization and prints completion."""
    mock_pipe = MagicMock()
    mock_doc = MagicMock()
    mock_doc.chunks = [MagicMock(text="a", token_count=1)]
    mock_doc.metadata = {}
    mock_pipe.run.return_value = mock_doc

    with patch("chonkie.cli.cli_utils.Pipeline", return_value=mock_pipe):
        result = runner.invoke(
            app,
            [
                "pipeline",
                "Hello world.",
                "--chunker",
                "sentence",
                "--handshaker",
                "json",
            ],
        )

    assert result.exit_code == 0
    assert "Pipeline completed and data stored in json" in result.stdout


def test_pipeline_multi_file_summary() -> None:
    """Multiple documents trigger the multi-file summary branch."""
    d1 = MagicMock()
    d1.chunks = [MagicMock(text="x", token_count=1)]
    d1.metadata = {"filename": "a.md"}
    d2 = MagicMock()
    d2.chunks = [MagicMock(text="y", token_count=1), MagicMock(text="z", token_count=1)]
    d2.metadata = {}

    mock_pipe = MagicMock()
    mock_pipe.run.return_value = [d1, d2]

    with patch("chonkie.cli.cli_utils.Pipeline", return_value=mock_pipe):
        result = runner.invoke(
            app,
            ["pipeline", "ignored", "--chunker", "sentence"],
        )

    assert result.exit_code == 0
    assert "Processed 2 files" in result.stdout
    assert "total chunks" in result.stdout


def test_pipeline_top_level_error() -> None:
    """Outer except in pipeline() catches unexpected setup errors."""
    with patch("chonkie.cli.cli_utils.Pipeline", side_effect=OSError("boom")):
        result = runner.invoke(app, ["pipeline", "Hello.", "--chunker", "sentence"])

    assert result.exit_code == 1
    assert "Pipeline error" in result.stdout
    assert "boom" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("uvicorn") is None, reason="uvicorn not installed")
def test_serve_configures_uvicorn(monkeypatch) -> None:
    """serve() sets LOG_LEVEL and calls uvicorn.run with expected arguments."""
    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(
            app,
            [
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "18765",
                "--log-level",
                "warning",
                "--reload",
            ],
        )

    assert result.exit_code == 0
    assert "Starting Chonkie API server" in result.stdout
    assert "127.0.0.1:18765" in result.stdout
    assert "Auto-reload enabled" in result.stdout
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 18765
    assert kwargs["reload"] is True
    assert kwargs["log_level"] == "warning"
    monkeypatch.delenv("LOG_LEVEL", raising=False)


@pytest.mark.skipif(importlib.util.find_spec("uvicorn") is None, reason="uvicorn not installed")
def test_serve_without_reload_skips_reload_message() -> None:
    """Default --reload=false does not print the auto-reload line."""
    with patch("uvicorn.run"):
        result = runner.invoke(
            app,
            ["serve", "--host", "0.0.0.0", "--port", "19999", "--log-level", "info"],
        )
    assert result.exit_code == 0
    assert "Starting Chonkie API server" in result.stdout
    assert "Auto-reload enabled" not in result.stdout
