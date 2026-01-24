"""Tests for the Pipeline class async methods."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chonkie.pipeline import Pipeline
from chonkie.types import Document


@pytest.mark.asyncio
class TestPipelineAsync:
    """Test async pipeline functionality."""

    async def test_pipeline_async_with_direct_text_input(self) -> None:
        """Test async pipeline with direct text input."""
        doc = await (
            Pipeline()
            .chunk_with("recursive", chunk_size=512)
            .run_async(texts="This is a test document for chunking.")
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        assert doc.content == "This is a test document for chunking."

    async def test_pipeline_async_with_multiple_texts(self) -> None:
        """Test async pipeline with multiple text inputs."""
        texts = ["First document text.", "Second document text.", "Third document text."]

        docs = await Pipeline().chunk_with("recursive", chunk_size=512).run_async(texts=texts)

        assert isinstance(docs, list)
        assert len(docs) == 3
        for doc in docs:
            assert isinstance(doc, Document)
            assert len(doc.chunks) > 0

    async def test_pipeline_async_chaining(self) -> None:
        """Test async pipeline with complex chaining."""
        doc = await (
            Pipeline()
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .refine_with("overlap", context_size=50)
            .run_async(texts="Complex chaining test text. " * 50)
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0

    async def test_pipeline_async_file_input(self) -> None:
        """Test async pipeline with file input."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content for the file fetcher.")
            temp_path = Path(f.name)

        try:
            doc = await (
                Pipeline()
                .fetch_from("file", path=str(temp_path))
                .process_with("text")
                .chunk_with("recursive", chunk_size=512)
                .run_async()
            )

            assert isinstance(doc, Document)
            assert len(doc.chunks) > 0
            assert "test content" in doc.content.lower()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def test_pipeline_async_directory_input(self) -> None:
        """Test async pipeline with directory input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "doc1.txt").write_text("Content 1")
            (temp_path / "doc2.txt").write_text("Content 2")

            docs = await (
                Pipeline()
                .fetch_from("file", dir=str(temp_path))
                .process_with("text")
                .chunk_with("recursive", chunk_size=512)
                .run_async()
            )

            assert isinstance(docs, list)
            assert len(docs) == 2
            for doc in docs:
                assert isinstance(doc, Document)

    async def test_pipeline_async_concurrency(self) -> None:
        """Test that async pipeline runs concurrently for batch inputs."""
        # It's hard to deterministically test concurrency without mocking delays,
        # but we can verify it runs correctly on a batch.
        texts = [f"Text {i}" for i in range(10)]

        docs = await Pipeline().chunk_with("recursive").run_async(texts=texts)

        assert len(docs) == 10

    async def test_pipeline_async_error_handling(self) -> None:
        """Test error handling in async pipeline."""
        with pytest.raises((ValueError, RuntimeError)):
            await Pipeline().chunk_with("recursive", invalid_param=999).run_async(texts="fail")
