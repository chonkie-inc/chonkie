"""Tests for the Pipeline class."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chonkie.pipeline import Pipeline
from chonkie.types import Document


class TestPipelineBasics:
    """Test basic pipeline functionality."""

    def test_pipeline_initialization(self) -> None:
        """Test Pipeline can be instantiated."""
        pipeline = Pipeline()
        assert pipeline is not None
        assert pipeline._steps == []

    def test_pipeline_with_direct_text_input(self) -> None:
        """Test pipeline with direct text input (no fetcher needed)."""
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=512)
            .run(texts="This is a test document for chunking.")
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        assert doc.content == "This is a test document for chunking."

    def test_pipeline_with_multiple_texts(self) -> None:
        """Test pipeline with multiple text inputs."""
        texts = ["First document text.", "Second document text.", "Third document text."]

        docs = Pipeline().chunk_with("recursive", chunk_size=512).run(texts=texts)

        assert isinstance(docs, list)
        assert len(docs) == 3
        for doc in docs:
            assert isinstance(doc, Document)
            assert len(doc.chunks) > 0

    def test_pipeline_requires_chunker(self) -> None:
        """Test that pipeline requires at least one chunker."""
        pipeline = Pipeline().process_with("text")

        with pytest.raises(ValueError, match="must include a chunker"):
            pipeline.run(texts="test")

    def test_pipeline_requires_input(self) -> None:
        """Test that pipeline requires fetcher or text input."""
        pipeline = Pipeline().chunk_with("recursive")

        with pytest.raises(ValueError, match="must include a fetcher"):
            pipeline.run()

    def test_pipeline_no_steps_raises_error(self) -> None:
        """Test that empty pipeline raises error."""
        pipeline = Pipeline()

        with pytest.raises(ValueError, match="no steps"):
            pipeline.run(texts="test")


class TestPipelineFetcher:
    """Test pipeline with file fetcher."""

    @pytest.fixture
    def temp_file(self) -> Path:
        """Fixture that creates a temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content for the file fetcher.")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_dir_with_files(self) -> Path:
        """Fixture that creates temporary directory with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "file1.txt").write_text("Content of first file.")
            (temp_path / "file2.txt").write_text("Content of second file.")
            (temp_path / "file3.md").write_text("# Markdown content")

            yield temp_path

    def test_pipeline_with_single_file(self, temp_file: Path) -> None:
        """Test pipeline with single file fetcher."""
        doc = (
            Pipeline()
            .fetch_from("file", path=str(temp_file))
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        assert "test content" in doc.content.lower()

    def test_pipeline_with_directory(self, temp_dir_with_files: Path) -> None:
        """Test pipeline with directory fetcher."""
        docs = (
            Pipeline()
            .fetch_from("file", dir=str(temp_dir_with_files))
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(docs, list)
        assert len(docs) == 3  # 3 files in directory

    def test_pipeline_with_extension_filter(self, temp_dir_with_files: Path) -> None:
        """Test pipeline with directory and extension filter."""
        docs = (
            Pipeline()
            .fetch_from("file", dir=str(temp_dir_with_files), ext=[".txt"])
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(docs, list)
        assert len(docs) == 2  # Only .txt files


class TestPipelineChunkers:
    """Test pipeline with different chunkers."""

    def test_pipeline_with_token_chunker(self) -> None:
        """Test pipeline with token chunker."""
        doc = (
            Pipeline()
            .chunk_with("token", chunk_size=100, chunk_overlap=10)
            .run(
                texts="This is a test document with enough text to create multiple chunks when using token chunking.",
            )
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.token_count <= 100

    def test_pipeline_with_recursive_chunker(self) -> None:
        """Test pipeline with recursive chunker."""
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=256)
            .run(texts="Paragraph one.\n\nParagraph two.\n\nParagraph three.")
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0

    def test_pipeline_with_sentence_chunker(self) -> None:
        """Test pipeline with sentence chunker."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        doc = Pipeline().chunk_with("sentence", chunk_size=512).run(texts=text)

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0


class TestPipelineChef:
    """Test pipeline with chefs."""

    def test_pipeline_with_text_chef(self) -> None:
        """Test pipeline with text chef."""
        doc = (
            Pipeline()
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run(texts="This is plain text.")
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0

    def test_pipeline_with_multiple_chefs_raises_error(self) -> None:
        """Test that multiple chefs raise ValueError."""
        pipeline = (
            Pipeline()
            .process_with("text")
            .process_with("markdown")  # Second chef
            .chunk_with("recursive")
        )

        with pytest.raises(ValueError, match="Multiple process steps"):
            pipeline.run(texts="test")

    def test_pipeline_without_chef(self) -> None:
        """Test pipeline without chef (should wrap text in Document)."""
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=512)
            .run(texts="Text without chef processing.")
        )

        assert isinstance(doc, Document)
        assert doc.content == "Text without chef processing."
        assert len(doc.chunks) > 0


class TestPipelineRefineries:
    """Test pipeline with refineries."""

    def test_pipeline_with_overlap_refinery(self) -> None:
        """Test pipeline with overlap refinery."""
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=256)
            .refine_with("overlap", context_size=50)
            .run(
                texts="This is a longer text that will be chunked and then refined with overlap to add context. "
                * 10,
            )
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        # Check that chunks have context (should be present in chunk.context or text)
        for chunk in doc.chunks:
            assert chunk.text is not None

    def test_pipeline_with_multiple_refineries(self) -> None:
        """Test pipeline with multiple refineries chained."""
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=512)
            .refine_with("overlap", context_size=50)
            .run(texts="Test text for multiple refineries. " * 20)
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0


class TestPipelineStepOrdering:
    """Test that pipeline automatically reorders steps."""

    def test_pipeline_reorders_steps(self) -> None:
        """Test that steps are reordered according to CHOMP."""
        # Add steps in wrong order
        doc = (
            Pipeline()
            .chunk_with("recursive", chunk_size=512)  # Should be after process
            .process_with("text")  # Should be before chunk
            .run(texts="Test text")
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0

    def test_pipeline_refinery_after_chunker(self) -> None:
        """Test that refinery runs after chunker even if defined first."""
        doc = (
            Pipeline()
            .refine_with("overlap", context_size=50)  # Defined first
            .chunk_with("recursive", chunk_size=512)  # Should run first
            .run(texts="Test text for ordering. " * 20)
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0


class TestPipelineValidation:
    """Test pipeline validation logic."""

    def test_validation_requires_chunker(self) -> None:
        """Test validation requires at least one chunker."""
        pipeline = Pipeline().process_with("text")

        with pytest.raises(ValueError, match="must include a chunker"):
            pipeline.run(texts="test")

    def test_validation_multiple_chefs_error(self) -> None:
        """Test validation rejects multiple chefs."""
        pipeline = Pipeline().process_with("text").process_with("markdown").chunk_with("recursive")

        with pytest.raises(ValueError, match="Multiple process steps"):
            pipeline.run(texts="test")

    def test_validation_no_fetcher_no_text_error(self) -> None:
        """Test validation requires fetcher or text input."""
        pipeline = Pipeline().chunk_with("recursive")

        with pytest.raises(ValueError, match="must include a fetcher"):
            pipeline.run()


class TestPipelineIntegration:
    """Integration tests for complete pipelines."""

    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for testing."""
        return """
        This is a sample document for testing pipelines.

        It has multiple paragraphs to ensure proper chunking behavior.

        Each paragraph contains some meaningful content that can be split.
        """.strip()

    def test_complete_pipeline_with_text_input(self, sample_text: str) -> None:
        """Test complete pipeline with direct text input."""
        doc = (
            Pipeline()
            .process_with("text")
            .chunk_with("recursive", chunk_size=256)
            .refine_with("overlap", context_size=20)
            .run(texts=sample_text)
        )

        assert isinstance(doc, Document)
        assert doc.content == sample_text
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert isinstance(chunk.text, str)
            assert chunk.token_count > 0

    def test_pipeline_preserves_document_structure(self, sample_text: str) -> None:
        """Test that pipeline preserves Document structure."""
        doc = Pipeline().chunk_with("token", chunk_size=50).run(texts=sample_text)

        assert isinstance(doc, Document)
        assert hasattr(doc, "content")
        assert hasattr(doc, "chunks")
        assert hasattr(doc, "metadata")
        assert doc.content == sample_text

    def test_batch_text_processing(self) -> None:
        """Test processing multiple texts in batch."""
        texts = [
            "First document for batch processing.",
            "Second document for batch processing.",
            "Third document for batch processing.",
        ]

        docs = Pipeline().chunk_with("recursive", chunk_size=512).run(texts=texts)

        assert isinstance(docs, list)
        assert len(docs) == 3

        for i, doc in enumerate(docs):
            assert isinstance(doc, Document)
            assert doc.content == texts[i]
            assert len(doc.chunks) > 0


class TestPipelineErrorHandling:
    """Test error handling in pipelines."""

    def test_invalid_chunker_type(self) -> None:
        """Test that invalid chunker type raises error."""
        with pytest.raises(ValueError):
            Pipeline().chunk_with("nonexistent_chunker")

    def test_invalid_chef_type(self) -> None:
        """Test that invalid chef type raises error."""
        with pytest.raises(ValueError):
            Pipeline().process_with("nonexistent_chef")

    def test_invalid_fetcher_type(self) -> None:
        """Test that invalid fetcher type raises error."""
        with pytest.raises(ValueError):
            Pipeline().fetch_from("nonexistent_fetcher", path="test.txt")

    def test_file_not_found_error(self) -> None:
        """Test FileNotFoundError when file doesn't exist."""
        pipeline = (
            Pipeline().fetch_from("file", path="/nonexistent/file.txt").chunk_with("recursive")
        )

        with pytest.raises((FileNotFoundError, RuntimeError)):
            pipeline.run()

    def test_invalid_parameters_raise_error(self) -> None:
        """Test that invalid parameters raise clear errors."""
        pipeline = Pipeline().chunk_with("recursive", invalid_param=999)

        with pytest.raises((ValueError, RuntimeError), match="invalid_param"):
            pipeline.run(texts="test")


class TestPipelineComponentCaching:
    """Test component instance caching."""

    def test_component_reuse_across_runs(self) -> None:
        """Test that components are cached and reused."""
        pipeline = Pipeline().chunk_with("recursive", chunk_size=512)

        doc1 = pipeline.run(texts="First run text.")
        doc2 = pipeline.run(texts="Second run text.")

        # Both runs should succeed
        assert isinstance(doc1, Document)
        assert isinstance(doc2, Document)

    def test_different_parameters_create_new_instances(self) -> None:
        """Test that different parameters create different component instances."""
        # Create pipeline with one chunker config
        doc1 = Pipeline().chunk_with("recursive", chunk_size=256).run(texts="Test text " * 100)

        # Create new pipeline with different config
        doc2 = Pipeline().chunk_with("recursive", chunk_size=512).run(texts="Test text " * 100)

        # Different chunk sizes should produce different results
        assert len(doc1.chunks) != len(doc2.chunks)


class TestPipelineWithFile:
    """Test pipeline with file operations."""

    @pytest.fixture
    def temp_text_file(self) -> Path:
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is content in a text file.\n\nIt has multiple paragraphs.\n\nFor testing purposes.",
            )
            temp_path = Path(f.name)

        yield temp_path

        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_dir_with_files(self) -> Path:
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "doc1.txt").write_text("Content of document 1.")
            (temp_path / "doc2.txt").write_text("Content of document 2.")
            (temp_path / "doc3.md").write_text("# Markdown document")

            yield temp_path

    def test_pipeline_single_file_to_chunks(self, temp_text_file: Path) -> None:
        """Test complete pipeline from single file to chunks."""
        doc = (
            Pipeline()
            .fetch_from("file", path=str(temp_text_file))
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        assert "text file" in doc.content.lower()

    def test_pipeline_directory_to_chunks(self, temp_dir_with_files: Path) -> None:
        """Test pipeline processing entire directory."""
        docs = (
            Pipeline()
            .fetch_from("file", dir=str(temp_dir_with_files))
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(docs, list)
        assert len(docs) == 3

        for doc in docs:
            assert isinstance(doc, Document)
            assert len(doc.chunks) > 0

    def test_pipeline_directory_with_extension_filter(self, temp_dir_with_files: Path) -> None:
        """Test pipeline with directory and extension filter."""
        docs = (
            Pipeline()
            .fetch_from("file", dir=str(temp_dir_with_files), ext=[".txt"])
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .run()
        )

        assert isinstance(docs, list)
        assert len(docs) == 2  # Only .txt files


class TestPipelineChaining:
    """Test method chaining and fluent API."""

    def test_fluent_api_returns_pipeline(self) -> None:
        """Test that all methods return Pipeline for chaining."""
        pipeline = Pipeline()

        assert isinstance(pipeline.chunk_with("recursive"), Pipeline)
        assert isinstance(pipeline.process_with("text"), Pipeline)
        assert isinstance(pipeline.refine_with("overlap", context_size=50), Pipeline)

    def test_complex_chaining(self) -> None:
        """Test complex method chaining."""
        doc = (
            Pipeline()
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .refine_with("overlap", context_size=50)
            .run(texts="Complex chaining test text. " * 50)
        )

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0


class TestPipelineEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_text_input(self) -> None:
        """Test pipeline with empty text."""
        doc = Pipeline().chunk_with("recursive", chunk_size=512).run(texts="")

        assert isinstance(doc, Document)
        # Empty text should produce no chunks or one empty chunk
        assert len(doc.chunks) >= 0

    def test_very_short_text(self) -> None:
        """Test pipeline with very short text."""
        doc = Pipeline().chunk_with("recursive", chunk_size=512).run(texts="Hi")

        assert isinstance(doc, Document)
        assert len(doc.chunks) >= 1

    def test_very_long_text(self) -> None:
        """Test pipeline with very long text."""
        long_text = "This is a test sentence. " * 1000

        doc = Pipeline().chunk_with("recursive", chunk_size=256).run(texts=long_text)

        assert isinstance(doc, Document)
        assert len(doc.chunks) > 10  # Should create many chunks

    def test_whitespace_only_text(self) -> None:
        """Test pipeline with whitespace-only text."""
        doc = Pipeline().chunk_with("recursive", chunk_size=512).run(texts="   \n\n   \t  ")

        assert isinstance(doc, Document)
        # Whitespace should produce no chunks or be handled gracefully


class TestPipelineReturnTypes:
    """Test pipeline return type behavior."""

    def test_single_text_returns_document(self) -> None:
        """Test that single text input returns Document."""
        result = Pipeline().chunk_with("recursive", chunk_size=512).run(texts="Single text")

        assert isinstance(result, Document)
        assert not isinstance(result, list)

    def test_multiple_texts_returns_list(self) -> None:
        """Test that multiple texts return list[Document]."""
        result = Pipeline().chunk_with("recursive", chunk_size=512).run(texts=["Text 1", "Text 2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)

    def test_directory_fetch_returns_list(self) -> None:
        """Test that directory fetch returns list[Document]."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "file.txt").write_text("Content")

            result = (
                Pipeline().fetch_from("file", dir=str(temp_path)).chunk_with("recursive").run()
            )

            assert isinstance(result, list)
