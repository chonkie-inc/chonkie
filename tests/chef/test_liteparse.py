"""Tests for LiteParse chef."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import pytest

from chonkie import LiteParse
from chonkie.types import Document


@pytest.fixture
def mock_liteparse():
    """Provide a mocked liteparse module with a spec'd mock parser.

    Uses spec=True on the parser so that calling parse() with incorrect
    arguments (e.g. unexpected kwargs) raises TypeError just like the real
    liteparse.LiteParse would.
    """
    try:
        from liteparse import LiteParse as RealLiteParse
    except ImportError:
        pytest.skip("liteparse not installed")

    mock_parser = create_autospec(RealLiteParse, instance=True)
    mock_parser.parse.return_value = Mock(text="")

    mock_module = Mock()
    mock_module.LiteParse.return_value = mock_parser
    with patch.dict(sys.modules, {"liteparse": mock_module}):
        yield mock_module, mock_parser


class TestLiteParseImport:
    """Test LiteParse import and class attributes."""

    def test_import(self) -> None:
        assert LiteParse is not None

    def test_has_required_methods(self) -> None:
        assert hasattr(LiteParse, "process")
        assert hasattr(LiteParse, "process_batch")
        assert hasattr(LiteParse, "aprocess")
        assert hasattr(LiteParse, "aprocess_batch")
        assert hasattr(LiteParse, "parse")

    def test_supported_types(self) -> None:
        assert ".pdf" in LiteParse.SUPPORTED_PDF_TYPES
        assert ".png" in LiteParse.SUPPORTED_IMAGE_TYPES
        assert ".jpg" in LiteParse.SUPPORTED_IMAGE_TYPES
        assert ".docx" in LiteParse.SUPPORTED_OFFICE_TYPES
        assert ".xlsx" in LiteParse.SUPPORTED_OFFICE_TYPES
        assert ".pptx" in LiteParse.SUPPORTED_OFFICE_TYPES
        assert LiteParse.SUPPORTED_TYPES == (
            LiteParse.SUPPORTED_PDF_TYPES
            | LiteParse.SUPPORTED_OFFICE_TYPES
            | LiteParse.SUPPORTED_IMAGE_TYPES
        )


class TestLiteParseInit:
    """Test LiteParse initialization."""

    def test_missing_dependency(self) -> None:
        with patch.dict(sys.modules, {"liteparse": None}):
            with pytest.raises(ImportError, match="required module is not available"):
                LiteParse()

    def test_defaults(self, mock_liteparse) -> None:
        mock_module, _ = mock_liteparse
        chef = LiteParse()

        assert chef.ocr_enabled is None
        assert chef.ocr_language is None
        mock_module.LiteParse.assert_called_once_with(
            ocr_enabled=None,
            ocr_server_url=None,
            ocr_language=None,
            tessdata_path=None,
            max_pages=None,
            target_pages=None,
            dpi=None,
            output_format=None,
            preserve_very_small_text=None,
            password=None,
            quiet=None,
            num_workers=None,
        )

    def test_custom_options(self, mock_liteparse) -> None:
        mock_module, _ = mock_liteparse
        chef = LiteParse(
            ocr_enabled=False,
            ocr_language="fra",
            ocr_server_url="http://ocr:8080",
            dpi=300,
            max_pages=50,
            target_pages="1-5,10",
            num_workers=8,
            password="secret",
        )

        assert chef.ocr_enabled is False
        assert chef.ocr_language == "fra"
        mock_module.LiteParse.assert_called_once_with(
            ocr_enabled=False,
            ocr_language="fra",
            ocr_server_url="http://ocr:8080",
            tessdata_path=None,
            max_pages=50,
            target_pages="1-5,10",
            dpi=300,
            output_format=None,
            preserve_very_small_text=None,
            password="secret",
            quiet=None,
            num_workers=8,
        )


class TestLiteParseProcess:
    """Test LiteParse file processing."""

    def test_process_pdf(self, tmp_path: Path, mock_liteparse) -> None:
        _, mock_parser = mock_liteparse
        mock_parser.parse.return_value = Mock(text="Extracted PDF content")

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4" + b"\x00" * 100)

        chef = LiteParse()
        result = chef.process(pdf_file)

        assert isinstance(result, Document)
        assert result.content == "Extracted PDF content"
        assert result.metadata["filename"] == "test.pdf"
        mock_parser.parse.assert_called_once_with(str(pdf_file))

    def test_process_image(self, tmp_path: Path, mock_liteparse) -> None:
        _, mock_parser = mock_liteparse
        mock_parser.parse.return_value = Mock(text="OCR text from image")

        img_file = tmp_path / "scan.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        chef = LiteParse()
        result = chef.process(img_file)

        assert isinstance(result, Document)
        assert result.content == "OCR text from image"
        assert result.metadata["filename"] == "scan.png"

    def test_process_docx(self, tmp_path: Path, mock_liteparse) -> None:
        _, mock_parser = mock_liteparse
        mock_parser.parse.return_value = Mock(text="Word document content")

        docx_file = tmp_path / "report.docx"
        docx_file.write_bytes(b"PK" + b"\x00" * 100)

        chef = LiteParse()
        result = chef.process(docx_file)

        assert isinstance(result, Document)
        assert result.content == "Word document content"
        assert result.metadata["filename"] == "report.docx"

    def test_process_passes_options_to_init(self, tmp_path: Path, mock_liteparse) -> None:
        mock_module, mock_parser = mock_liteparse
        mock_parser.parse.return_value = Mock(text="content")

        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4" + b"\x00" * 100)

        chef = LiteParse(
            ocr_enabled=False,
            ocr_language="deu",
            dpi=300,
            max_pages=5,
            target_pages="1-3",
            num_workers=4,
            password="pw",
        )
        chef.process(pdf_file)

        # Options go to LiteParse constructor
        mock_module.LiteParse.assert_called_once_with(
            ocr_enabled=False,
            ocr_language="deu",
            ocr_server_url=None,
            tessdata_path=None,
            max_pages=5,
            target_pages="1-3",
            dpi=300,
            output_format=None,
            preserve_very_small_text=None,
            password="pw",
            quiet=None,
            num_workers=4,
        )
        # parse() only receives the file path
        mock_parser.parse.assert_called_once_with(str(pdf_file))

    def test_process_file_not_found(self, mock_liteparse) -> None:  # noqa: ARG002
        chef = LiteParse()
        with pytest.raises(FileNotFoundError, match="File not found"):
            chef.process("/nonexistent/path/document.pdf")

    def test_process_unsupported_type(self, tmp_path: Path, mock_liteparse) -> None:  # noqa: ARG002
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not a supported format")

        chef = LiteParse()
        with pytest.raises(ValueError, match="Unsupported file type"):
            chef.process(bad_file)


class TestLiteParseParse:
    """Test LiteParse text parsing."""

    def test_parse_wraps_text(self, mock_liteparse) -> None:  # noqa: ARG002
        chef = LiteParse()
        result = chef.parse("Some plain text")

        assert isinstance(result, Document)
        assert result.content == "Some plain text"


class TestLiteParseBatch:
    """Test LiteParse batch processing."""

    def test_process_batch(self, tmp_path: Path, mock_liteparse) -> None:
        _, mock_parser = mock_liteparse
        mock_parser.parse.side_effect = [Mock(text=f"Content from doc {i}") for i in range(3)]

        files: list[str | os.PathLike] = []
        for i in range(3):
            f = tmp_path / f"doc_{i}.pdf"
            f.write_bytes(b"%PDF-1.4" + bytes([i]) * 50)
            files.append(f)

        chef = LiteParse()
        results = chef.process_batch(files)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, Document)
            assert result.content == f"Content from doc {i}"
            assert result.metadata["filename"] == f"doc_{i}.pdf"


class TestLiteParseUtilities:
    """Test LiteParse utility methods."""

    def test_call_delegates_to_process(self, tmp_path: Path, mock_liteparse) -> None:
        _, mock_parser = mock_liteparse
        mock_parser.parse.return_value = Mock(text="Called text")

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4" + b"\x00" * 100)

        chef = LiteParse()
        result = chef(pdf_file)

        assert isinstance(result, Document)
        assert result.content == "Called text"

    def test_repr(self, mock_liteparse) -> None:  # noqa: ARG002
        chef = LiteParse(ocr_enabled=False, ocr_language="deu")
        r = repr(chef)
        assert "LiteParse" in r
        assert "ocr_enabled=False" in r
        assert "deu" in r
