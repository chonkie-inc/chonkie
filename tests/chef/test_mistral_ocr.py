"""Tests for MistralOCR chef class."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chonkie import MistralOCR
from chonkie.types import MarkdownDocument


class TestMistralOCRImportAndConstruction:
    """Test MistralOCR import and basic construction."""

    def test_mistral_ocr_import(self) -> None:
        """Test that MistralOCR can be imported."""
        assert MistralOCR is not None

    def test_mistral_ocr_has_required_methods(self) -> None:
        """Test that MistralOCR has all required methods."""
        assert hasattr(MistralOCR, "process")
        assert hasattr(MistralOCR, "process_batch")
        assert hasattr(MistralOCR, "aprocess")
        assert hasattr(MistralOCR, "aprocess_batch")

    def test_mistral_ocr_supported_types(self) -> None:
        """Test that supported types are defined."""
        assert ".png" in MistralOCR.SUPPORTED_IMAGE_TYPES
        assert ".jpg" in MistralOCR.SUPPORTED_IMAGE_TYPES
        assert ".jpeg" in MistralOCR.SUPPORTED_IMAGE_TYPES
        assert ".pdf" in MistralOCR.SUPPORTED_PDF_TYPES
        assert MistralOCR.SUPPORTED_TYPES == (
            MistralOCR.SUPPORTED_IMAGE_TYPES | MistralOCR.SUPPORTED_PDF_TYPES
        )


class TestMistralOCRErrorHandling:
    """Test MistralOCR error handling."""

    def test_mistral_ocr_missing_api_key(self) -> None:
        """Test MistralOCR raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MistralOCR requires an API key"):
                MistralOCR()

    def test_mistral_ocr_missing_dependencies(self) -> None:
        """Test MistralOCR raises error without dependencies."""
        with patch.dict(sys.modules, {"mistralai": Mock(), "mistralai.client": None}):
            with pytest.raises(ImportError, match="One or more of the required modules"):
                MistralOCR(api_key="test")


class TestMistralOCRBasicFunctionality:
    """Test MistralOCR basic functionality with mocking."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_initialization(self) -> None:
        """Test MistralOCR can be initialized with mocked dependencies."""
        mock_client = Mock()
        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            assert ocr.model == "mistral-ocr-latest"
            mock_mistral_module.Mistral.assert_called_with(api_key="test_key")

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_custom_model(self) -> None:
        """Test MistralOCR with custom model."""
        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = Mock()

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR(model="mistral-ocr-2505")
            assert ocr.model == "mistral-ocr-2505"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_process_image(self, tmp_path: Path) -> None:
        """Test MistralOCR processes an image file."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_page = Mock()
        mock_page.markdown = "Hello World"

        mock_response = Mock()
        mock_response.pages = [mock_page]

        mock_client = Mock()
        mock_client.ocr.process.return_value = mock_response

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            result = ocr.process(img_file)

            assert isinstance(result, MarkdownDocument)
            assert result.content == "Hello World"
            assert result.metadata["filename"] == "test.png"
            mock_client.ocr.process.assert_called_once()
            call_kwargs = mock_client.ocr.process.call_args[1]
            assert call_kwargs["model"] == "mistral-ocr-latest"
            assert call_kwargs["document"]["type"] == "image_url"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_process_pdf(self, tmp_path: Path) -> None:
        """Test MistralOCR processes a PDF file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4" + b"\x00" * 100)

        mock_page1 = Mock()
        mock_page1.markdown = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.markdown = "Page 2 content"

        mock_response = Mock()
        mock_response.pages = [mock_page1, mock_page2]

        mock_client = Mock()
        mock_client.ocr.process.return_value = mock_response

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            result = ocr.process(pdf_file)

            assert isinstance(result, MarkdownDocument)
            assert result.content == "Page 1 content\n\nPage 2 content"
            call_kwargs = mock_client.ocr.process.call_args[1]
            assert call_kwargs["document"]["type"] == "document_url"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_file_not_found(self) -> None:
        """Test MistralOCR raises error for missing file."""
        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = Mock()

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            with pytest.raises(FileNotFoundError, match="File not found"):
                ocr.process("/nonexistent/path/image.png")

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_unsupported_type(self, tmp_path: Path) -> None:
        """Test MistralOCR raises error for unsupported file type."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = Mock()

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            with pytest.raises(ValueError, match="Unsupported file type"):
                ocr.process(bad_file)

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_empty_pages(self, tmp_path: Path) -> None:
        """Test MistralOCR handles empty response gracefully."""
        img_file = tmp_path / "blank.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        mock_response = Mock()
        mock_response.pages = []

        mock_client = Mock()
        mock_client.ocr.process.return_value = mock_response

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            result = ocr.process(img_file)

            assert isinstance(result, MarkdownDocument)
            assert result.content == ""


class TestMistralOCRBatchProcessing:
    """Test MistralOCR batch processing."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_process_batch(self, tmp_path: Path) -> None:
        """Test MistralOCR batch processing."""
        files = []
        for i in range(3):
            f = tmp_path / f"test_{i}.png"
            f.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 50)
            files.append(f)

        responses = []
        for i in range(3):
            page = Mock()
            page.markdown = f"Text from image {i}"
            resp = Mock()
            resp.pages = [page]
            responses.append(resp)

        mock_client = Mock()
        mock_client.ocr.process.side_effect = responses

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            results = ocr.process_batch(files)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, MarkdownDocument)
                assert result.content == f"Text from image {i}"


class TestMistralOCRUtilities:
    """Test MistralOCR utility methods."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_call(self, tmp_path: Path) -> None:
        """Test MistralOCR __call__ delegates to process."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_page = Mock()
        mock_page.markdown = "Called text"

        mock_response = Mock()
        mock_response.pages = [mock_page]

        mock_client = Mock()
        mock_client.ocr.process.return_value = mock_response

        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = mock_client

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR()
            result = ocr(img_file)

            assert isinstance(result, MarkdownDocument)
            assert result.content == "Called text"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"})
    def test_mistral_ocr_repr(self) -> None:
        """Test MistralOCR string representation."""
        mock_mistral_module = Mock()
        mock_mistral_module.Mistral.return_value = Mock()

        with patch.dict(
            sys.modules,
            {"mistralai": mock_mistral_module, "mistralai.client": mock_mistral_module},
        ):
            ocr = MistralOCR(model="mistral-ocr-2505")
            assert "MistralOCR" in repr(ocr)
            assert "mistral-ocr-2505" in repr(ocr)
