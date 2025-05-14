"""Tests for MistralOCRChef."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from chonkie.chefs import MistralOCRChef, ProcessingStatus
from chonkie.chefs.exceptions import OCRProcessingError


@pytest.fixture
def test_pdf():
    """Create a test PDF file."""
    pdf_path = Path("test.pdf")
    pdf_path.write_bytes(b"%PDF-1.4\nTest PDF content")
    yield str(pdf_path)
    pdf_path.unlink()


@pytest.fixture
def mock_mistral_client():
    """Mock Mistral client."""
    with patch("mistralai.client.MistralClient") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client


def test_mistral_ocr_chef_initialization():
    """Test MistralOCRChef initialization."""
    chef = MistralOCRChef()
    assert chef.name == "MistralOCRChef"
    assert chef.version == "1.0.0"
    assert chef.supported_formats == ["pdf"]


def test_mistral_ocr_chef_missing_dependencies():
    """Test MistralOCRChef with missing dependencies."""
    import sys
    modules_backup = sys.modules.copy()
    sys.modules["mistralai"] = None
    sys.modules["mistralai.client"] = None
    sys.modules["mistralai.models.chat_completion"] = None
    try:
        with pytest.raises(ImportError) as exc_info:
            MistralOCRChef()
        assert "Mistral dependencies not found" in str(exc_info.value)
    finally:
        sys.modules = modules_backup


def test_mistral_ocr_chef_process_success(test_pdf, mock_mistral_client):
    """Test successful PDF processing."""
    # Mock the Mistral API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Extracted text from PDF"
            )
        )
    ]
    mock_mistral_client.chat.return_value = mock_response

    # Set up environment variable
    os.environ["MISTRAL_API_KEY"] = "test_key"

    chef = MistralOCRChef()
    result = chef.process(test_pdf)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert result.document.text == "Extracted text from PDF"
    assert result.metadata["ocr_engine"] == "mistral"
    assert result.metadata["pages_processed"] == 1


def test_mistral_ocr_chef_process_multi_page(test_pdf, mock_mistral_client):
    """Test multi-page PDF processing."""
    # Mock the Mistral API response with multi-page content
    multi_page_text = """Page 1: This is the content of page 1.
    With multiple lines of text.

    Page 2: This is the content of page 2.
    With some additional details.

    Page 3: This is the content of page 3.
    With conclusion remarks."""
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=multi_page_text
            )
        )
    ]
    mock_mistral_client.chat.return_value = mock_response

    # Set up environment variable
    os.environ["MISTRAL_API_KEY"] = "test_key"

    chef = MistralOCRChef()
    result = chef.process(test_pdf)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert len(result.document.pages) == 3
    assert result.document.pages[0].page_number == 1
    assert result.document.pages[1].page_number == 2
    assert result.document.pages[2].page_number == 3
    assert "content of page 1" in result.document.pages[0].text
    assert "content of page 2" in result.document.pages[1].text
    assert "content of page 3" in result.document.pages[2].text
    assert result.metadata["pages_processed"] == 3


def test_mistral_ocr_chef_process_with_images(test_pdf, mock_mistral_client):
    """Test PDF processing with images."""
    # Mock the Mistral API response with images
    response_with_images = """Page 1: This is the content of page 1.
    With multiple lines of text.

    Image: iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABGdBTUEAAK/INwWK6QAAABl0RVh0
    Format: png
    Page: 1
    Width: 100
    Height: 100

    Page 2: This is the content of page 2.
    With some additional details.

    Image: /9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAFA3PEY8MlBGQUZaVVBfeMiCeG5uePWvuZHI////
    Format: jpeg
    Page: 2
    Width: 200
    Height: 150"""
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=response_with_images
            )
        )
    ]
    mock_mistral_client.chat.return_value = mock_response

    # Set up environment variable
    os.environ["MISTRAL_API_KEY"] = "test_key"

    chef = MistralOCRChef()
    
    # Mock the _extract_pages_from_response method to return pages directly
    with patch.object(chef, '_extract_pages_from_response') as mock_extract_pages:
        mock_extract_pages.return_value = [
            {"page_number": 1, "text": "This is the content of page 1. With multiple lines of text."},
            {"page_number": 2, "text": "This is the content of page 2. With some additional details."}
        ]
        
        result = chef.process(test_pdf)

    assert result.status == ProcessingStatus.SUCCESS
    assert result.document is not None
    assert len(result.document.pages) == 2
    assert len(result.document.images) == 2
    assert result.document.images[0].page_number == 1
    assert result.document.images[1].page_number == 2
    assert result.document.images[0].format == "png"
    assert result.document.images[1].format == "jpeg"
    # The images should be added to the correct pages
    assert len(result.document.pages[0].images) >= 1
    assert result.document.pages[0].images[0].format == "png"
    assert result.metadata["images_processed"] == 2


def test_mistral_ocr_chef_extract_pages():
    """Test the page extraction method."""
    chef = MistralOCRChef()
    
    # Test with explicit page markers
    text_with_markers = """Page 1: This is page one content.

    Page 2: This is page two content.

    Page 3: This is page three content."""
    
    pages = chef._extract_pages_from_response(text_with_markers)
    assert len(pages) == 3
    assert pages[0]["page_number"] == 1
    assert "page one content" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "page two content" in pages[1]["text"]
    assert pages[2]["page_number"] == 3
    assert "page three content" in pages[2]["text"]
    
    # Test with implicit page breaks (paragraphs)
    text_without_markers = """First paragraph that might be page one.

    Second paragraph that could be page two.

    Third paragraph, possibly page three."""
    
    pages = chef._extract_pages_from_response(text_without_markers)
    assert len(pages) == 3
    assert pages[0]["page_number"] == 1
    assert "First paragraph" in pages[0]["text"]
    
    # Test with single content block
    single_content = "Just one block of content without clear page divisions."
    
    pages = chef._extract_pages_from_response(single_content)
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert single_content in pages[0]["text"]


def test_mistral_ocr_chef_extract_images():
    """Test the image extraction method."""
    chef = MistralOCRChef()
    
    # Test with valid image data
    text_with_images = """Some text with embedded images.
    
    Image: iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=
    Format: png
    Page: 1
    Width: 100
    Height: 100
    
    More text in between.
    
    Image: /9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQE=
    Format: jpeg
    Page: 2
    Width: 200
    Height: 150
    """
    
    images = chef._extract_images_from_response(text_with_images)
    assert len(images) == 2
    assert images[0]["format"] == "png"
    assert images[0]["page_number"] == 1
    assert images[0]["width"] == 100
    assert images[0]["height"] == 100
    assert images[1]["format"] == "jpeg"
    assert images[1]["page_number"] == 2
    assert images[1]["width"] == 200
    assert images[1]["height"] == 150
    
    # Test with no images
    text_without_images = "This text has no embedded images."
    
    images = chef._extract_images_from_response(text_without_images)
    assert len(images) == 0


def test_mistral_ocr_chef_process_failure(test_pdf, mock_mistral_client):
    """Test PDF processing failure."""
    # Mock the Mistral API to raise an exception
    mock_mistral_client.chat.side_effect = Exception("API Error")

    # Set up environment variable
    os.environ["MISTRAL_API_KEY"] = "test_key"

    chef = MistralOCRChef()
    result = chef.process(test_pdf)

    assert result.status == ProcessingStatus.FAILED
    assert "API Error" in result.error


def test_mistral_ocr_chef_invalid_file():
    """Test processing invalid file."""
    chef = MistralOCRChef()
    result = chef.process("nonexistent.pdf")

    assert result.status == ProcessingStatus.FAILED
    assert "Validation failed" in result.error


def test_mistral_ocr_chef_missing_api_key(test_pdf, mock_mistral_client):
    """Test processing without API key."""
    if "MISTRAL_API_KEY" in os.environ:
        del os.environ["MISTRAL_API_KEY"]

    chef = MistralOCRChef()
    result = chef.process(test_pdf)

    assert result.status == ProcessingStatus.FAILED
    assert "OCR processing failed" in result.error 