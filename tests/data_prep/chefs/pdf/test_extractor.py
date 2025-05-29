"""Tests for the PDF extractor chef."""

import io
import os
import pytest
from pathlib import Path
from typing import Dict, Any

from chonkie.chefs.pdf import PDFExtractorChef, PDFExtractorConfig

# Path to a real, minimal valid PDF for testing (update as needed)
VALID_PDF_PATH = "tests/data_prep/chefs/pdf/sample.pdf"

@pytest.fixture
def sample_pdf_file():
    """Provide a valid PDF file for testing, or skip if not available or invalid."""
    try:
        import PyPDF2
        pdf_path = Path(VALID_PDF_PATH)
        if not pdf_path.exists():
            pytest.skip("No valid PDF file available for testing.")
        # Try to open with PyPDF2 to ensure it's valid
        with open(pdf_path, "rb") as f:
            PyPDF2.PdfReader(f)
        return pdf_path
    except Exception:
        pytest.skip("No valid PDF file available for testing.")

@pytest.fixture
def pdf_chef():
    """Create a PDF extractor chef instance."""
    return PDFExtractorChef()

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory for image output."""
    return tmp_path / "images"

def test_pdf_chef_initialization():
    """Test PDF chef initialization."""
    chef = PDFExtractorChef()
    assert chef.config.name == "pdf_extractor"
    assert chef.config.extract_metadata is True
    assert chef.config.extract_images is False
    assert chef.config.image_format == "png"
    assert chef.config.image_quality == 85
    
    config = PDFExtractorConfig(
        name="custom_pdf_chef",
        extract_metadata=False,
        extract_images=True,
        image_format="jpeg",
        image_quality=90,
        page_range=(0, 5)
    )
    chef = PDFExtractorChef(config)
    assert chef.config.name == "custom_pdf_chef"
    assert chef.config.extract_metadata is False
    assert chef.config.extract_images is True
    assert chef.config.image_format == "jpeg"
    assert chef.config.image_quality == 90
    assert chef.config.page_range == (0, 5)

def test_pdf_chef_validation(sample_pdf_file, pdf_chef):
    """Test PDF validation."""
    assert pdf_chef.validate(sample_pdf_file) is True

def test_pdf_chef_prepare(sample_pdf_file, pdf_chef):
    """Test PDF preparation."""
    result = pdf_chef.prepare(sample_pdf_file)
    assert isinstance(result, dict)
    assert "text" in result
    assert "num_pages" in result
    assert "processed_pages" in result

def test_pdf_chef_clean(pdf_chef):
    """Test PDF cleaning."""
    data = {
        "text": "Line 1\n\nLine 2\n\n\nLine 3",
        "num_pages": 1,
        "processed_pages": [0]
    }
    cleaned = pdf_chef.clean(data)
    assert cleaned["text"] == "Line 1\nLine 2\nLine 3"
    assert cleaned["num_pages"] == 1
    assert cleaned["processed_pages"] == [0]

def test_pdf_chef_pipeline(sample_pdf_file, pdf_chef):
    """Test the full PDF processing pipeline."""
    result = pdf_chef(sample_pdf_file)
    assert isinstance(result, dict)
    assert "text" in result
    assert "num_pages" in result
    assert "processed_pages" in result

def test_pdf_chef_page_range(sample_pdf_file):
    """Test PDF processing with page range."""
    config = PDFExtractorConfig(name="test", page_range=(0, 2))
    chef = PDFExtractorChef(config)
    result = chef.prepare(sample_pdf_file)
    assert result["processed_pages"] == [0, 1]

def test_pdf_chef_metadata(sample_pdf_file):
    """Test PDF metadata extraction."""
    config = PDFExtractorConfig(name="test", extract_metadata=True)
    chef = PDFExtractorChef(config)
    result = chef.prepare(sample_pdf_file)
    assert "metadata" in result

def test_pdf_chef_image_extraction(sample_pdf_file, temp_image_dir):
    """Test PDF image extraction."""
    config = PDFExtractorConfig(
        name="test",
        extract_images=True,
        image_output_dir=str(temp_image_dir),
        image_format="png"
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(sample_pdf_file)
    
    # Check if images were extracted
    if "images" in result:
        assert isinstance(result["images"], list)
        for image in result["images"]:
            assert "path" in image
            assert "format" in image
            assert "size" in image
            assert "dimensions" in image
            assert os.path.exists(image["path"])
            assert image["format"] == "png"

def test_pdf_chef_image_extraction_jpeg(sample_pdf_file, temp_image_dir):
    """Test PDF image extraction with JPEG format."""
    config = PDFExtractorConfig(
        name="test",
        extract_images=True,
        image_output_dir=str(temp_image_dir),
        image_format="jpeg",
        image_quality=90
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(sample_pdf_file)
    
    # Check if images were extracted
    if "images" in result:
        assert isinstance(result["images"], list)
        for image in result["images"]:
            assert "path" in image
            assert "format" in image
            assert "size" in image
            assert "dimensions" in image
            assert os.path.exists(image["path"])
            assert image["format"] == "jpeg"

def test_pdf_chef_image_extraction_memory(sample_pdf_file):
    """Test PDF image extraction without saving to disk."""
    config = PDFExtractorConfig(
        name="test",
        extract_images=True,
        image_format="png"
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(sample_pdf_file)
    
    # Check if images were extracted
    if "images" in result:
        assert isinstance(result["images"], list)
        for image in result["images"]:
            assert "data" in image
            assert "format" in image
            assert "dimensions" in image
            assert isinstance(image["data"], bytes)
            assert image["format"] == "png" 