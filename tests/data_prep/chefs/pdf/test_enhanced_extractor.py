"""Tests for the enhanced PDF extractor chef features."""

import io
import os
import pytest
from pathlib import Path
from typing import Dict, Any

from chonkie.chefs.pdf import PDFExtractorChef, PDFExtractorConfig

# Path to a real PDF with tables, forms, annotations for testing
# Note: This test file assumes a sample PDF with these features exists
ENHANCED_PDF_PATH = "tests/data_prep/chefs/pdf/enhanced_sample.pdf"

@pytest.fixture
def enhanced_pdf_file():
    """Provide a valid enhanced PDF file for testing, or skip if not available."""
    try:
        import PyPDF2
        pdf_path = Path(ENHANCED_PDF_PATH)
        if not pdf_path.exists():
            pytest.skip("No enhanced PDF file available for testing.")
        # Try to open with PyPDF2 to ensure it's valid
        with open(pdf_path, "rb") as f:
            PyPDF2.PdfReader(f)
        return pdf_path
    except Exception:
        pytest.skip("No enhanced PDF file available for testing.")

def test_pdf_chef_table_extraction(enhanced_pdf_file):
    """Test PDF table extraction."""
    try:
        import tabula
    except ImportError:
        pytest.skip("tabula-py not available, skipping table extraction test")
        
    config = PDFExtractorConfig(
        name="test",
        extract_tables=True,
        table_format="markdown"
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(enhanced_pdf_file)
    
    # Test may be skipped if tabula is not available or tables not found
    if "tables" in result and result["tables"]:
        assert isinstance(result["tables"], list)
        for table in result["tables"]:
            assert "table_index" in table
            assert "format" in table
            assert "num_rows" in table
            assert "num_columns" in table
            assert "data" in table
            assert table["format"] == "markdown"

def test_pdf_chef_form_fields_extraction(enhanced_pdf_file):
    """Test PDF form fields extraction."""
    config = PDFExtractorConfig(
        name="test",
        extract_form_fields=True
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(enhanced_pdf_file)
    
    # Check for form fields data structure
    assert "form_fields" in result
    assert isinstance(result["form_fields"], dict)
    assert "fields" in result["form_fields"]
    
    # If the PDF has form fields, detailed_fields should be present
    if result["form_fields"]["fields"]:
        assert "detailed_fields" in result["form_fields"]

def test_pdf_chef_annotations_extraction(enhanced_pdf_file):
    """Test PDF annotations extraction."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available, skipping annotations extraction test")
        
    config = PDFExtractorConfig(
        name="test",
        extract_annotations=True
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(enhanced_pdf_file)
    
    # Check for annotations data structure
    assert "annotations" in result
    assert isinstance(result["annotations"], list)
    
    # If annotations are found, check their structure
    for annotation in result["annotations"]:
        assert "page" in annotation
        assert "type" in annotation
        assert "rect" in annotation
        assert "content" in annotation

def test_pdf_chef_ocr(enhanced_pdf_file):
    """Test PDF OCR text extraction."""
    try:
        import pytesseract
        import cv2
    except ImportError:
        pytest.skip("pytesseract or OpenCV not available, skipping OCR test")
        
    try:
        # Try to check if Tesseract is installed
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("Tesseract not properly installed, skipping OCR test")
    
    config = PDFExtractorConfig(
        name="test",
        use_ocr=True,
        ocr_language="eng",
        ocr_only_if_needed=True
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(enhanced_pdf_file)
    
    # Check that text extraction succeeded
    assert "text" in result
    assert isinstance(result["text"], str)
    
    # Test if ocr was applied (hard to test automatically without specific sample)
    # This just verifies the process completes without errors

def test_pdf_chef_vector_graphics_extraction(enhanced_pdf_file):
    """Test PDF vector graphics extraction."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available, skipping vector graphics extraction test")
        
    config = PDFExtractorConfig(
        name="test",
        extract_vector_graphics=True
    )
    chef = PDFExtractorChef(config)
    result = chef.prepare(enhanced_pdf_file)
    
    # Check for vector graphics data structure
    assert "vector_graphics" in result
    assert isinstance(result["vector_graphics"], list)
    
    # If vector graphics are found, check their structure
    for graphics in result["vector_graphics"]:
        assert "page" in graphics
        assert "num_paths" in graphics
        if graphics["num_paths"] > 0:
            assert "paths_sample" in graphics
            assert isinstance(graphics["paths_sample"], list)
        if "svg" in graphics:
            assert isinstance(graphics["svg"], str) 