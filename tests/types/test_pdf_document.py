"""Tests for PDF document types."""

import base64
import pytest
from pathlib import Path

from chonkie.types import (
    PDFDocument,
    PDFPage,
    PDFImage,
    PDFTable,
    ContentType,
)


def test_pdf_document_creation():
    """Test PDFDocument creation and basic functionality."""
    doc = PDFDocument(
        text="Test document",
        metadata={"test": True}
    )
    
    assert doc.text == "Test document"
    assert doc.metadata["test"]
    assert len(doc.pages) == 0
    assert len(doc.images) == 0
    assert len(doc.tables) == 0


def test_pdf_page_management():
    """Test PDF page management functionality."""
    doc = PDFDocument()
    
    # Add pages
    page1 = PDFPage(page_number=1, text="Page 1")
    page2 = PDFPage(page_number=2, text="Page 2")
    doc.pages.extend([page1, page2])
    
    # Test page retrieval
    assert doc.get_page(1) == page1
    assert doc.get_page(2) == page2
    assert doc.get_page(3) is None
    
    # Test text extraction
    assert doc.extract_text() == "Page 1\nPage 2"


def test_pdf_image_handling(tmp_path: Path):
    """Test PDF image handling functionality."""
    # Create a test image
    image_data = b"test image data"
    encoded_data = base64.b64encode(image_data).decode()
    
    image = PDFImage(
        data=encoded_data,
        format="png",
        width=100,
        height=100,
        page_number=1
    )
    
    # Test image saving
    image_path = tmp_path / "test.png"
    image.save(image_path)
    assert image_path.read_bytes() == image_data


def test_pdf_table_handling():
    """Test PDF table handling functionality."""
    table = PDFTable(
        data=[["Header 1", "Header 2"], ["Value 1", "Value 2"]],
        headers=["Header 1", "Header 2"],
        page_number=1
    )
    
    assert len(table.data) == 2
    assert len(table.headers) == 2
    assert table.page_number == 1


def test_content_type_retrieval():
    """Test content type retrieval functionality."""
    doc = PDFDocument()
    
    # Add some content
    image = PDFImage(
        data="test",
        format="png",
        width=100,
        height=100,
        page_number=1
    )
    table = PDFTable(
        data=[["test"]],
        page_number=1
    )
    
    doc.images.append(image)
    doc.tables.append(table)
    
    # Test content retrieval
    assert len(doc.get_content_by_type(ContentType.IMAGE)) == 1
    assert len(doc.get_content_by_type(ContentType.TABLE)) == 1
    assert len(doc.get_content_by_type(ContentType.TEXT)) == 0 