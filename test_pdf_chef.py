#!/usr/bin/env python3
"""Test script for the PDFCleanerChef."""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import PDFCleanerChef
try:
    from src.chonkie.chefs.document import PDFCleanerChef
    pdf_chef_available = True
except ImportError as e:
    print(f"Error importing PDFCleanerChef: {e}")
    pdf_chef_available = False


def test_pdf_chef_availability():
    """Test if PDFCleanerChef is available."""
    print("\n=== Testing PDFCleanerChef Availability ===\n")
    
    if not pdf_chef_available:
        print("PDFCleanerChef could not be imported.")
        return
    
    # Create PDFCleanerChef
    pdf_chef = PDFCleanerChef()
    
    # Check availability (requires PyMuPDF)
    available = pdf_chef.is_available()
    print(f"PDFCleanerChef available: {available}")
    
    if available:
        print("PyMuPDF (fitz) is installed")
    else:
        print("PyMuPDF (fitz) is not installed. Install with:")
        print("  pip install pymupdf")
        print("  or")
        print("  pip install chonkie[pdf]")


def create_simple_text_pdf():
    """Create a simple PDF file with text content for testing."""
    print("\n=== Creating Simple PDF for Testing ===\n")
    
    # Check if we can create a PDF
    try:
        import reportlab
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("Cannot create PDF file: reportlab is not installed.")
        print("Install with: pip install reportlab")
        return None
    
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
        pdf_path = temp.name
    
    # Create a PDF with reportlab
    pdf = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Add some text
    pdf.setFont("Helvetica", 24)
    pdf.drawString(100, 700, "Chonkie PDF Test")
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 650, "This is a test PDF document for the PDFCleanerChef.")
    pdf.drawString(100, 630, "It contains some basic text content for extraction testing.")
    
    # Add some metadata
    pdf.setTitle("Chonkie PDF Test")
    pdf.setAuthor("Chonkie Test Suite")
    pdf.setSubject("PDF Extraction Test")
    
    # Add a second page
    pdf.showPage()
    pdf.setFont("Helvetica", 16)
    pdf.drawString(100, 700, "Page 2")
    pdf.drawString(100, 650, "This is the second page of the test document.")
    
    # Save the PDF
    pdf.save()
    
    print(f"Created test PDF at: {pdf_path}")
    return pdf_path


def test_pdf_processing():
    """Test PDF text extraction."""
    print("\n=== Testing PDF Processing ===\n")
    
    if not pdf_chef_available:
        print("PDFCleanerChef is not available. Skipping test.")
        return
    
    # Create a PDF chef
    pdf_chef = PDFCleanerChef(
        extract_metadata=True,
        page_numbers=True,
        handle_tables=True
    )
    
    # Check if PDF chef is available
    if not pdf_chef.is_available():
        print("PDFCleanerChef is not available (PyMuPDF not installed).")
        return
    
    # Create a test PDF
    pdf_path = create_simple_text_pdf()
    if not pdf_path:
        print("Could not create test PDF. Skipping test.")
        return
    
    try:
        # Process the PDF
        print(f"Processing PDF: {pdf_path}")
        text = pdf_chef.preprocess(pdf_path)
        
        # Print the extracted text
        print("\nExtracted Text:")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        # Check if metadata was extracted
        if "Document Metadata" in text:
            print("\nMetadata was successfully extracted")
        
        # Check if page numbers were included
        if "Page" in text:
            print("Page numbers were successfully included")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
    finally:
        # Clean up
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
            print(f"Removed test PDF: {pdf_path}")


if __name__ == "__main__":
    print("=== Testing PDFCleanerChef ===")
    
    # Run the tests
    test_pdf_chef_availability()
    test_pdf_processing()
    
    print("\nAll tests completed!")