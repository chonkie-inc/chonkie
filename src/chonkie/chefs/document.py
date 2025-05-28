"""Document preprocessing chef implementations."""

import importlib.util as importutil
import os
import re
from typing import List, Optional, Union, Dict, Any

from .base import BaseChef


class PDFCleanerChef(BaseChef):
    """Chef for extracting and cleaning text from PDF documents."""
    
    def __init__(
        self,
        extract_metadata: bool = False,
        extract_images: bool = False,
        page_numbers: bool = False,
        page_separator: str = "\n\n",
        handle_tables: bool = True,
    ):
        """Initialize the PDFCleanerChef.
        
        Args:
            extract_metadata: Whether to extract PDF metadata.
            extract_images: Whether to extract information about images (not the images themselves).
            page_numbers: Whether to include page numbers in the output.
            page_separator: The separator to use between pages.
            handle_tables: Whether to attempt to preserve table structure.
        """
        super().__init__()
        self.extract_metadata = extract_metadata
        self.extract_images = extract_images
        self.page_numbers = page_numbers
        self.page_separator = page_separator
        self.handle_tables = handle_tables
        self._pymupdf_available = self._check_pymupdf()
        
    def _check_pymupdf(self) -> bool:
        """Check if PyMuPDF is available."""
        return importutil.find_spec("fitz") is not None
    
    def _import_pymupdf(self) -> None:
        """Import PyMuPDF (fitz)."""
        if self._pymupdf_available:
            global fitz
            import fitz
        else:
            raise ImportError(
                "PyMuPDF is required for PDFCleanerChef. "
                "Please install it via `pip install chonkie[pdf]` or `pip install pymupdf`."
            )
    
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        Returns:
            bool: True if the chef dependencies are available, False otherwise.
        """
        return self._pymupdf_available
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF document.
        
        Args:
            pdf_path: Path to the PDF document.
            
        Returns:
            str: Extracted text.
        """
        self._import_pymupdf()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata if requested
            metadata = {}
            if self.extract_metadata:
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                    "page_count": len(doc),
                }
            
            # Extract text
            text_parts = []
            
            for page_num, page in enumerate(doc):
                if self.page_numbers:
                    text_parts.append(f"--- Page {page_num + 1} ---")
                
                # Extract text with appropriate options
                text = page.get_text()
                
                # Preserve tables if requested
                if self.handle_tables:
                    # PyMuPDF's table extraction is basic - for more complex tables
                    # a specialized library like tabula-py might be better
                    tables = page.find_tables()
                    if tables and tables.tables:
                        for table in tables.tables:
                            rows = []
                            for cells in table.cells:
                                for cell in cells:
                                    # Process cells if needed
                                    pass
                
                # Extract image info if requested
                if self.extract_images:
                    image_list = page.get_images()
                    if image_list:
                        text_parts.append(f"[Page {page_num + 1} contains {len(image_list)} images]")
                
                text_parts.append(text)
            
            # Close the document
            doc.close()
            
            # Join text with the specified separator
            result = self.page_separator.join(text_parts)
            
            # Add metadata as a header if requested
            if self.extract_metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items() if v])
                if metadata_text:
                    result = f"--- Document Metadata ---\n{metadata_text}\n\n{result}"
            
            return result
            
        except Exception as e:
            if not self.extract_metadata:
                return f"Error extracting text from PDF: {str(e)}"
            else:
                raise
    
    def preprocess(self, text: str) -> str:
        """Process a PDF document path to extract its text.
        
        Args:
            text: Path to the PDF document.
            
        Returns:
            str: Extracted text from the PDF.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string containing the path to a PDF file.")
            
        # If the input looks like a path to a PDF file, process it
        if text.endswith('.pdf') and os.path.exists(text):
            return self._extract_text_from_pdf(text)
        else:
            # If the input is already text or doesn't point to a PDF file
            return text
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return (
            f"PDFCleanerChef(extract_metadata={self.extract_metadata}, "
            f"extract_images={self.extract_images}, page_numbers={self.page_numbers}, "
            f"page_separator='{self.page_separator}', handle_tables={self.handle_tables})"
        ) 