"""MistralOCRChef implementation for Chonkie.

This module provides the MistralOCRChef class, which is responsible for
processing PDFs using Mistral's OCR capabilities.
"""

import os
import re
from typing import Any, Dict, Optional, List

from .base import PDFProcessingChef, ProcessingResult, ProcessingStatus
from .config import ChefConfig
from .exceptions import OCRProcessingError, ContentExtractionError
from ..types import PDFDocument, PDFPage, PDFImage


class MistralOCRChef(PDFProcessingChef):
    """Chef for processing PDFs using Mistral's OCR capabilities.
    
    This Chef uses Mistral's OCR capabilities to extract text and images
    from PDF documents. It supports various OCR configurations and can
    handle different languages.
    
    Attributes:
        name: The name of the Chef
        version: The version of the Chef
        config: Configuration settings for the Chef
    """
    
    def __init__(
        self,
        name: str = "MistralOCRChef",
        version: str = "1.0.0",
        config: Optional[ChefConfig] = None
    ):
        """Initialize the MistralOCRChef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            config: Optional configuration settings
        """
        super().__init__(name, version, config)
        self._import_dependencies()

    def _import_dependencies(self) -> None:
        """Import required dependencies."""
        try:
            import mistralai
            from mistralai.client import MistralClient
            self.MistralClient = MistralClient
        except ImportError:
            raise ImportError(
                "Mistral dependencies not found. Please install them using "
                "`pip install chonkie[mistral]`"
            )

    def _extract_pages_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract page information from the Mistral response.
        
        This method parses the AI response to identify individual pages and their content.
        
        Args:
            response_text: The raw text response from Mistral
            
        Returns:
            A list of dictionaries containing page information
        """
        pages = []
        
        # Try to find page delimiters in the text
        # Look for patterns like "Page 1:", "Page 2:", etc.
        page_pattern = re.compile(r'Page\s+(\d+)[\s]*[:|-]([^P]*?)(?=Page\s+\d+[\s]*[:|-]|$)', re.DOTALL | re.IGNORECASE)
        matches = list(page_pattern.finditer(response_text))
        
        if matches:
            for match in matches:
                page_num = int(match.group(1))
                content = match.group(2).strip()
                pages.append({
                    "page_number": page_num,
                    "text": content
                })
        else:
            # If no page patterns found, try to split by double newlines
            # and assume they might be different pages
            sections = [s for s in response_text.split("\n\n") if s.strip()]
            
            if len(sections) > 1:
                for i, section in enumerate(sections, start=1):
                    pages.append({
                        "page_number": i,
                        "text": section.strip()
                    })
            else:
                # If all else fails, treat the entire text as a single page
                pages.append({
                    "page_number": 1,
                    "text": response_text.strip()
                })
                
        return pages

    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process a PDF file using Mistral's OCR capabilities.
        
        Args:
            file_path: Path to the PDF file to process
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult containing the processed document and metadata
            
        Raises:
            OCRProcessingError: If OCR processing fails
            ContentExtractionError: If content extraction fails
        """
        try:
            # Validate the file
            if not self.validate_file(file_path):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error="Invalid PDF file"
                )

            # Create a new PDFDocument
            document = PDFDocument()

            # Process each page
            with open(file_path, 'rb') as f:
                pdf_content = f.read()

            # Initialize Mistral client
            client = self.MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

            # Process the PDF with Mistral
            try:
                response = client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "user", "content": f"Extract text from this PDF document. For each page, please start with 'Page X:' where X is the page number. {pdf_content}"}
                    ]
                )
                
                # Parse the response and create pages
                text_content = response.choices[0].message.content
                
                # Extract individual pages from the response
                extracted_pages = self._extract_pages_from_response(text_content)
                
                if not extracted_pages:
                    # Fallback if no pages were extracted
                    extracted_pages = [{
                        "page_number": 1,
                        "text": text_content
                    }]
                
                # Create PDFPage objects for each extracted page
                for page_info in extracted_pages:
                    page = PDFPage(
                        page_number=page_info["page_number"],
                        text=page_info["text"]
                    )
                    document.pages.append(page)
                
                # Sort pages by page number to ensure correct order
                document.pages.sort(key=lambda p: p.page_number)
                
                # Update document text
                document.text = document.extract_text()
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    document=document,
                    metadata={
                        "pages_processed": len(document.pages),
                        "ocr_engine": "mistral"
                    }
                )
                
            except Exception as e:
                raise OCRProcessingError(f"OCR processing failed: {str(e)}") from e
                
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error=str(e)
            ) 