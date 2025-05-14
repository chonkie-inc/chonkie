"""PDF-specific document types for Chonkie.

This module provides specialized document types for handling PDF content,
including support for images, tables, and layout information.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import base64
from pathlib import Path

from .document import Document


class ContentType(Enum):
    """Types of content that can be extracted from a PDF."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORM = "form"
    UNKNOWN = "unknown"


@dataclass
class PDFImage:
    """Represents an image extracted from a PDF.
    
    Attributes:
        data: Base64 encoded image data
        format: Image format (e.g., 'png', 'jpeg')
        width: Image width in pixels
        height: Image height in pixels
        page_number: Page number where the image was found
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        metadata: Additional image metadata
    """
    data: str  # Base64 encoded
    format: str
    width: int
    height: int
    page_number: int
    bbox: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        """Save the image to a file.
        
        Args:
            path: Path where to save the image
        """
        path = Path(path)
        image_data = base64.b64decode(self.data)
        path.write_bytes(image_data)


@dataclass
class PDFTable:
    """Represents a table extracted from a PDF.
    
    Attributes:
        data: Table data as a list of lists
        headers: Table headers
        page_number: Page number where the table was found
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        metadata: Additional table metadata
    """
    data: List[List[str]]
    headers: Optional[List[str]] = None
    page_number: int = 1
    bbox: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFPage:
    """Represents a single page from a PDF.
    
    Attributes:
        page_number: Page number
        text: Text content of the page
        images: List of images found on the page
        tables: List of tables found on the page
        width: Page width in points
        height: Page height in points
        metadata: Additional page metadata
    """
    page_number: int
    text: str
    images: List[PDFImage] = field(default_factory=list)
    tables: List[PDFTable] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFDocument(Document):
    """Enhanced Document class for PDF content.
    
    This class extends the base Document class with PDF-specific features
    like page management, image handling, and table extraction.
    
    Attributes:
        pages: List of PDF pages
        images: List of all images in the document
        tables: List of all tables in the document
        pdf_metadata: PDF-specific metadata
    """
    pages: List[PDFPage] = field(default_factory=list)
    images: List[PDFImage] = field(default_factory=list)
    tables: List[PDFTable] = field(default_factory=list)
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_page(self, page_number: int) -> Optional[PDFPage]:
        """Get a specific page by number.
        
        Args:
            page_number: The page number to retrieve
            
        Returns:
            The requested page or None if not found
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def get_images_by_page(self, page_number: int) -> List[PDFImage]:
        """Get all images from a specific page.
        
        Args:
            page_number: The page number to get images from
            
        Returns:
            List of images found on the specified page
        """
        page = self.get_page(page_number)
        return page.images if page else []

    def get_tables_by_page(self, page_number: int) -> List[PDFTable]:
        """Get all tables from a specific page.
        
        Args:
            page_number: The page number to get tables from
            
        Returns:
            List of tables found on the specified page
        """
        page = self.get_page(page_number)
        return page.tables if page else []

    def extract_text(self) -> str:
        """Extract all text from the document.
        
        Returns:
            Concatenated text from all pages
        """
        return "\n".join(page.text for page in self.pages)

    def get_content_by_type(self, content_type: ContentType) -> List[Union[PDFImage, PDFTable]]:
        """Get all content of a specific type from the document.
        
        Args:
            content_type: The type of content to retrieve
            
        Returns:
            List of content items of the specified type
        """
        if content_type == ContentType.IMAGE:
            return self.images
        elif content_type == ContentType.TABLE:
            return self.tables
        return [] 