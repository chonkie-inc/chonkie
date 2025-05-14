"""Base classes and interfaces for Chonkie Chefs.

This module provides the foundational classes and interfaces for implementing
various document processing Chefs in Chonkie. Chefs are responsible for processing
documents (PDFs, markdown, restructured text, etc.) and extracting structured 
information from them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..types.document import Document
from .config import BaseChefConfig, PDFChefConfig
from .exceptions import (
    ChefError,
    PDFProcessingError,
    ValidationError,
    ContentExtractionError,
    OCRProcessingError,
)


class ProcessingStatus(Enum):
    """Enum representing the status of PDF processing."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of PDF processing by a Chef.
    
    Attributes:
        status: The status of the processing operation
        document: The processed document if successful
        error: Error message if processing failed
        metadata: Additional metadata about the processing
    """
    status: ProcessingStatus
    document: Optional[Document] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseChef(ABC):
    """Base class for all PDF processing Chefs.
    
    This abstract base class defines the interface that all PDF processing
    Chefs must implement. It provides common functionality and enforces
    consistent behavior across different Chef implementations.
    
    Attributes:
        name: The name of the Chef
        version: The version of the Chef
        supported_formats: List of supported file formats
        config: Configuration settings for the Chef
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        supported_formats: List[str],
        config: Optional[BaseChefConfig] = None
    ):
        """Initialize the Chef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            supported_formats: List of supported file formats
            config: Optional configuration settings
        """
        self.name = name
        self.version = version
        self.supported_formats = supported_formats
        self.config = config or BaseChefConfig()

    @abstractmethod
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process a PDF file.
        
        Args:
            file_path: Path to the PDF file to process
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult containing the processed document and metadata
            
        Raises:
            PDFProcessingError: If processing fails
            ValidationError: If file validation fails
            ContentExtractionError: If content extraction fails
            OCRProcessingError: If OCR processing fails
        """
        pass

    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file can be processed by this Chef.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file can be processed, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Chef.
        
        Returns:
            Dictionary containing Chef metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_formats": self.supported_formats,
            "config": self.config.to_dict()
        }


class PDFProcessingChef(BaseChef):
    """Base class specifically for PDF processing Chefs.
    
    This class extends BaseChef with PDF-specific functionality and validation.
    All PDF processing Chefs should inherit from this class.
    """
    
    def __init__(self, name: str, version: str, config: Optional[PDFChefConfig] = None):
        """Initialize the PDF processing Chef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            config: Optional configuration settings
        """
        config = config or PDFChefConfig()
        super().__init__(name, version, ["pdf"], config)

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a valid PDF, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        import os
        try:
            if not os.path.exists(file_path):
                raise ValidationError(f"File not found: {file_path}")
            if not file_path.lower().endswith('.pdf'):
                raise ValidationError(f"Not a PDF file: {file_path}")
            # Basic PDF validation could be added here
            return True
        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e 