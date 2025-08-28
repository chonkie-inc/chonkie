"""Chonkie Chefs module.

This module provides PDF processing capabilities through various Chef implementations.
Each Chef is responsible for processing PDFs and extracting structured information
from them using different underlying technologies.
"""

from .base import (
    BaseChef,
    PDFProcessingChef,
    ProcessingResult,
    ProcessingStatus,
)
from .config import BaseChefConfig, PDFChefConfig, MarkdownChefConfig, DocChefConfig
from .exceptions import (
    ChefError,
    PDFProcessingError,
    ValidationError,
    ContentExtractionError,
    OCRProcessingError,
)
from .mistral_ocr import MistralOCRChef
from .markitdown import MarkitdownChef
from .docling import DoclingChef

# For backward compatibility
from .config import ChefConfig

__all__ = [
    "BaseChef",
    "PDFProcessingChef",
    "ProcessingResult",
    "ProcessingStatus",
    "BaseChefConfig",
    "PDFChefConfig",
    "MarkdownChefConfig",
    "DocChefConfig",
    "ChefConfig",  # Keep for backward compatibility
    "ChefError",
    "PDFProcessingError",
    "ValidationError",
    "ContentExtractionError",
    "OCRProcessingError",
    "MistralOCRChef",
    "MarkitdownChef",
    "DoclingChef",
] 