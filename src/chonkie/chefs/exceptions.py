"""Custom exceptions for Chonkie Chefs."""

class ChefError(Exception):
    """Base exception for all Chef-related errors."""
    pass


class PDFProcessingError(ChefError):
    """Exception raised when PDF processing fails."""
    pass


class ValidationError(ChefError):
    """Exception raised when file validation fails."""
    pass


class ContentExtractionError(ChefError):
    """Exception raised when content extraction fails."""
    pass


class OCRProcessingError(ChefError):
    """Exception raised when OCR processing fails."""
    pass 