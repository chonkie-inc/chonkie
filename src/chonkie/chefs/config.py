"""Configuration management for Chonkie Chefs."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ChefConfig:
    """Configuration settings for PDF processing Chefs.
    
    Attributes:
        ocr_enabled: Whether to enable OCR processing
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        extract_forms: Whether to extract form fields
        preserve_layout: Whether to preserve layout information
        language: Language for OCR processing
        dpi: DPI for image extraction
        max_pages: Maximum number of pages to process
        timeout: Processing timeout in seconds
        additional_settings: Additional Chef-specific settings
    """
    ocr_enabled: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    extract_forms: bool = False
    preserve_layout: bool = True
    language: str = "eng"
    dpi: int = 300
    max_pages: Optional[int] = None
    timeout: int = 300
    additional_settings: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update configuration settings.
        
        Args:
            **kwargs: Settings to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_settings[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            "ocr_enabled": self.ocr_enabled,
            "extract_images": self.extract_images,
            "extract_tables": self.extract_tables,
            "extract_forms": self.extract_forms,
            "preserve_layout": self.preserve_layout,
            "language": self.language,
            "dpi": self.dpi,
            "max_pages": self.max_pages,
            "timeout": self.timeout,
        }
        config_dict.update(self.additional_settings)
        return config_dict 