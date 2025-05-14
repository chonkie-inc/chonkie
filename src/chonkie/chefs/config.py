"""Configuration management for Chonkie Chefs."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BaseChefConfig:
    """Base configuration settings for all Chef types.
    
    Attributes:
        max_file_size: Maximum file size in bytes to process
        timeout: Processing timeout in seconds
        additional_settings: Additional Chef-specific settings
    """
    max_file_size: Optional[int] = None  # None means no limit
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
            "timeout": self.timeout,
            "max_file_size": self.max_file_size,
        }
        config_dict.update(self.additional_settings)
        return config_dict


@dataclass
class PDFChefConfig(BaseChefConfig):
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
    """
    ocr_enabled: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    extract_forms: bool = False
    preserve_layout: bool = True
    language: str = "eng"
    dpi: int = 300
    max_pages: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = super().to_dict()
        pdf_config = {
            "ocr_enabled": self.ocr_enabled,
            "extract_images": self.extract_images,
            "extract_tables": self.extract_tables,
            "extract_forms": self.extract_forms,
            "preserve_layout": self.preserve_layout,
            "language": self.language,
            "dpi": self.dpi,
            "max_pages": self.max_pages,
        }
        config_dict.update(pdf_config)
        return config_dict


@dataclass
class MarkdownChefConfig(BaseChefConfig):
    """Configuration settings for Markdown processing Chefs.
    
    Attributes:
        html_output: Whether to generate HTML output
        parse_metadata: Whether to parse YAML metadata from the markdown
        extensions: List of markdown extensions to use
    """
    html_output: bool = True
    parse_metadata: bool = True
    extensions: list = field(default_factory=lambda: ['extra', 'codehilite', 'tables'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = super().to_dict()
        md_config = {
            "html_output": self.html_output,
            "parse_metadata": self.parse_metadata,
            "extensions": self.extensions,
        }
        config_dict.update(md_config)
        return config_dict


@dataclass
class DocChefConfig(BaseChefConfig):
    """Configuration settings for documentation file processing Chefs.
    
    Attributes:
        extract_sections: Whether to extract sections from the document
        extract_code_blocks: Whether to extract code blocks
        html_output: Whether to generate HTML output
    """
    extract_sections: bool = True
    extract_code_blocks: bool = True
    html_output: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = super().to_dict()
        doc_config = {
            "extract_sections": self.extract_sections,
            "extract_code_blocks": self.extract_code_blocks,
            "html_output": self.html_output,
        }
        config_dict.update(doc_config)
        return config_dict


# For backward compatibility
ChefConfig = BaseChefConfig 