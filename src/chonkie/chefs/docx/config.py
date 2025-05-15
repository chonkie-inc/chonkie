from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator

class DOCXExtractorConfig(BaseModel):
    """Configuration for DOCX document extraction."""
    
    extract_metadata: bool = Field(True, description="Whether to extract document metadata")
    extract_images: bool = Field(True, description="Whether to extract embedded images")
    image_format: str = Field("png", description="Format to save extracted images in (png, jpeg)")
    image_quality: int = Field(85, description="Image quality for JPEG format (1-100)")
    
    # New configuration options
    extract_tables: bool = Field(True, description="Whether to extract tables with structure")
    extract_headers_footers: bool = Field(True, description="Whether to extract headers and footers")
    extract_comments: bool = Field(True, description="Whether to extract comments and track changes")
    extract_styles: bool = Field(True, description="Whether to extract text styles and formatting")
    extract_hyperlinks: bool = Field(True, description="Whether to extract hyperlinks")
    extract_lists: bool = Field(True, description="Whether to extract lists and numbering")
    
    # Table extraction options
    table_format: str = Field("markdown", description="Format for table output (markdown, html, json)")
    preserve_table_structure: bool = Field(True, description="Whether to preserve table structure in output")
    
    # Image extraction options
    extract_image_metadata: bool = Field(True, description="Whether to extract image metadata")
    image_output_dir: Optional[str] = Field(None, description="Directory to save extracted images")
    image_naming_pattern: str = Field("image_{index}", description="Pattern for naming extracted images")
    
    # Style extraction options
    extract_paragraph_styles: bool = Field(True, description="Whether to extract paragraph styles")
    extract_character_styles: bool = Field(True, description="Whether to extract character styles")
    extract_section_styles: bool = Field(True, description="Whether to extract section styles")
    
    # New enhanced features
    extract_list_structure: bool = Field(True, description="Whether to extract list structure with nesting levels")
    extract_section_properties: bool = Field(True, description="Whether to extract section properties (page size, margins, etc.)")
    extract_bookmarks: bool = Field(True, description="Whether to extract bookmarks and cross-references")
    extract_footnotes_endnotes: bool = Field(True, description="Whether to extract footnotes and endnotes")
    extract_equations: bool = Field(True, description="Whether to extract mathematical equations")
    extract_form_fields: bool = Field(True, description="Whether to extract form fields and their properties")
    
    @field_validator("table_format")
    def validate_table_format(cls, v: str) -> str:
        """Validate table format."""
        valid_formats = ["markdown", "html", "json"]
        if v not in valid_formats:
            raise ValueError(f"Invalid table format. Must be one of {valid_formats}")
        return v

    @field_validator("image_format")
    def validate_image_format(cls, v: str) -> str:
        """Validate image format."""
        valid_formats = ["png", "jpeg"]
        if v not in valid_formats:
            raise ValueError(f"Invalid image format. Must be one of {valid_formats}")
        return v

    @field_validator("image_quality")
    def validate_image_quality(cls, v: int) -> int:
        """Validate image quality."""
        if not (1 <= v <= 100):
            raise ValueError("Image quality must be between 1 and 100")
        return v

    def validate(self) -> None:
        """Validate configuration values."""
        if self.image_format not in ["png", "jpeg"]:
            raise ValueError("Image format must be 'png' or 'jpeg'")
        
        if not 1 <= self.image_quality <= 100:
            raise ValueError("Image quality must be between 1 and 100")
        
        if self.table_format not in ["markdown", "html", "json"]:
            raise ValueError("Table format must be 'markdown', 'html', or 'json'")
        
        if self.image_output_dir is not None and not isinstance(self.image_output_dir, str):
            raise ValueError("Image output directory must be a string") 