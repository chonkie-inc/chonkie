"""MarkitdownChef implementation for Chonkie.

This module provides the MarkitdownChef class, which is responsible for
processing markdown files and converting them to structured documents.
"""

import os
from typing import Any, Dict, Optional

from .base import BaseChef, ProcessingResult, ProcessingStatus
from .config import MarkdownChefConfig
from .exceptions import ContentExtractionError
from ..types import Document


class MarkitdownChef(BaseChef):
    """Chef for processing markdown files.
    
    This Chef processes markdown files and converts them to structured documents.
    It supports various markdown features and can handle different markdown flavors.
    
    Attributes:
        name: The name of the Chef
        version: The version of the Chef
        config: Configuration settings for the Chef
    """
    
    def __init__(
        self,
        name: str = "MarkitdownChef",
        version: str = "1.0.0",
        config: Optional[MarkdownChefConfig] = None
    ):
        """Initialize the MarkitdownChef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            config: Optional configuration settings
        """
        config = config or MarkdownChefConfig()
        super().__init__(name, version, ["md", "markdown"], config)
        self._import_dependencies()

    def _import_dependencies(self) -> None:
        """Import required dependencies."""
        try:
            import markdown
            self.markdown = markdown
        except ImportError:
            raise ImportError(
                "Markdown dependencies not found. Please install them using "
                "`pip install chonkie[markdown]`"
            )

    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process a markdown file.
        
        Args:
            file_path: Path to the markdown file to process
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult containing the processed document and metadata
            
        Raises:
            ContentExtractionError: If content extraction fails
        """
        try:
            # Validate the file
            if not self.validate_file(file_path):
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error="Invalid markdown file"
                )

            # Read the markdown content
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Convert markdown to HTML
            html_content = None
            if self.config.html_output:
                html_content = self.markdown.markdown(
                    markdown_content,
                    extensions=self.config.extensions
                )

            # Create a new Document
            document = Document(
                text=markdown_content,
                metadata={
                    "html_content": html_content,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "processing_engine": "markdown"
                }
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                document=document,
                metadata={
                    "processing_engine": "markdown",
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path)
                }
            )

        except (IOError, FileNotFoundError, PermissionError, ValueError, AttributeError, TypeError) as e:
            raise ContentExtractionError(f"Failed to process markdown file: {str(e)}") from e

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a valid markdown file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a valid markdown file, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            if not file_path.lower().endswith(('.md', '.markdown')):
                return False
            return True
        except Exception:
            return False 