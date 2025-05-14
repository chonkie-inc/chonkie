"""DoclingChef implementation for Chonkie.

This module provides the DoclingChef class, which is responsible for
processing documentation files and extracting structured information.
"""

import os
from typing import Any, Dict, Optional, List
import re

from .base import BaseChef, ProcessingResult, ProcessingStatus
from .config import ChefConfig
from .exceptions import ContentExtractionError
from ..types import Document


class DoclingChef(BaseChef):
    """Chef for processing documentation files.
    
    This Chef processes documentation files and extracts structured information
    such as sections, code blocks, and metadata. It supports various documentation
    formats and can handle different documentation styles.
    
    Attributes:
        name: The name of the Chef
        version: The version of the Chef
        config: Configuration settings for the Chef
    """
    
    def __init__(
        self,
        name: str = "DoclingChef",
        version: str = "1.0.0",
        config: Optional[ChefConfig] = None
    ):
        """Initialize the DoclingChef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            config: Optional configuration settings
        """
        super().__init__(name, version, ["md", "markdown", "rst", "txt"], config)
        self._import_dependencies()

    def _import_dependencies(self) -> None:
        """Import required dependencies."""
        try:
            import docutils
            from docutils.core import publish_parts
            self.docutils = docutils
            self.publish_parts = publish_parts
        except ImportError:
            raise ImportError(
                "Docutils dependencies not found. Please install them using "
                "`pip install chonkie[docutils]`"
            )

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from the content.
        
        Args:
            content: The content to extract sections from
            
        Returns:
            List of dictionaries containing section information
        """
        sections = []
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            # Check for section headers
            if line.startswith('# '):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'level': 1
                    })
                current_section = line[2:].strip()
                current_content = []
            elif line.startswith('## '):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'level': 1
                    })
                current_section = line[3:].strip()
                current_content = []
                sections.append({
                    'title': current_section,
                    'content': '',
                    'level': 2
                })
            else:
                if current_section:
                    current_content.append(line)
        
        # Add the last section
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip(),
                'level': 1
            })
        
        return sections

    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from the content.
        
        Args:
            content: The content to extract code blocks from
            
        Returns:
            List of dictionaries containing code block information
        """
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            code_blocks.append({
                'language': language,
                'code': code
            })
        
        return code_blocks

    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process a documentation file.
        
        Args:
            file_path: Path to the documentation file to process
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
                    error="Invalid documentation file"
                )

            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract sections and code blocks
            sections = self._extract_sections(content)
            code_blocks = self._extract_code_blocks(content)

            # Convert to HTML if it's a reStructuredText file
            html_content = None
            if file_path.lower().endswith('.rst'):
                html_content = self.publish_parts(
                    content,
                    writer_name='html'
                )['html_body']

            # Create a new Document
            document = Document(
                text=content,
                metadata={
                    'sections': sections,
                    'code_blocks': code_blocks,
                    'html_content': html_content,
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'processing_engine': 'docling'
                }
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                document=document,
                metadata={
                    'processing_engine': 'docling',
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'sections_count': len(sections),
                    'code_blocks_count': len(code_blocks)
                }
            )

        except Exception as e:
            raise ContentExtractionError(f"Failed to process documentation file: {str(e)}") from e

    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a valid documentation file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if the file is a valid documentation file, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            if not file_path.lower().endswith(('.md', '.markdown', '.rst', '.txt')):
                return False
            return True
        except Exception:
            return False 