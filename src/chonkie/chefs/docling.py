"""DoclingChef implementation for Chonkie.

This module provides the DoclingChef class, which is responsible for
processing documentation files and extracting structured information.
"""

import os
from typing import Any, Dict, Optional, List
import re

from .base import BaseChef, ProcessingResult, ProcessingStatus
from .config import DocChefConfig
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
        config: Optional[DocChefConfig] = None
    ):
        """Initialize the DoclingChef.
        
        Args:
            name: The name of the Chef
            version: The version of the Chef
            config: Optional configuration settings
        """
        config = config or DocChefConfig()
        super().__init__(name, version, ["md", "markdown", "rst", "txt"], config)
        self._import_dependencies()

    def _import_dependencies(self) -> None:
        """Import required dependencies."""
        try:
            # Import docutils for reStructuredText processing
            import docutils
            from docutils.core import publish_parts
            self.docutils = docutils
            self.publish_parts = publish_parts
            
            # Import markdown for markdown processing
            import markdown
            from markdown.extensions import toc
            self.markdown = markdown
            self.toc_extension = toc
        except ImportError:
            raise ImportError(
                "Dependencies not found. Please install them using "
                "`pip install chonkie[docutils,markdown]`"
            )

    def _extract_sections(self, content: str, file_format: str = 'md') -> List[Dict[str, Any]]:
        """Extract sections from the content using appropriate parsers.
        
        Args:
            content: The content to extract sections from
            file_format: Format of the content ('md', 'rst', etc.)
            
        Returns:
            List of dictionaries containing section information
        """
        sections = []
        
        if file_format in ['md', 'markdown']:
            # Use the markdown library to extract headings
            # First, create a TOC to help parse the document structure
            md = self.markdown.Markdown(extensions=['toc'])
            md.convert(content)
            
            # Use regex to find all headings and their content
            heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)$', re.MULTILINE)
            matches = list(heading_pattern.finditer(content))
            
            # Process each heading and collect content
            for i, match in enumerate(matches):
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Find the content between this heading and the next
                start_pos = match.end()
                end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)
                
                section_content = content[start_pos:end_pos].strip()
                
                # Remove any headings from the content
                section_content = re.sub(r'^#{1,6}\s+.*?$', '', section_content, flags=re.MULTILINE).strip()
                
                sections.append({
                    'title': title,
                    'content': section_content,
                    'level': level
                })
                
            # Check for content before the first heading
            if matches and matches[0].start() > 0:
                pre_content = content[:matches[0].start()].strip()
                if pre_content:
                    sections.insert(0, {
                        'title': 'Introduction',
                        'content': pre_content,
                        'level': 0
                    })
        
        elif file_format == 'rst':
            # For reStructuredText, we can parse the document structure
            # by examining the HTML output's structure
            html_parts = self.publish_parts(content, writer_name='html')
            
            # Use the document structure from docutils
            doc = self.docutils.core.publish_doctree(content)
            
            # Extract sections from the document tree
            for section in doc.findall(self.docutils.nodes.section):
                title_nodes = [n for n in section.children if isinstance(n, self.docutils.nodes.title)]
                if title_nodes:
                    title = title_nodes[0].astext()
                    level = section['level'] if 'level' in section else 1
                    
                    # Extract content by removing the title node
                    content_nodes = [n for n in section.children if not isinstance(n, self.docutils.nodes.title)]
                    section_content = "\n".join(n.astext() for n in content_nodes)
                    
                    sections.append({
                        'title': title,
                        'content': section_content,
                        'level': level
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

            # Determine file format
            file_format = os.path.splitext(file_path.lower())[1][1:]
            if file_format in ['markdown', 'md']:
                file_format = 'md'
            elif file_format == 'rst':
                file_format = 'rst'
            else:
                file_format = 'txt'

            # Extract sections and code blocks
            sections = self._extract_sections(content, file_format)
            code_blocks = self._extract_code_blocks(content)

            # Convert to HTML if it's a reStructuredText file
            html_content = None
            if file_format == 'rst':
                html_content = self.publish_parts(
                    content,
                    writer_name='html'
                )['html_body']
            elif file_format == 'md':
                html_content = self.markdown.markdown(content)

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

        except (IOError, FileNotFoundError, PermissionError) as e:
            # File-related errors (not found, permission denied, etc.)
            raise ContentExtractionError(f"Failed to read documentation file: {str(e)}") from e
        except (ValueError, AttributeError, TypeError) as e:
            # Data parsing and processing errors
            raise ContentExtractionError(f"Failed to parse documentation content: {str(e)}") from e
        except self.docutils.ApplicationError as e:
            # Handle docutils specific errors
            raise ContentExtractionError(f"Failed to process reStructuredText: {str(e)}") from e
        except Exception as e:
            # Fallback for any other unexpected exceptions
            raise ContentExtractionError(f"Unexpected error processing documentation file: {str(e)}") from e

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