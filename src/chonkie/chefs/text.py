"""Text preprocessing chef implementations."""

import importlib.util as importutil
import re
from typing import List, Optional

from .base import BaseChef


class TextCleanerChef(BaseChef):
    """Chef for basic text cleaning operations."""
    
    def __init__(
        self,
        normalize_whitespace: bool = True,
        strip: bool = True,
        lowercase: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_numbers: bool = False,
    ):
        """Initialize the TextCleanerChef.
        
        Args:
            normalize_whitespace: Whether to normalize whitespace (replace multiple spaces with single space).
            strip: Whether to strip whitespace from the beginning and end of the text.
            lowercase: Whether to convert text to lowercase.
            remove_urls: Whether to remove URLs from the text.
            remove_emails: Whether to remove email addresses from the text.
            remove_numbers: Whether to remove numbers from the text.
        """
        super().__init__()
        self.normalize_whitespace = normalize_whitespace
        self.strip = strip
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        This chef has no external dependencies, so it's always available.
        
        Returns:
            bool: True
        """
        return True
        
    def preprocess(self, text: str) -> str:
        """Clean the text according to the configured options.
        
        Args:
            text: The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        
        # Apply text transformations in a specific order
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
            
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            
        if self.strip:
            text = text.strip()
            
        if self.lowercase:
            text = text.lower()
            
        return text
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return (
            f"TextCleanerChef(normalize_whitespace={self.normalize_whitespace}, "
            f"strip={self.strip}, lowercase={self.lowercase}, "
            f"remove_urls={self.remove_urls}, remove_emails={self.remove_emails}, "
            f"remove_numbers={self.remove_numbers})"
        )


class HTMLCleanerChef(BaseChef):
    """Chef for converting HTML to plain text."""
    
    def __init__(self, strip_tags: bool = True, preserve_line_breaks: bool = True):
        """Initialize the HTMLCleanerChef.
        
        Args:
            strip_tags: Whether to strip HTML tags.
            preserve_line_breaks: Whether to preserve line breaks in the HTML.
        """
        super().__init__()
        self.strip_tags = strip_tags
        self.preserve_line_breaks = preserve_line_breaks
        self._bs4_available = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """Check if BeautifulSoup is available."""
        return importutil.find_spec("bs4") is not None
    
    def _import_dependencies(self) -> None:
        """Import BeautifulSoup."""
        if self._bs4_available:
            global BeautifulSoup
            from bs4 import BeautifulSoup
        else:
            raise ImportError(
                "BeautifulSoup is required for HTMLCleanerChef. "
                "Please install it via `pip install chonkie[html]` or `pip install bs4`."
            )
    
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        Returns:
            bool: True if the chef dependencies are available, False otherwise.
        """
        return self._bs4_available
    
    def preprocess(self, html: str) -> str:
        """Convert HTML to plain text.
        
        Args:
            html: The HTML to convert.
            
        Returns:
            str: The extracted plain text.
        """
        if not isinstance(html, str):
            raise ValueError("Input must be a string.")
            
        # Return quickly if the input is empty
        if not html:
            return ""
            
        # If stripping tags is disabled, return the original HTML
        if not self.strip_tags:
            return html
        
        # Import BeautifulSoup if needed
        self._import_dependencies()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Handle line breaks if needed
        if self.preserve_line_breaks:
            # Replace <br> with newlines before extracting text
            for br in soup.find_all('br'):
                br.replace_with('\n')
            
            # Replace </p> with newlines
            for p in soup.find_all('p'):
                p.append('\n')
        
        # Get the text content without HTML tags
        text = soup.get_text()
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return f"HTMLCleanerChef(strip_tags={self.strip_tags}, preserve_line_breaks={self.preserve_line_breaks})"


class MarkdownCleanerChef(BaseChef):
    """Chef for converting Markdown to plain text."""
    
    def __init__(
        self,
        strip_markdown: bool = True,
        preserve_headings: bool = True,
        preserve_links: bool = False,
    ):
        """Initialize the MarkdownCleanerChef.
        
        Args:
            strip_markdown: Whether to strip Markdown formatting.
            preserve_headings: Whether to preserve headings structure.
            preserve_links: Whether to preserve links as "text (url)".
        """
        super().__init__()
        self.strip_markdown = strip_markdown
        self.preserve_headings = preserve_headings
        self.preserve_links = preserve_links
        self._markdown_available = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """Check if markdown is available."""
        return importutil.find_spec("markdown") is not None
    
    def _import_dependencies(self) -> None:
        """Import markdown."""
        if self._markdown_available:
            global markdown
            import markdown
        else:
            raise ImportError(
                "markdown is required for MarkdownCleanerChef. "
                "Please install it via `pip install chonkie[markdown]` or `pip install markdown`."
            )
    
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        Returns:
            bool: True if the chef dependencies are available, False otherwise.
        """
        return self._markdown_available
    
    def preprocess(self, md_text: str) -> str:
        """Convert Markdown to plain text.
        
        Args:
            md_text: The Markdown text to convert.
            
        Returns:
            str: The extracted plain text.
        """
        if not isinstance(md_text, str):
            raise ValueError("Input must be a string.")
            
        # Return quickly if the input is empty
        if not md_text or not self.strip_markdown:
            return md_text
            
        # For simple text extraction without preserving structure,
        # we can use regex-based approach
        
        text = md_text
        
        # Process links
        if self.preserve_links:
            # Convert [text](url) to "text (url)"
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        else:
            # Remove URLs in Markdown links and keep just the text
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
        
        # Process headings (# Heading)
        if self.preserve_headings:
            # Keep heading text with level indicator
            text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        else:
            # Remove heading markers
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove other Markdown formatting
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)  # Bold
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)     # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)         # Strikethrough
        text = re.sub(r'`([^`]+)`', r'\1', text)         # Inline code
        text = re.sub(r'```\w*\n(.*?)```', r'\1', text, flags=re.DOTALL)  # Code blocks
        
        # Remove list markers
        text = re.sub(r'^[\*\-+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return (
            f"MarkdownCleanerChef(strip_markdown={self.strip_markdown}, "
            f"preserve_headings={self.preserve_headings}, preserve_links={self.preserve_links})"
        ) 