"""Markdown chef for Chonkie."""

import re
from pathlib import Path
from typing import Dict, Union

from typing_extensions import List

from chonkie.types import MarkdownCode, MarkdownDocument, MarkdownTable

from .base import BaseChef


class MarkdownChef(BaseChef):
  """Chef to process a markdown file into a MarkdownDocument type.
  
  Args:
    path (Union[str, Path]): The path to the markdown file.

  Returns:
    MarkdownDocument: The processed markdown document.

  """

  def __init__(self) -> None:
    """Initialize the MarkdownChef."""
    super().__init__()
    self.code_pattern = re.compile(r"```([a-zA-Z0-9+\-_]*)\n?(.*?)\n?```", re.DOTALL)
    self.table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")
    self.image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

  def extract_tables(self, markdown: str) -> List[str]:
    """Extract markdown tables from a markdown string.

    Args:
        markdown (str): The markdown text containing tables.

    Returns:
        List[str]: A list of strings, each representing a markdown table found in the input.

    """
    tables: List[str] = []
    for match in self.table_pattern.finditer(markdown):
        tables.append(match.group(0))
    return tables

  def prepare_tables(self, markdown: str) -> List[MarkdownTable]:
    """Prepare the tables for the MarkdownDocument.

    Args:
        markdown (str): The markdown text containing tables.

    Returns:
        List[MarkdownTable]: The list of tables with their start and end indices.

    """
    # Extract the tables from the markdown
    tables = self.extract_tables(markdown)
    
    # Convert the extracted tables into MarkdownTables
    markdown_tables: List[MarkdownTable] = []
    for table in tables:
      start_index = markdown.find(table)
      end_index = start_index + len(table)
      markdown_tables.append(MarkdownTable(content=table, start_index=start_index, end_index=end_index))
    return markdown_tables

  def prepare_code(self, markdown: str) -> List[MarkdownCode]:
    """Extract markdown code snippets from a markdown string.

    Args:
        markdown (str): The markdown text containing code snippets.

    Returns:
        List[MarkdownCode]: A list of MarkdownCode objects, each containing
        the code content, language (if specified), and position indices.

    """
    # Pattern to capture language and content separately
    code_snippets: List[MarkdownCode] = []
    for match in self.code_pattern.finditer(markdown):
        language = match.group(1) if match.group(1) else None
        content = match.group(2)
        
        start_index = match.start()
        end_index = match.end()
        
        code_snippets.append(MarkdownCode(
            content=content,
            language=language,
            start_index=start_index,
            end_index=end_index
        ))
    return code_snippets

  def extract_images(self, markdown: str) -> Dict[str, str]:
    """Extract images from a markdown string.

    Args:
        markdown (str): The markdown text containing images.

    Returns:
        Dict[str, str]: A dictionary where keys are image names (alt text or filename)
        and values are image paths or base64 data URLs.

    """
    images: Dict[str, str] = {}

    for match in self.image_pattern.finditer(markdown):
        alt_text = match.group(1)
        image_src = match.group(2)

        # Determine the key for the image
        if alt_text:
            key = alt_text
        else:
            # If no alt text, use filename from path
            if image_src.startswith("data:"):
                # For base64 data URLs, use a generic name or extract from data URL
                key = "base64_image"
            else:
                # Extract filename from path
                key = Path(image_src).name

        # Handle duplicate keys by appending a counter
        original_key = key
        counter = 1
        while key in images:
            key = f"{original_key}_{counter}"
            counter += 1

        images[key] = image_src

    return images

  def process(self, path: Union[str, Path]) -> MarkdownDocument:
    """Process a markdown file into a MarkdownDocument.

    Args:
        path (Union[str, Path]): The path to the markdown file.

    Returns:
        MarkdownDocument: The processed markdown document.

    """
    # Read the markdown file
    markdown = self.read(path)

    # Extract all the tables, code snippets, and images
    tables = self.prepare_tables(markdown)
    code = self.prepare_code(markdown)
    images = self.extract_images(markdown)

    return MarkdownDocument(
      content=markdown,
      tables=tables,
      code=code,
      images=images
    )