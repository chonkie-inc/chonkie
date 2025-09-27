"""Markdown chef for Chonkie."""

import re
from pathlib import Path
from typing import Tuple, Union

from typing_extensions import List

from chonkie.tokenizer import Tokenizer
from chonkie.types import (
  Chunk,
  MarkdownCode,
  MarkdownDocument,
  MarkdownImage,
  MarkdownTable,
)

from .base import BaseChef


class MarkdownChef(BaseChef):
  """Chef to process a markdown file into a MarkdownDocument type.
  
  Args:
    path (Union[str, Path]): The path to the markdown file.

  Returns:
    MarkdownDocument: The processed markdown document.

  """

  def __init__(self, tokenizer: Union[Tokenizer, str] = "character") -> None:
    """Initialize the MarkdownChef."""
    super().__init__()
    self.tokenizer = tokenizer if isinstance(tokenizer, Tokenizer) else Tokenizer(tokenizer)
    self.code_pattern = re.compile(r"```([a-zA-Z0-9+\-_]*)\n?(.*?)\n?```", re.DOTALL)
    self.table_pattern = re.compile(r"(\|.*?\n\|[-: ]+\|.*?\n(?:\|.*?\n)*)")
    self.image_pattern = re.compile(r"(\[)?!\[([^\]]*)\]\(([^)]+)\)(?(1)\]\(([^)]+)\)|)")

  def prepare_tables(self, markdown: str) -> List[MarkdownTable]:
    """Prepare the tables for the MarkdownDocument.

    Args:
        markdown (str): The markdown text containing tables.

    Returns:
        List[MarkdownTable]: The list of tables with their start and end indices.

    """
    markdown_tables: List[MarkdownTable] = []
    for match in self.table_pattern.finditer(markdown):
        table_content = match.group(0)
        start_index = match.start()
        end_index = match.end()
        markdown_tables.append(MarkdownTable(content=table_content, start_index=start_index, end_index=end_index))
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

  def extract_images(self, markdown: str) -> List[MarkdownImage]:
    """Extract images from a markdown string.

    Args:
        markdown (str): The markdown text containing images.

    Returns:
        Dict[str, str]: A dictionary where keys are image names (alt text or filename)
        and values are image paths or base64 data URLs.

    """
    images: List[MarkdownImage] = []

    for match in self.image_pattern.finditer(markdown):
        # Extract the match groups
        _, alt_text, image_src, link_url = match.groups()
        
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

        images.append(MarkdownImage(
            alias=key,
            content=image_src,
            start_index=match.start(),
            end_index=match.end(),
            link=link_url
        ))

    return images

  def extract_chunks(
    self,
    markdown: str,
    tables: List[MarkdownTable],
    code: List[MarkdownCode],
    images: List[MarkdownImage]) -> List[Chunk]:
    """Parse out the remaining markdown content into chunks.

    Args:
        markdown (str): The markdown text containing the remaining content.
        tables (List[MarkdownTable]): The list of tables.
        code (List[MarkdownCode]): The list of code snippets.
        images (List[MarkdownImage]): The list of images.

    Returns:
        List[Chunk]: The list of chunks.

    """
    chunks: List[Chunk] = []

    # Get all the occupied
    occupied_indices: List[Tuple[int, int]] = []
    occupied_indices.extend([(table.start_index, table.end_index) for table in tables])
    occupied_indices.extend([(code.start_index, code.end_index) for code in code])
    occupied_indices.extend([(image.start_index, image.end_index) for image in images])

    # Sort the occupied indices, by start and end index
    occupied_indices.sort(key=lambda x: (x[0], x[1]))

    # Get the remaining indices
    current_index = 0
    remaining_indices: List[Tuple[int, int]] = []
    for index in occupied_indices:
      if index[0] > current_index:
        remaining_indices.append((current_index, index[0]))
      current_index = index[1]
    if current_index < len(markdown):
      remaining_indices.append((current_index, len(markdown)))

    # Get the chunks
    for index in remaining_indices:
      # Start and end index
      start_index = index[0]
      end_index = index[1]
      text = markdown[start_index:end_index]

      # Only create chunk if it contains meaningful content (not just whitespace)
      if text.strip():
        token_count = self.tokenizer.count_tokens(text)
        chunks.append(Chunk(text=text, start_index=start_index, end_index=end_index, token_count=token_count))

    return chunks

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

    # Extract the chunks
    chunks: List[Chunk] = self.extract_chunks(markdown, tables, code, images)

    return MarkdownDocument(
      content=markdown,
      tables=tables,
      code=code,
      images=images,
      chunks=chunks
    )