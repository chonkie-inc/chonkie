"""Implementation of the MistralOCR chef for processing images and PDFs."""

import base64
import os
from pathlib import Path
from typing import Optional

from chonkie.logger import get_logger
from chonkie.pipeline import chef
from chonkie.types import Document, MarkdownDocument

from .base import BaseChef

logger = get_logger(__name__)


@chef("mistral")
class MistralOCR(BaseChef):
    """Processes images and PDFs using Mistral's OCR capabilities.

    This chef uses the Mistral AI OCR API to extract text from images and PDF files,
    returning the content as a MarkdownDocument.

    """

    SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
    SUPPORTED_PDF_TYPES = {".pdf"}
    SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_PDF_TYPES

    MIME_MAP = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".pdf": "application/pdf",
    }

    def __init__(
        self,
        model: str = "mistral-ocr-latest",
        api_key: Optional[str] = None,
    ):
        """Initialize the MistralOCR chef.

        Args:
            model: The Mistral OCR model to use.
            api_key: The API key. Falls back to MISTRAL_API_KEY env var.

        """
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MistralOCR requires an API key. Either pass the `api_key` parameter "
                "or set the `MISTRAL_API_KEY` in your environment.",
            )

        try:
            from mistralai.client import Mistral
        except ImportError as ie:
            raise ImportError(
                "One or more of the required modules are not available: [mistralai]. "
                "Please install the dependencies via `pip install chonkie[mistral]`"
            ) from ie

        self.client = Mistral(api_key=api_key)
        self.model = model

    def _ocr(self, path: Path) -> str:
        """Run OCR on a file and return extracted markdown text."""
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported types: {sorted(self.SUPPORTED_TYPES)}"
            )

        mime_type = self.MIME_MAP[suffix]
        file_bytes = path.read_bytes()
        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"

        if suffix in self.SUPPORTED_PDF_TYPES:
            document = {"type": "document_url", "document_url": data_uri}
        else:
            document = {"type": "image_url", "image_url": data_uri}

        response = self.client.ocr.process(model=self.model, document=document)

        pages_text = []
        for page in response.pages:
            pages_text.append(page.markdown)
        return "\n\n".join(pages_text)

    def process(self, path: str | os.PathLike) -> MarkdownDocument:
        """Extract text from an image or PDF file.

        Args:
            path: Path to the image or PDF file.

        Returns:
            MarkdownDocument with extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.

        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.debug(f"Processing file with MistralOCR: {path}")
        content = self._ocr(p)
        logger.info(f"MistralOCR processing complete: extracted {len(content)} characters from {path}")

        doc = MarkdownDocument(content=content)
        self._set_source_filename(doc, path)
        return doc

    def parse(self, text: str) -> Document:
        """Parse raw text into a Document.

        Since MistralOCR operates on files, this wraps the text as-is.

        Args:
            text: Raw text to parse.

        Returns:
            Document created from the text.

        """
        return Document(content=text)

    def __repr__(self) -> str:
        """Return a string representation of the MistralOCR instance."""
        return f"MistralOCR(model={self.model})"
