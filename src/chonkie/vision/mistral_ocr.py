"""Implementation of the MistralOCR class for processing images and PDFs."""

import base64
import importlib.util as importutil
import os
from pathlib import Path
from typing import Optional, Union

from chonkie.pipeline import vision

from .base import BaseVision


@vision("mistral")
class MistralOCR(BaseVision):
    """Processes images and PDFs using Mistral's OCR capabilities.

    This class uses the Mistral AI OCR API to extract text from images and PDF files,
    returning the content as markdown.

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
        """Initialize the MistralOCR class.

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

    def process(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from an image or PDF file.

        Args:
            file_path: Path to the image or PDF file.

        Returns:
            Extracted text content as markdown.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.

        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

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

    def process_batch(self, file_paths: list[Union[str, os.PathLike]]) -> list[str]:
        """Extract text from multiple image or PDF files.

        Args:
            file_paths: List of paths to image or PDF files.

        Returns:
            List of extracted text content, one per file.

        """
        return [self.process(fp) for fp in file_paths]

    async def aprocess(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from an image or PDF file asynchronously.

        Args:
            file_path: Path to the image or PDF file.

        Returns:
            Extracted text content as markdown.

        """
        import asyncio

        return await asyncio.to_thread(self.process, file_path)

    async def aprocess_batch(
        self, file_paths: list[Union[str, os.PathLike]], max_concurrency: int = 5
    ) -> list[str]:
        """Extract text from multiple files asynchronously.

        Args:
            file_paths: List of paths to image or PDF files.
            max_concurrency: Maximum number of concurrent requests.

        Returns:
            List of extracted text content.

        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_process(fp: Union[str, os.PathLike]) -> str:
            async with semaphore:
                return await self.aprocess(fp)

        return await asyncio.gather(*[_bounded_process(fp) for fp in file_paths])

    @classmethod
    def _is_available(cls) -> bool:
        """Check if all required dependencies are available."""
        return importutil.find_spec("mistralai") is not None

    def __call__(self, file_path: Union[str, os.PathLike]) -> str:
        """Extract text from an image or PDF file."""
        return self.process(file_path)

    def __repr__(self) -> str:
        """Return a string representation of the MistralOCR instance."""
        return f"MistralOCR(model={self.model})"
