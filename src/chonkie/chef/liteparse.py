"""Implementation of the LiteParse chef for processing documents locally."""

import os
from pathlib import Path
from typing import Optional

from chonkie.logger import get_logger
from chonkie.pipeline import chef
from chonkie.types import Document

from .base import BaseChef

logger = get_logger(__name__)


@chef("liteparse")
class LiteParse(BaseChef):
    """Processes documents using LiteParse from LlamaIndex.

    This chef uses LiteParse to extract text from PDFs, office documents, and images
    locally without any cloud API dependencies.

    """

    SUPPORTED_PDF_TYPES = {".pdf"}
    SUPPORTED_OFFICE_TYPES = {
        ".doc",
        ".docx",
        ".docm",
        ".odt",
        ".rtf",
        ".ppt",
        ".pptx",
        ".pptm",
        ".odp",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".ods",
        ".csv",
        ".tsv",
    }
    SUPPORTED_IMAGE_TYPES = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".svg",
    }
    SUPPORTED_TYPES = SUPPORTED_PDF_TYPES | SUPPORTED_OFFICE_TYPES | SUPPORTED_IMAGE_TYPES

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_language: str = "en",
        ocr_server_url: Optional[str] = None,
        max_pages: int = 10000,
        target_pages: Optional[str] = None,
        dpi: int = 150,
        num_workers: Optional[int] = None,
        password: Optional[str] = None,
    ):
        """Initialize the LiteParse chef.

        Args:
            ocr_enabled: Whether to enable OCR for scanned/image text.
            ocr_language: Language code for OCR (e.g., "en", "fr", "de").
            ocr_server_url: Optional HTTP OCR server URL.
            max_pages: Maximum number of pages to parse.
            target_pages: Specific pages to parse (e.g., "1-5,10").
            dpi: Rendering resolution for PDF pages.
            num_workers: Number of pages to OCR in parallel.
            password: Password for protected PDFs.

        """
        try:
            from liteparse import LiteParse as _LiteParse
        except ImportError as ie:
            raise ImportError(
                "The required module is not available: [liteparse]. "
                "Please install the dependency via `pip install chonkie[liteparse]`"
            ) from ie

        if ocr_enabled and not os.environ.get("TESSDATA_PREFIX"):
            self._auto_detect_tessdata()

        self.parser = _LiteParse(  # type: ignore[call-arg]
            ocr_enabled=ocr_enabled,
            ocr_language=ocr_language,
            ocr_server_url=ocr_server_url,
            max_pages=max_pages,
            target_pages=target_pages,
            dpi=dpi,
            num_workers=num_workers,
            password=password,
        )
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language

    @staticmethod
    def _auto_detect_tessdata() -> None:
        """Set TESSDATA_PREFIX if Tesseract is installed in a standard location."""
        candidates = [
            Path("C:/Program Files/Tesseract-OCR/tessdata"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/share/tessdata"),
            Path("/usr/local/share/tessdata"),
            Path("/opt/homebrew/share/tessdata"),
        ]
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "eng.traineddata").exists():
                os.environ["TESSDATA_PREFIX"] = str(candidate)
                logger.debug(f"Auto-detected TESSDATA_PREFIX: {candidate}")
                return

    def process(self, path: str | os.PathLike) -> Document:
        """Extract text from a document file.

        Args:
            path: Path to the file to process.

        Returns:
            Document with extracted text content.

        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = p.suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported types: {sorted(self.SUPPORTED_TYPES)}"
            )

        logger.debug(f"Processing file with LiteParse: {path}")
        result = self.parser.parse(str(p))
        content = result.text

        logger.info(
            f"LiteParse processing complete: extracted {len(content)} characters from {path}"
        )

        doc = Document(content=content)
        self._set_source_filename(doc, path)
        return doc

    def parse(self, text: str) -> Document:
        """Parse raw text into a Document.

        Since LiteParse operates on files, this wraps the text as-is.

        Args:
            text: Raw text to parse.

        Returns:
            Document created from the text.

        """
        return Document(content=text)

    def __repr__(self) -> str:
        """Return a string representation of the LiteParse instance."""
        return f"LiteParse(ocr_enabled={self.ocr_enabled}, ocr_language={self.ocr_language!r})"
