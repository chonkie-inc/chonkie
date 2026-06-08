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
        ocr_server_url: Optional[str] = None,
        ocr_language: Optional[str] = None,
        tessdata_path: Optional[str] = None,
        max_pages: Optional[int] = None,
        target_pages: Optional[str] = None,
        dpi: Optional[float] = None,
        output_format: Optional[str] = None,
        preserve_very_small_text: Optional[bool] = None,
        password: Optional[str] = None,
        quiet: Optional[bool] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize the LiteParse chef.

        Args:
            ocr_enabled: Whether to enable OCR for scanned documents (default: True)
            ocr_server_url: URL of HTTP OCR server (uses Tesseract if not provided)
            ocr_language: Language code for OCR (e.g., "eng", "fra")
            tessdata_path: Path to tessdata directory for Tesseract
            max_pages: Maximum number of pages to parse
            target_pages: Specific pages to parse (e.g., "1-5,10,15-20")
            dpi: DPI for rendering (affects OCR quality)
            output_format: Output format: "json" or "text" (default: "json")
            preserve_very_small_text: Whether to preserve very small text
            password: Password for encrypted/protected documents
            quiet: Suppress progress output
            num_workers: Number of concurrent OCR workers (default: CPU cores - 1)

        """
        try:
            from liteparse import LiteParse as _LiteParse
        except ImportError as ie:
            raise ImportError(
                "The required module is not available: [liteparse]. "
                "Please install the dependency via `pip install chonkie[liteparse]`"
            ) from ie

        if ocr_enabled and not (tessdata_path or os.environ.get("TESSDATA_PREFIX")):
            self._auto_detect_tessdata()

        self.parser = _LiteParse(
            ocr_enabled=ocr_enabled,
            ocr_server_url=ocr_server_url,
            ocr_language=ocr_language,
            tessdata_path=tessdata_path,
            max_pages=max_pages,
            target_pages=target_pages,
            dpi=dpi,
            output_format=output_format,
            preserve_very_small_text=preserve_very_small_text,
            password=password,
            quiet=quiet,
            num_workers=num_workers,
        )
        self.ocr_enabled = ocr_enabled
        self.ocr_server_url = ocr_server_url
        self.ocr_language = ocr_language
        self.tessdata_path = tessdata_path
        self.max_pages = max_pages
        self.target_pages = target_pages
        self.dpi = dpi
        self.output_format = output_format
        self.preserve_very_small_text = preserve_very_small_text
        self.password = password
        self.quiet = quiet
        self.num_workers = num_workers

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
        content = result.text if result and result.text is not None else ""

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
        return (
            f"LiteParse("
            f"ocr_enabled={self.ocr_enabled}, "
            f"ocr_server_url={self.ocr_server_url!r}, "
            f"ocr_language={self.ocr_language!r}, "
            f"tessdata_path={self.tessdata_path!r}, "
            f"max_pages={self.max_pages}, "
            f"target_pages={self.target_pages!r}, "
            f"dpi={self.dpi}, "
            f"output_format={self.output_format!r}, "
            f"preserve_very_small_text={self.preserve_very_small_text}, "
            f"quiet={self.quiet}, "
            f"num_workers={self.num_workers})"
        )
