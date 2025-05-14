"""Test fixtures and utilities for Chonkie Chefs."""

import os
import pytest
from pathlib import Path
from typing import Generator

from chonkie.chefs import ChefConfig
from chonkie.types import PDFDocument


@pytest.fixture
def test_config() -> ChefConfig:
    """Create a test configuration."""
    return ChefConfig(
        ocr_enabled=True,
        extract_images=True,
        extract_tables=True,
        language="eng",
        dpi=300
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a sample PDF file for testing.
    
    This is a placeholder fixture. In a real implementation, you would
    create an actual PDF file with known content for testing.
    """
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    yield pdf_path
    pdf_path.unlink()


@pytest.fixture
def sample_pdf_document() -> PDFDocument:
    """Create a sample PDF document for testing."""
    return PDFDocument(
        text="Test document",
        metadata={"test": True}
    ) 