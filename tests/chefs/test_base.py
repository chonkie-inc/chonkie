"""Tests for Chonkie Chef base classes."""

import os
import pytest
from pathlib import Path

from chonkie.chefs import (
    BaseChef,
    PDFProcessingChef,
    ProcessingResult,
    ProcessingStatus,
    ChefConfig,
    ValidationError,
)
from chonkie.types import Document, PDFDocument


class TestChef(PDFProcessingChef):
    """Test implementation of PDFProcessingChef."""
    
    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Test implementation of process."""
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            document=PDFDocument(text="Test document"),
            metadata={"test": True}
        )


def test_chef_initialization():
    """Test Chef initialization."""
    config = ChefConfig(ocr_enabled=False)
    chef = TestChef("test", "1.0.0", config)
    
    assert chef.name == "test"
    assert chef.version == "1.0.0"
    assert not chef.config.ocr_enabled
    assert chef.supported_formats == ["pdf"]


def test_chef_metadata():
    """Test Chef metadata retrieval."""
    chef = TestChef("test", "1.0.0")
    metadata = chef.get_metadata()
    
    assert metadata["name"] == "test"
    assert metadata["version"] == "1.0.0"
    assert "config" in metadata


def test_file_validation():
    """Test file validation."""
    chef = TestChef("test", "1.0.0")
    
    # Test non-existent file
    with pytest.raises(ValidationError):
        chef.validate_file("nonexistent.pdf")
    
    # Test non-PDF file
    with pytest.raises(ValidationError):
        chef.validate_file("test.txt")
    
    # Create a test PDF file
    test_pdf = Path("test.pdf")
    test_pdf.touch()
    
    try:
        assert chef.validate_file("test.pdf")
    finally:
        test_pdf.unlink()


def test_processing_result():
    """Test ProcessingResult creation and access."""
    doc = PDFDocument(text="Test")
    result = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        document=doc,
        metadata={"test": True}
    )
    
    assert result.status == ProcessingStatus.SUCCESS
    assert result.document == doc
    assert result.metadata["test"]
    assert result.error is None


def test_chef_config():
    """Test ChefConfig functionality."""
    config = ChefConfig()
    
    # Test default values
    assert config.ocr_enabled
    assert config.extract_images
    assert config.language == "eng"
    
    # Test update
    config.update(ocr_enabled=False, custom_setting="test")
    assert not config.ocr_enabled
    assert config.additional_settings["custom_setting"] == "test"
    
    # Test to_dict
    config_dict = config.to_dict()
    assert "ocr_enabled" in config_dict
    assert "custom_setting" in config_dict 