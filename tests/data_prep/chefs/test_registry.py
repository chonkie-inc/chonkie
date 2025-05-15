"""Tests for the chef registry functionality."""

import pytest
from pathlib import Path
from chonkie.chefs.registry import ChefRegistry
from chonkie.chefs.base import BaseChef, ChefError
from chonkie.chefs.pdf.extractor import PDFExtractorChef, PDFExtractorConfig
from chonkie.chefs.docx.extractor import DOCXExtractorChef, DOCXExtractorConfig

class MockChef(BaseChef):
    """Mock chef for testing."""
    def validate(self, input_data):
        return True
    
    def prepare(self, input_data):
        return {"text": "mock text"}
    
    def clean(self):
        pass

def test_registry_initialization():
    """Test registry initialization with default chefs."""
    registry = ChefRegistry()
    
    # Check if default chefs are registered
    assert ".pdf" in registry.get_supported_extensions()
    assert ".docx" in registry.get_supported_extensions()
    
    # Check if correct chef classes are registered
    assert isinstance(registry.get_chef("test.pdf"), PDFExtractorChef)
    assert isinstance(registry.get_chef("test.docx"), DOCXExtractorChef)

def test_register_chef():
    """Test registering a new chef."""
    registry = ChefRegistry()
    
    # Register a new chef
    registry.register_chef(".txt", MockChef)
    assert ".txt" in registry.get_supported_extensions()
    assert isinstance(registry.get_chef("test.txt"), MockChef)

def test_register_chef_with_config():
    """Test registering a chef with config class."""
    registry = ChefRegistry()
    
    # Register a new chef with config
    registry.register_chef(".txt", MockChef, PDFExtractorConfig)
    assert ".txt" in registry.get_supported_extensions()
    chef = registry.get_chef("test.txt", extract_metadata=False)
    assert isinstance(chef, MockChef)

def test_register_chef_invalid_extension():
    """Test registering a chef with invalid extension."""
    registry = ChefRegistry()
    
    with pytest.raises(ChefError):
        registry.register_chef("pdf", MockChef)  # Missing dot

def test_register_chef_invalid_class():
    """Test registering an invalid chef class."""
    registry = ChefRegistry()
    
    class InvalidChef:
        pass
    
    with pytest.raises(ChefError):
        registry.register_chef(".txt", InvalidChef)

def test_get_chef_unsupported_type():
    """Test getting a chef for an unsupported file type."""
    registry = ChefRegistry()
    
    with pytest.raises(ChefError):
        registry.get_chef("test.txt")

def test_get_chef_with_kwargs():
    """Test getting a chef with additional arguments."""
    registry = ChefRegistry()
    
    # Test with PDFExtractorChef config
    chef = registry.get_chef("test.pdf", extract_metadata=False)
    assert isinstance(chef, PDFExtractorChef)
    assert isinstance(chef.config, PDFExtractorConfig)
    assert chef.config.extract_metadata is False

def test_is_supported():
    """Test checking if a file type is supported."""
    registry = ChefRegistry()
    
    assert registry.is_supported("test.pdf")
    assert registry.is_supported("test.docx")
    assert not registry.is_supported("test.txt")
    assert not registry.is_supported("test")

def test_get_supported_extensions():
    """Test getting list of supported extensions."""
    registry = ChefRegistry()
    extensions = registry.get_supported_extensions()
    
    assert isinstance(extensions, list)
    assert ".pdf" in extensions
    assert ".docx" in extensions
    assert len(extensions) == 2  # Only PDF and DOCX by default

def test_path_object_support():
    """Test support for Path objects."""
    registry = ChefRegistry()
    
    # Test with Path objects
    assert registry.is_supported(Path("test.pdf"))
    assert isinstance(registry.get_chef(Path("test.pdf")), PDFExtractorChef)
    
    # Test with Path objects for unsupported types
    assert not registry.is_supported(Path("test.txt"))
    with pytest.raises(ChefError):
        registry.get_chef(Path("test.txt"))

def test_backward_compatibility():
    """Test backward compatibility methods."""
    registry = ChefRegistry()
    
    # Test register
    registry.register("mock", MockChef)
    assert ".mock" in registry.get_supported_extensions()
    assert isinstance(registry.get_chef("test.mock"), MockChef)
    
    # Test unregister
    registry.unregister("mock")
    assert ".mock" not in registry.get_supported_extensions()
    
    # Test list_chefs
    chefs = registry.list_chefs()
    assert isinstance(chefs, dict)
    assert "pdf" in chefs
    assert "docx" in chefs

def test_global_registry():
    """Test the global registry instance."""
    from chonkie.chefs.registry import registry
    
    # Test default chefs
    assert ".pdf" in registry.get_supported_extensions()
    assert ".docx" in registry.get_supported_extensions()
    
    # Test getting chefs
    assert isinstance(registry.get_chef("test.pdf"), PDFExtractorChef)
    assert isinstance(registry.get_chef("test.docx"), DOCXExtractorChef) 