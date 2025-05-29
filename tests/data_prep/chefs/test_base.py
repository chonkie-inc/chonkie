"""Tests for the base chef functionality."""

import pytest
from typing import Dict, Any

from chonkie.chefs import BaseChef, ChefConfig, ChefError

class MockChef(BaseChef[str, Dict[str, Any]]):
    """Mock chef for testing base functionality."""
    
    def validate(self, data: str) -> bool:
        return data == "valid"
    
    def prepare(self, data: str) -> Dict[str, Any]:
        return {"text": data, "processed": True}
    
    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["cleaned"] = True
        return data

def test_chef_config():
    """Test chef configuration."""
    config = ChefConfig(
        name="test_chef",
        description="Test chef",
        enabled=True,
        config={"key": "value"}
    )
    
    assert config.name == "test_chef"
    assert config.description == "Test chef"
    assert config.enabled is True
    assert config.config == {"key": "value"}

def test_base_chef_initialization():
    """Test base chef initialization."""
    # Test with explicit config
    config = ChefConfig(name="test_chef")
    chef = MockChef(config)
    assert chef.config.name == "test_chef"
    
    # Test with default config
    chef = MockChef()
    assert chef.config.name == "MockChef"

def test_base_chef_validation():
    """Test base chef validation."""
    chef = MockChef()
    
    # Test valid data
    assert chef.validate("valid") is True
    
    # Test invalid data
    assert chef.validate("invalid") is False

def test_base_chef_pipeline():
    """Test the full chef pipeline."""
    chef = MockChef()
    
    # Test successful pipeline
    result = chef("valid")
    assert result["text"] == "valid"
    assert result["processed"] is True
    assert result["cleaned"] is True
    
    # Test pipeline with invalid data
    with pytest.raises(ValueError):
        chef("invalid")

def test_chef_error():
    """Test chef error handling."""
    with pytest.raises(ChefError):
        raise ChefError("Test error") 