"""Tests for the BaseChef abstract class."""

from __future__ import annotations

import pytest

from chonkie.chef import BaseChef
from chonkie.types import Document


class ConcreteChef(BaseChef):
    """Concrete implementation of BaseChef for testing."""

    def process(self, path: str) -> Document:
        """Test implementation that returns a Document."""
        return Document(content=f"processed: {path}")

    def parse(self, text: str) -> Document:
        """Test implementation that parses text."""
        return Document(content=f"parsed: {text}")


class TestBaseChef:
    """Test cases for BaseChef abstract class."""

    def test_cannot_instantiate_abstract_class(self: "TestBaseChef") -> None:
        """Test that BaseChef cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChef()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self: "TestBaseChef") -> None:
        """Test that concrete subclass can be instantiated."""
        chef = ConcreteChef()
        assert isinstance(chef, BaseChef)

    def test_call_delegates_to_process(self: "TestBaseChef") -> None:
        """Test that __call__ method delegates to process method."""
        chef = ConcreteChef()
        result = chef("test_path")
        assert isinstance(result, Document)
        assert result.content == "processed: test_path"

    def test_repr_method(self: "TestBaseChef") -> None:
        """Test __repr__ method returns correct string."""
        chef = ConcreteChef()
        assert repr(chef) == "ConcreteChef()"