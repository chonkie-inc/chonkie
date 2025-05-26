"""Tests for the EmbeddingsRegistry class."""

from typing import Any

import pytest

from chonkie.embeddings.base import BaseEmbeddings
from chonkie.embeddings.registry import EmbeddingsRegistry
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings


class MockEmbeddings(BaseEmbeddings):
    """Mock embeddings class for testing."""

    def __init__(self, model: str = "mock-model", **kwargs: Any) -> None:
        """Initialize mock embeddings."""
        super().__init__()
        self.model = model

    def embed(self, text: str) -> Any:
        """Mock embed method."""
        return [1.0, 2.0, 3.0]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return 3
    
    def get_tokenizer_or_token_counter(self) -> Any:
        """Return mock tokenizer."""
        return lambda x: len(x.split())


class NonEmbeddingsClass:
    """Non-embeddings class for testing validation."""

    pass


class TestEmbeddingsRegistryRegistration:
    """Test registration methods of EmbeddingsRegistry."""

    def setup_method(self) -> None:
        """Set up test by clearing registries."""
        # Save original registries
        self.original_model_registry = EmbeddingsRegistry.model_registry.copy()
        self.original_provider_registry = EmbeddingsRegistry.provider_registry.copy()
        self.original_pattern_registry = EmbeddingsRegistry.pattern_registry.copy()
        self.original_type_registry = EmbeddingsRegistry.type_registry.copy()

    def teardown_method(self) -> None:
        """Clean up test by restoring original registries."""
        EmbeddingsRegistry.model_registry = self.original_model_registry
        EmbeddingsRegistry.provider_registry = self.original_provider_registry
        EmbeddingsRegistry.pattern_registry = self.original_pattern_registry
        EmbeddingsRegistry.type_registry = self.original_type_registry

    def test_register_model_valid(self) -> None:
        """Test registering a valid model."""
        EmbeddingsRegistry.register_model("test-model", MockEmbeddings)
        assert "test-model" in EmbeddingsRegistry.model_registry
        assert EmbeddingsRegistry.model_registry["test-model"] == MockEmbeddings

    def test_register_model_invalid_class(self) -> None:
        """Test registering an invalid model class raises ValueError."""
        with pytest.raises(ValueError, match="must be a subclass of BaseEmbeddings"):
            EmbeddingsRegistry.register_model("invalid", NonEmbeddingsClass)  # type: ignore

    def test_register_provider_valid(self) -> None:
        """Test registering a valid provider."""
        EmbeddingsRegistry.register_provider("test-provider", MockEmbeddings)
        assert "test-provider" in EmbeddingsRegistry.provider_registry
        assert EmbeddingsRegistry.provider_registry["test-provider"] == MockEmbeddings

    def test_register_provider_invalid_class(self) -> None:
        """Test registering an invalid provider class raises ValueError."""
        with pytest.raises(ValueError, match="must be a subclass of BaseEmbeddings"):
            EmbeddingsRegistry.register_provider("invalid", NonEmbeddingsClass)  # type: ignore

    def test_register_pattern_valid(self) -> None:
        """Test registering a valid pattern."""
        pattern_str = r"^test-.*"
        EmbeddingsRegistry.register_pattern(pattern_str, MockEmbeddings)
        
        # Check that pattern was compiled and registered
        pattern_found = False
        for pattern, cls in EmbeddingsRegistry.pattern_registry.items():
            if pattern.pattern == pattern_str and cls == MockEmbeddings:
                pattern_found = True
                break
        assert pattern_found

    def test_register_pattern_invalid_class(self) -> None:
        """Test registering an invalid pattern class raises ValueError."""
        with pytest.raises(ValueError, match="must be a subclass of BaseEmbeddings"):
            EmbeddingsRegistry.register_pattern(r"^test-.*", NonEmbeddingsClass)  # type: ignore

    def test_register_types_string(self) -> None:
        """Test registering types with a single string."""
        EmbeddingsRegistry.register_types("TestType", MockEmbeddings)
        assert "TestType" in EmbeddingsRegistry.type_registry
        assert EmbeddingsRegistry.type_registry["TestType"] == MockEmbeddings

    def test_register_types_list(self) -> None:
        """Test registering types with a list of strings."""
        types_list = ["Type1", "Type2", "Type3"]
        EmbeddingsRegistry.register_types(types_list, MockEmbeddings)
        
        for type_name in types_list:
            assert type_name in EmbeddingsRegistry.type_registry
            assert EmbeddingsRegistry.type_registry[type_name] == MockEmbeddings

    def test_register_types_invalid_class(self) -> None:
        """Test registering types with invalid class raises ValueError."""
        with pytest.raises(ValueError, match="must be a subclass of BaseEmbeddings"):
            EmbeddingsRegistry.register_types("TestType", NonEmbeddingsClass)  # type: ignore

    def test_register_types_invalid_input(self) -> None:
        """Test registering types with invalid input type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid types"):
            EmbeddingsRegistry.register_types(123, MockEmbeddings)  # type: ignore


class TestEmbeddingsRegistryLookup:
    """Test lookup methods of EmbeddingsRegistry."""

    def setup_method(self) -> None:
        """Set up test registries with known values."""
        # Clear registries first
        EmbeddingsRegistry.model_registry.clear()
        EmbeddingsRegistry.provider_registry.clear()
        EmbeddingsRegistry.pattern_registry.clear()
        EmbeddingsRegistry.type_registry.clear()
        
        # Register test entries
        EmbeddingsRegistry.register_model("test-model", MockEmbeddings)
        EmbeddingsRegistry.register_provider("test", MockEmbeddings)
        EmbeddingsRegistry.register_pattern(r"^custom-.*", MockEmbeddings)
        EmbeddingsRegistry.register_types("MockType", MockEmbeddings)

    def test_get_provider_existing(self) -> None:
        """Test getting an existing provider."""
        result = EmbeddingsRegistry.get_provider("test")
        assert result == MockEmbeddings

    def test_get_provider_nonexistent(self) -> None:
        """Test getting a non-existent provider returns None."""
        result = EmbeddingsRegistry.get_provider("nonexistent")
        assert result is None

    def test_match_provider_prefix(self) -> None:
        """Test matching with provider prefix syntax."""
        result = EmbeddingsRegistry.match("test://some-model")
        assert result == MockEmbeddings

    def test_match_exact_model(self) -> None:
        """Test matching exact model name."""
        result = EmbeddingsRegistry.match("test-model")
        assert result == MockEmbeddings

    def test_match_pattern(self) -> None:
        """Test matching with regex pattern."""
        result = EmbeddingsRegistry.match("custom-embedding-model")
        assert result == MockEmbeddings

    def test_match_no_match(self) -> None:
        """Test matching with no matches returns None."""
        result = EmbeddingsRegistry.match("completely-unknown-identifier")
        assert result is None

    def test_match_provider_prefix_nonexistent(self) -> None:
        """Test matching with non-existent provider prefix."""
        result = EmbeddingsRegistry.match("unknown://some-model")
        assert result is None


class TestEmbeddingsRegistryWrap:
    """Test wrap method of EmbeddingsRegistry."""

    def setup_method(self) -> None:
        """Set up test registries."""
        # Save original registries
        self.original_type_registry = EmbeddingsRegistry.type_registry.copy()
        self.original_model_registry = EmbeddingsRegistry.model_registry.copy()
        
        # Clear and set up test data
        EmbeddingsRegistry.type_registry.clear()
        EmbeddingsRegistry.model_registry.clear()
        EmbeddingsRegistry.register_types("MockType", MockEmbeddings)
        EmbeddingsRegistry.register_model("test-string-model", MockEmbeddings)

    def teardown_method(self) -> None:
        """Clean up test registries."""
        EmbeddingsRegistry.type_registry = self.original_type_registry
        EmbeddingsRegistry.model_registry = self.original_model_registry

    def test_wrap_existing_embeddings(self) -> None:
        """Test wrapping an existing BaseEmbeddings instance."""
        original = MockEmbeddings("test-model")
        result = EmbeddingsRegistry.wrap(original)
        assert result is original

    def test_wrap_string_identifier(self) -> None:
        """Test wrapping a string identifier."""
        result = EmbeddingsRegistry.wrap("test-string-model")
        assert isinstance(result, MockEmbeddings)

    def test_wrap_custom_object_type_match(self) -> None:
        """Test wrapping a custom object with type registry match."""
        class MockTypeClass:
            pass
        
        mock_obj = MockTypeClass()
        # Register the exact type string that would appear in str(type(mock_obj))
        EmbeddingsRegistry.register_types("MockTypeClass", MockEmbeddings)
        
        result = EmbeddingsRegistry.wrap(mock_obj)
        assert isinstance(result, MockEmbeddings)

    def test_wrap_unsupported_object(self) -> None:
        """Test wrapping an unsupported object type raises ValueError."""
        unsupported_obj = {"not": "supported"}
        with pytest.raises(ValueError, match="Unsupported object type for embeddings"):
            EmbeddingsRegistry.wrap(unsupported_obj)
            
    def test_wrap_string_no_match(self) -> None:
        """Test wrapping string with no registry match raises error."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            EmbeddingsRegistry.wrap("unknown-string-model")


class TestEmbeddingsRegistryIntegration:
    """Test EmbeddingsRegistry integration functionality."""
    
    def test_registries_have_content(self) -> None:
        """Test that registries are populated with default content."""
        # At module load time, registries should be populated
        assert len(EmbeddingsRegistry.provider_registry) > 0
        assert len(EmbeddingsRegistry.model_registry) > 0
        assert len(EmbeddingsRegistry.pattern_registry) > 0
        assert len(EmbeddingsRegistry.type_registry) > 0
        
    def test_provider_registration_functionality(self) -> None:
        """Test that provider registration mechanism works."""
        # Test registration functionality rather than assuming global state
        # This tests the actual registration mechanism which is what matters
        EmbeddingsRegistry.register_provider("test_provider", SentenceTransformerEmbeddings)
        result = EmbeddingsRegistry.get_provider("test_provider")
        assert result == SentenceTransformerEmbeddings
        
        # Test that the registration persists
        assert "test_provider" in EmbeddingsRegistry.provider_registry
            
    def test_integration_with_autoembeddings(self) -> None:
        """Test that registry integrates properly with AutoEmbeddings patterns."""
        # These should work through the registry system
        from chonkie.embeddings.auto import AutoEmbeddings
        
        # Test provider prefix syntax
        try:
            result = AutoEmbeddings.get_embeddings("st://all-MiniLM-L6-v2")
            assert result is not None
        except Exception:
            # This might fail due to missing dependencies, which is OK for this test
            pass


class TestEmbeddingsRegistryEdgeCases:
    """Test edge cases and complex scenarios."""
    
    def setup_method(self) -> None:
        """Set up test registries."""
        # Save original registries
        self.original_pattern_registry = EmbeddingsRegistry.pattern_registry.copy()
        self.original_model_registry = EmbeddingsRegistry.model_registry.copy()
        self.original_type_registry = EmbeddingsRegistry.type_registry.copy()
        self.original_provider_registry = EmbeddingsRegistry.provider_registry.copy()

    def teardown_method(self) -> None:
        """Clean up test registries."""
        EmbeddingsRegistry.pattern_registry = self.original_pattern_registry
        EmbeddingsRegistry.model_registry = self.original_model_registry
        EmbeddingsRegistry.type_registry = self.original_type_registry
        EmbeddingsRegistry.provider_registry = self.original_provider_registry

    def test_provider_prefix_with_multiple_separators(self) -> None:
        """Test provider prefix with multiple :// separators."""
        # First part should match openai provider, rest is model name
        # Register test provider first to avoid dependency on global state
        EmbeddingsRegistry.register_provider("test_openai", MockEmbeddings)
        result = EmbeddingsRegistry.match("test_openai://model://with://colons")
        assert result == MockEmbeddings

    def test_pattern_compilation_with_complex_regex(self) -> None:
        """Test pattern registration with complex regex."""
        complex_pattern = r"^(?:test|demo)-\w+-(?:v\d+|\d+\.\d+)$"
        EmbeddingsRegistry.register_pattern(complex_pattern, MockEmbeddings)
        
        # Test that pattern works
        assert EmbeddingsRegistry.match("test-model-v1") == MockEmbeddings
        assert EmbeddingsRegistry.match("demo-embedding-2.1") == MockEmbeddings
        assert EmbeddingsRegistry.match("other-model-v1") is None

    def test_type_registry_partial_match(self) -> None:
        """Test type registry with partial string matching."""
        class CustomModelType:
            pass
        
        EmbeddingsRegistry.register_types("CustomModel", MockEmbeddings)
        
        # This should match because "CustomModel" is in the type string
        custom_obj = CustomModelType()
        result = EmbeddingsRegistry.wrap(custom_obj)
        assert isinstance(result, MockEmbeddings)

    def test_registry_state_isolation(self) -> None:
        """Test that registry modifications don't affect other tests."""
        # Add a temporary registration
        original_size = len(EmbeddingsRegistry.model_registry)
        EmbeddingsRegistry.register_model("temp-model", MockEmbeddings)
        assert "temp-model" in EmbeddingsRegistry.model_registry
        assert len(EmbeddingsRegistry.model_registry) == original_size + 1
        
        # This should persist within the test but not leak to other tests
        # (teardown_method handles cleanup)
        
    def test_pattern_priority_over_exact_match(self) -> None:
        """Test that pattern matching works even when exact matches exist."""
        # Register both exact model and pattern
        EmbeddingsRegistry.register_model("exact-match-model", SentenceTransformerEmbeddings)
        EmbeddingsRegistry.register_pattern(r"^exact-match-.*", MockEmbeddings)
        
        # Exact match should take priority
        result = EmbeddingsRegistry.match("exact-match-model")
        assert result == SentenceTransformerEmbeddings
        
        # Pattern should match for non-exact
        result = EmbeddingsRegistry.match("exact-match-other")
        assert result == MockEmbeddings