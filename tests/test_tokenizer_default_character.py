"""Tests for the default character tokenizer and optional dependency handling.

This module tests the RFC changes that make character-based chunking the default
and ensure proper error handling for optional tokenizer dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock

from chonkie import Tokenizer, CharacterTokenizer, WordTokenizer


class TestDefaultCharacterTokenizer:
    """Test that character tokenizer is the default and works correctly."""

    def test_default_tokenizer_is_character(self):
        """Test that the default tokenizer is character-based."""
        tokenizer = Tokenizer()
        assert isinstance(tokenizer.tokenizer, CharacterTokenizer)
        assert tokenizer._backend == "chonkie"

    def test_character_tokenizer_explicit(self):
        """Test that character tokenizer works when explicitly specified."""
        tokenizer = Tokenizer("character")
        assert isinstance(tokenizer.tokenizer, CharacterTokenizer)
        assert tokenizer._backend == "chonkie"

    def test_word_tokenizer_works(self):
        """Test that word tokenizer works correctly."""
        tokenizer = Tokenizer("word")
        assert isinstance(tokenizer.tokenizer, WordTokenizer)
        assert tokenizer._backend == "chonkie"

    def test_character_tokenizer_functionality(self):
        """Test that character tokenizer works correctly."""
        tokenizer = Tokenizer("character")
        text = "Hello, world!"
        
        # Test encoding
        encoded = tokenizer.encode(text)
        assert len(encoded) == len(text)
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        
        # Test token counting
        count = tokenizer.count_tokens(text)
        assert count == len(text)

    def test_word_tokenizer_functionality(self):
        """Test that word tokenizer works correctly."""
        tokenizer = Tokenizer("word")
        text = "Hello world"
        
        # Test encoding
        encoded = tokenizer.encode(text)
        assert len(encoded) == 2  # "Hello" and "world"
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        
        # Test token counting
        count = tokenizer.count_tokens(text)
        assert count == 2


class TestOptionalDependencies:
    """Test optional dependency handling for tokenizers."""

    @patch('importlib.util.find_spec')
    def test_tokenizers_not_installed(self, mock_find_spec):
        """Test that appropriate error is raised when tokenizers is not installed."""
        # Mock that tokenizers is not available
        mock_find_spec.return_value = None
        
        with pytest.raises(ImportError) as exc_info:
            Tokenizer("gpt2")
        
        assert "transformers" in str(exc_info.value)
        assert "pip install chonkie[transformers]" in str(exc_info.value)

    @patch('importlib.util.find_spec')
    def test_tiktoken_not_installed(self, mock_find_spec):
        """Test that appropriate error is raised when tiktoken is not installed."""
        # Mock that tokenizers is not available but tiktoken is requested
        def mock_find_spec_side_effect(name):
            if name == "tokenizers":
                return None
            elif name == "tiktoken":
                return None
            return MagicMock()
        
        mock_find_spec.side_effect = mock_find_spec_side_effect
        
        with pytest.raises(ImportError) as exc_info:
            Tokenizer("cl100k_base")
        
        assert "tiktoken" in str(exc_info.value)
        assert "pip install chonkie[tiktoken]" in str(exc_info.value)

    @patch('importlib.util.find_spec')
    def test_transformers_not_installed(self, mock_find_spec):
        """Test that appropriate error is raised when transformers is not installed."""
        # Mock that tokenizers and tiktoken are not available but transformers is requested
        def mock_find_spec_side_effect(name):
            if name in ["tokenizers", "tiktoken", "transformers"]:
                return None
            return MagicMock()
        
        mock_find_spec.side_effect = mock_find_spec_side_effect
        
        with pytest.raises(ImportError) as exc_info:
            Tokenizer("bert-base-uncased")
        
        assert "transformers" in str(exc_info.value)
        assert "pip install chonkie[transformers]" in str(exc_info.value)

    @patch('importlib.util.find_spec')
    @patch('tokenizers.Tokenizer.from_pretrained')
    def test_tokenizers_installed_but_fails(self, mock_from_pretrained, mock_find_spec):
        """Test that appropriate error is raised when tokenizers is installed but fails to load."""
        # Mock that tokenizers is available
        mock_find_spec.return_value = MagicMock()
        mock_from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(ImportError) as exc_info:
            Tokenizer("nonexistent-model")
        
        assert "Failed to load tokenizer" in str(exc_info.value)
        assert "tokenizers library" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""

    def test_custom_callable_tokenizer(self):
        """Test that custom callable tokenizers still work."""
        def custom_token_counter(text: str) -> int:
            return len(text.split())
        
        tokenizer = Tokenizer(custom_token_counter)
        assert tokenizer._backend == "callable"
        
        # Test token counting
        count = tokenizer.count_tokens("Hello world")
        assert count == 2

    def test_batch_operations_character(self):
        """Test batch operations with character tokenizer."""
        tokenizer = Tokenizer("character")
        texts = ["Hello", "World", "Test"]
        
        # Test batch encoding
        encoded_batch = tokenizer.encode_batch(texts)
        assert len(encoded_batch) == 3
        assert len(encoded_batch[0]) == 5  # "Hello"
        assert len(encoded_batch[1]) == 5  # "World"
        assert len(encoded_batch[2]) == 4  # "Test"
        
        # Test batch decoding
        decoded_batch = tokenizer.decode_batch(encoded_batch)
        assert decoded_batch == texts
        
        # Test batch token counting
        counts = tokenizer.count_tokens_batch(texts)
        assert counts == [5, 5, 4]


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_invalid_tokenizer_name(self):
        """Test error message for invalid tokenizer name."""
        with pytest.raises(ImportError) as exc_info:
            Tokenizer("invalid-tokenizer-name")
        
        error_msg = str(exc_info.value)
        assert "invalid-tokenizer-name" in error_msg
        assert "Failed to load tokenizer" in error_msg

    def test_unsupported_backend(self):
        """Test error message for unsupported backend."""
        tokenizer = Tokenizer("character")
        # Mock an unsupported backend
        tokenizer._backend = "unsupported"
        
        with pytest.raises(ValueError) as exc_info:
            tokenizer.encode("test")
        
        assert "Unsupported tokenizer backend" in str(exc_info.value) 