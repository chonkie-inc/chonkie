"""Tests for OpenAIGenie class."""

import os
from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie, OpenAIGenie


class TestOpenAIGenieImportAndConstruction:
    """Test OpenAIGenie import and basic construction."""
    
    def test_openai_genie_import(self) -> None:
        """Test that OpenAIGenie can be imported."""
        assert OpenAIGenie is not None
        assert issubclass(OpenAIGenie, BaseGenie)
    
    def test_openai_genie_has_required_methods(self) -> None:
        """Test that OpenAIGenie has all required methods."""
        assert hasattr(OpenAIGenie, 'generate')
        assert hasattr(OpenAIGenie, 'generate_batch')
        assert hasattr(OpenAIGenie, 'generate_json')
        assert hasattr(OpenAIGenie, 'generate_json_batch')
        assert hasattr(OpenAIGenie, '_is_available')


class TestOpenAIGenieErrorHandling:
    """Test OpenAIGenie error handling."""
    
    def test_openai_genie_missing_api_key(self) -> None:
        """Test OpenAIGenie raises error without API key."""
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="OpenAIGenie requires an API key"):
                    OpenAIGenie()
    
    def test_openai_genie_missing_dependencies(self) -> None:
        """Test OpenAIGenie raises error without dependencies."""
        with patch.object(OpenAIGenie, '_is_available', return_value=False):
            with pytest.raises(ImportError, match="One or more of the required modules are not available"):
                OpenAIGenie(api_key="test")


class TestOpenAIGenieBasicFunctionality:
    """Test OpenAIGenie basic functionality with mocking."""
    
    def test_openai_genie_initialization(self) -> None:
        """Test OpenAIGenie can be initialized with mocked dependencies."""
        mock_client = Mock()
        mock_openai_class = Mock(return_value=mock_client)
        
        def mock_import_dependencies(self):
            import chonkie.genie.openai as openai_module
            openai_module.OpenAI = mock_openai_class
            openai_module.BaseModel = Mock()
        
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie()
                    
                assert genie is not None
                assert isinstance(genie, BaseGenie)
                mock_openai_class.assert_called_once_with(api_key='test_key')
    
    def test_openai_genie_generate_text(self) -> None:
        """Test OpenAIGenie text generation with mocked response."""
        # Mock the response structure
        mock_message = Mock()
        mock_message.content = "Generated response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        # Mock the client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)
        
        def mock_import_dependencies(self):
            import chonkie.genie.openai as openai_module
            openai_module.OpenAI = mock_openai_class
            openai_module.BaseModel = Mock()
        
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie()
                    result = genie.generate("Test prompt")
                
                assert result == "Generated response"
                mock_client.chat.completions.create.assert_called_once()
    
    def test_openai_genie_batch_generation(self) -> None:
        """Test OpenAIGenie batch generation."""
        # Mock multiple responses
        mock_responses = []
        for i in range(3):
            mock_message = Mock()
            mock_message.content = f"Response {i}"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_responses.append(mock_response)
        
        # Mock the client
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class = Mock(return_value=mock_client)
        
        def mock_import_dependencies(self):
            import chonkie.genie.openai as openai_module
            openai_module.OpenAI = mock_openai_class
            openai_module.BaseModel = Mock()
        
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie()
                    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                    results = genie.generate_batch(prompts)
                
                assert len(results) == 3
                assert results == ["Response 0", "Response 1", "Response 2"]
                assert mock_client.chat.completions.create.call_count == 3


class TestOpenAIGenieUtilities:
    """Test OpenAIGenie utility methods."""
    
    def test_openai_genie_is_available_true(self) -> None:
        """Test _is_available returns True when dependencies are installed."""
        with patch('chonkie.genie.openai.importutil.find_spec') as mock_find_spec:
            mock_find_spec.side_effect = lambda x: Mock() if x in ["openai", "pydantic"] else None
            
            def mock_import_dependencies(self):
                import chonkie.genie.openai as openai_module
                openai_module.OpenAI = Mock()
                openai_module.BaseModel = Mock()
            
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie()
                    result = genie._is_available()
                    assert result is True
    
    def test_openai_genie_is_available_false(self) -> None:
        """Test _is_available returns False when dependencies are missing."""
        with patch('chonkie.genie.openai.importutil.find_spec') as mock_find_spec:
            mock_find_spec.return_value = None
            
            def mock_import_dependencies(self):
                import chonkie.genie.openai as openai_module
                openai_module.OpenAI = Mock()
                openai_module.BaseModel = Mock()
            
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie()
                    result = genie._is_available()
                    assert result is False
    
    def test_openai_genie_repr(self) -> None:
        """Test OpenAIGenie string representation."""
        mock_client = Mock()
        mock_openai_class = Mock(return_value=mock_client)
        
        def mock_import_dependencies(self):
            import chonkie.genie.openai as openai_module
            openai_module.OpenAI = mock_openai_class
            openai_module.BaseModel = Mock()
        
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie(model="gpt-4")
                    repr_str = repr(genie)
                
                assert "OpenAIGenie" in repr_str
                assert "gpt-4" in repr_str
    
    def test_openai_genie_custom_base_url(self) -> None:
        """Test OpenAIGenie with custom base URL."""
        mock_client = Mock()
        mock_openai_class = Mock(return_value=mock_client)
        
        def mock_import_dependencies(self):
            import chonkie.genie.openai as openai_module
            openai_module.OpenAI = mock_openai_class
            openai_module.BaseModel = Mock()
        
        with patch.object(OpenAIGenie, '_is_available', return_value=True):
            with patch.object(OpenAIGenie, '_import_dependencies', mock_import_dependencies):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
                    genie = OpenAIGenie(base_url="https://custom.openai.com")
                
                assert genie is not None
                mock_openai_class.assert_called_once_with(api_key='test_key', base_url="https://custom.openai.com")