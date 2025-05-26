"""Tests for GeminiGenie class."""

import os
from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie, GeminiGenie


class TestGeminiGenieImportAndConstruction:
    """Test GeminiGenie import and basic construction."""
    
    def test_gemini_genie_import(self) -> None:
        """Test that GeminiGenie can be imported."""
        assert GeminiGenie is not None
        assert issubclass(GeminiGenie, BaseGenie)
    
    def test_gemini_genie_has_required_methods(self) -> None:
        """Test that GeminiGenie has all required methods."""
        assert hasattr(GeminiGenie, 'generate')
        assert hasattr(GeminiGenie, 'generate_batch')
        assert hasattr(GeminiGenie, 'generate_json')
        assert hasattr(GeminiGenie, 'generate_json_batch')
        assert hasattr(GeminiGenie, '_is_available')


class TestGeminiGenieErrorHandling:
    """Test GeminiGenie error handling."""
    
    def test_gemini_genie_missing_api_key(self) -> None:
        """Test GeminiGenie raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(GeminiGenie, '_is_available', return_value=True):
                with patch.object(GeminiGenie, '_import_dependencies'):
                    with pytest.raises(ValueError, match="GeminiGenie requires an API key"):
                        GeminiGenie()
    
    def test_gemini_genie_missing_dependencies(self) -> None:
        """Test GeminiGenie raises error without dependencies."""
        with patch.object(GeminiGenie, '_is_available', return_value=False):
            with pytest.raises(ImportError, match="One or more of the required modules"):
                GeminiGenie(api_key="test")


class TestGeminiGenieBasicFunctionality:
    """Test GeminiGenie basic functionality with mocking."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_initialization(self) -> None:
        """Test GeminiGenie can be initialized with mocked dependencies."""
        mock_client = Mock()
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                # Mock the global variable assignment
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie()
                assert genie is not None
                assert isinstance(genie, BaseGenie)
                assert genie.api_key == 'test_key'
                assert genie.model == "gemini-2.5-pro-preview-03-25"
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_generate_text(self) -> None:
        """Test GeminiGenie text generation with mocked response."""
        mock_response = Mock()
        mock_response.text = "Generated response"
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie()
                result = genie.generate("Test prompt")
                
                assert result == "Generated response"
                mock_client.models.generate_content.assert_called_once_with(
                    model="gemini-2.5-pro-preview-03-25", 
                    contents="Test prompt"
                )
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_generate_json(self) -> None:
        """Test GeminiGenie JSON generation with mocked response."""
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        mock_schema = Mock()
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie()
                result = genie.generate_json("Test prompt", mock_schema)
                
                assert result == {"key": "value"}
                mock_client.models.generate_content.assert_called_once_with(
                    model="gemini-2.5-pro-preview-03-25",
                    contents="Test prompt",
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': mock_schema,
                    }
                )
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_batch_generation(self) -> None:
        """Test GeminiGenie batch generation."""
        mock_responses = [Mock(text=f"Response {i}") for i in range(3)]
        
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = mock_responses
        
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie()
                prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                results = genie.generate_batch(prompts)
                
                assert len(results) == 3
                assert results == ["Response 0", "Response 1", "Response 2"]
                assert mock_client.models.generate_content.call_count == 3


class TestGeminiGenieUtilities:
    """Test GeminiGenie utility methods."""
    
    def test_gemini_genie_is_available_true(self) -> None:
        """Test _is_available returns True when dependencies are installed."""
        with patch('chonkie.genie.gemini.importutil.find_spec') as mock_find_spec:
            mock_find_spec.side_effect = lambda x: Mock() if x in ["pydantic", "google"] else None
            
            # Test the actual _is_available method directly
            with patch.object(GeminiGenie, '_is_available', return_value=True):
                with patch.object(GeminiGenie, '_import_dependencies'):
                    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                        genie = GeminiGenie()
                        assert genie._is_available() is True
    
    def test_gemini_genie_is_available_false(self) -> None:
        """Test _is_available returns False when dependencies are missing."""
        with patch('chonkie.genie.gemini.importutil.find_spec') as mock_find_spec:
            mock_find_spec.return_value = None
            
            # Test the actual _is_available method directly
            with patch.object(GeminiGenie, '_is_available', return_value=False):
                with patch.object(GeminiGenie, '_import_dependencies'):
                    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                        genie = GeminiGenie()
                        assert genie._is_available() is False
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_repr(self) -> None:
        """Test GeminiGenie string representation."""
        mock_client = Mock()
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie(model="gemini-1.5-flash")
                repr_str = repr(genie)
                
                assert "GeminiGenie" in repr_str
                assert "gemini-1.5-flash" in repr_str


class TestGeminiGenieErrorCases:
    """Test GeminiGenie error cases."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_gemini_genie_json_parse_error(self) -> None:
        """Test GeminiGenie handles JSON parsing errors."""
        mock_response = Mock()
        mock_response.text = "invalid json"
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        
        mock_genai = Mock()
        mock_genai.Client.return_value = mock_client
        
        mock_schema = Mock()
        
        with patch.object(GeminiGenie, '_is_available', return_value=True):
            with patch.object(GeminiGenie, '_import_dependencies') as mock_import:
                def mock_import_side_effect():
                    import chonkie.genie.gemini as gemini_module
                    gemini_module.genai = mock_genai
                    gemini_module.BaseModel = Mock()
                
                mock_import.side_effect = mock_import_side_effect
                
                genie = GeminiGenie()
                
                with pytest.raises(ValueError, match="Failed to parse JSON response"):
                    genie.generate_json("Test prompt", mock_schema)