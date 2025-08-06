"""Tests for AzureOpenAIGenie class."""

from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie
from chonkie.genie.azure_openai import AzureOpenAIGenie


class TestAzureAIGenieImportAndConstruction:
    """Test AzureOpenAIGenie import and basic construction."""

    def test_import_and_type(self):
        """Test that AzureOpenAIGenie can be imported and is a subclass of BaseGenie."""
        assert AzureOpenAIGenie is not None
        assert issubclass(AzureOpenAIGenie, BaseGenie)

    def test_has_required_methods(self):
        """Test that AzureOpenAIGenie has all required methods."""
        assert hasattr(AzureOpenAIGenie, "generate")
        assert hasattr(AzureOpenAIGenie, "generate_json")
        assert hasattr(AzureOpenAIGenie, "_is_available")


class TestAzureAIGenieErrorHandling:
    """Test AzureOpenAIGenie error handling."""

    def test_missing_endpoint(self):
        """Test AzureOpenAIGenie raises error when endpoint is None or empty."""
        with patch.object(AzureOpenAIGenie, "_is_available", return_value=True):
            with pytest.raises(ValueError, match="`azure_endpoint` is required"):
                AzureOpenAIGenie(azure_endpoint=None, deployment="x", api_version="2024-02-15-preview")

    def test_missing_dependencies(self):
        """Test AzureOpenAIGenie raises error without dependencies."""
        with patch.object(AzureOpenAIGenie, "_is_available", return_value=False):
            with pytest.raises(ImportError, match="Missing required modules"):
                AzureOpenAIGenie(
                    azure_endpoint="https://test.openai.azure.com",
                    deployment="deployment",
                )


class TestAzureAIGenieMocked:
    """Test AzureOpenAIGenie with mocked dependencies."""

    def test_initialization_with_key(self):
        """Test AzureOpenAIGenie initialization with mocked dependencies."""
        mock_client = Mock()
        mock_class = Mock(return_value=mock_client)

        def mock_import(self):
            import chonkie.genie.azure_openai as m

            m.AzureOpenAI = mock_class
            m.BaseModel = Mock()

        with (
            patch.object(AzureOpenAIGenie, "_is_available", return_value=True),
            patch.object(AzureOpenAIGenie, "_import_dependencies", mock_import),
        ):

            genie = AzureOpenAIGenie(
                azure_api_key="test",
                azure_endpoint="https://test.openai.azure.com",
                deployment="deployment",
            )

            assert isinstance(genie, BaseGenie)
            mock_class.assert_called_once()

    def test_generate_returns_text(self):
        """Test AzureOpenAIGenie generate method returns text."""
        mock_msg = Mock()
        mock_msg.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_msg
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_class = Mock(return_value=mock_client)

        def mock_import(self):
            import chonkie.genie.azure_openai as m

            m.AzureOpenAI = mock_class
            m.BaseModel = Mock()

        with (
            patch.object(AzureOpenAIGenie, "_is_available", return_value=True),
            patch.object(AzureOpenAIGenie, "_import_dependencies", mock_import),
        ):

            genie = AzureOpenAIGenie(
                azure_api_key="test",
                azure_endpoint="https://test.openai.azure.com",
                deployment="deployment",
            )
            result = genie.generate("Hello?")
            assert result == "Test response"

    def test_repr(self):
        """Test AzureOpenAIGenie string representation."""
        mock_client = Mock()
        mock_class = Mock(return_value=mock_client)

        def mock_import(self):
            import chonkie.genie.azure_openai as m

            m.AzureOpenAI = mock_class
            m.BaseModel = Mock()

        with (
            patch.object(AzureOpenAIGenie, "_is_available", return_value=True),
            patch.object(AzureOpenAIGenie, "_import_dependencies", mock_import),
        ):

            genie = AzureOpenAIGenie(
                azure_api_key="test",
                azure_endpoint="https://test.openai.azure.com",
                deployment="deployment",
                model="gpt-4o",
            )
            rep = repr(genie)
            assert "AzureOpenAIGenie" in rep
            assert "gpt-4o" in rep
            assert "deployment" in rep


class TestAzureOpenAIGenieUtilities:
    """Tests for AzureOpenAIGenie utility methods."""

    def test_azure_genie_is_available_true(self) -> None:
        """Test _is_available returns True when dependencies are installed."""
        with patch("chonkie.genie.azure_openai.importutil.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda x: (
                Mock() if x in ["openai", "pydantic", "azure.identity"] else None
            )

            def mock_import_dependencies(self):
                import chonkie.genie.azure_openai as azure_module

                azure_module.AzureOpenAI = Mock()
                azure_module.BaseModel = Mock()

            with patch.object(
                AzureOpenAIGenie, "_import_dependencies", mock_import_dependencies
            ):
                genie = AzureOpenAIGenie(
                    azure_api_key="test_key",
                    azure_endpoint="https://custom.azure.com",
                    deployment="gpt-4",
                )
                result = genie._is_available()
                assert result is True

    def test_azure_genie_is_available_false(self) -> None:
        """Test _is_available returns False when dependencies are missing."""
        with patch(
            "chonkie.genie.azure_openai.importutil.find_spec", return_value=None
        ):

            def mock_import_dependencies(self):
                import chonkie.genie.azure_openai as azure_module

                azure_module.AzureOpenAI = Mock()
                azure_module.BaseModel = Mock()

            with patch.object(
                AzureOpenAIGenie, "_import_dependencies", mock_import_dependencies
            ):
                genie = AzureOpenAIGenie(
                    azure_api_key="test_key",
                    azure_endpoint="https://custom.azure.com",
                    deployment="gpt-4",
                )
                result = genie._is_available()
                assert result is False

    def test_azure_genie_repr(self) -> None:
        """Test AzureOpenAIGenie string representation."""
        mock_client = Mock()
        mock_openai_class = Mock(return_value=mock_client)

        def mock_import_dependencies(self):
            import chonkie.genie.azure_openai as azure_module

            azure_module.AzureOpenAI = mock_openai_class
            azure_module.BaseModel = Mock()

        with patch.object(AzureOpenAIGenie, "_is_available", return_value=True):
            with patch.object(
                AzureOpenAIGenie, "_import_dependencies", mock_import_dependencies
            ):
                genie = AzureOpenAIGenie(
                    model="gpt-4",
                    azure_api_key="test_key",
                    azure_endpoint="https://custom.azure.com",
                    deployment="gpt-4",
                )
                repr_str = repr(genie)
                assert "AzureOpenAIGenie" in repr_str
                assert "gpt-4" in repr_str

    def test_azure_genie_custom_base_url(self) -> None:
        """Test AzureOpenAIGenie with custom base URL."""
        mock_client = Mock()
        mock_openai_class = Mock(return_value=mock_client)

        def mock_import_dependencies(self):
            import chonkie.genie.azure_openai as azure_module

            azure_module.AzureOpenAI = mock_openai_class
            azure_module.BaseModel = Mock()

        with patch.object(AzureOpenAIGenie, "_is_available", return_value=True):
            with patch.object(
                AzureOpenAIGenie, "_import_dependencies", mock_import_dependencies
            ):
                genie = AzureOpenAIGenie(
                    azure_api_key="test_key",
                    azure_endpoint="https://custom.azure.com",
                    deployment="gpt-4",
                )
                assert genie is not None
                mock_openai_class.assert_called_once()
