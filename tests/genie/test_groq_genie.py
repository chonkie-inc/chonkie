"""Tests for GroqGenie class."""

import json
import os
import sys
import types
from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie, GroqGenie


def _install_fake_groq(mock_groq_class: type) -> None:
    groq_mod = types.ModuleType("groq")
    groq_mod.Groq = mock_groq_class
    sys.modules["groq"] = groq_mod


class TestGroqGenieImportAndConstruction:
    """Test GroqGenie import and basic construction."""

    def test_groq_genie_import(self) -> None:
        assert GroqGenie is not None
        assert issubclass(GroqGenie, BaseGenie)

    def test_groq_genie_has_required_methods(self) -> None:
        assert hasattr(GroqGenie, "generate")
        assert hasattr(GroqGenie, "generate_batch")
        assert hasattr(GroqGenie, "generate_json")
        assert hasattr(GroqGenie, "generate_json_batch")
        assert hasattr(GroqGenie, "_is_available")


class TestGroqGenieErrorHandling:
    """Test GroqGenie error handling."""

    def test_groq_genie_missing_api_key(self) -> None:
        mock_cls = Mock()
        _install_fake_groq(mock_cls)
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="GroqGenie requires an API key"):
                    GroqGenie()
        finally:
            sys.modules.pop("groq", None)

    def test_groq_genie_missing_dependencies(self) -> None:
        with patch.dict(sys.modules, {"groq": None}):
            with pytest.raises(ImportError, match="One or more of the required modules"):
                GroqGenie(api_key="test")


class TestGroqGenieBasicFunctionality:
    """Test GroqGenie with mocked client."""

    def test_groq_genie_initialization(self) -> None:
        mock_cls = Mock()
        mock_cls.return_value = Mock()
        _install_fake_groq(mock_cls)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
                genie = GroqGenie()
            assert isinstance(genie, BaseGenie)
            mock_cls.assert_called_once_with(api_key="k")
        finally:
            sys.modules.pop("groq", None)

    def test_groq_genie_generate_text(self) -> None:
        mock_message = Mock()
        mock_message.content = "out"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_groq(mock_cls)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
                genie = GroqGenie()
                assert genie.generate("p") == "out"
        finally:
            sys.modules.pop("groq", None)

    def test_groq_genie_generate_json(self) -> None:
        data = {"x": True}
        mock_message = Mock()
        mock_message.content = json.dumps(data)
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_groq(mock_cls)
        schema = Mock()
        schema.model_json_schema.return_value = {"type": "object"}
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
                genie = GroqGenie()
                assert genie.generate_json("p", schema) == data
            fmt = mock_client.chat.completions.create.call_args.kwargs["response_format"]
            assert fmt["type"] == "json_schema"
            assert fmt["json_schema"]["name"] == "response"
        finally:
            sys.modules.pop("groq", None)

    def test_groq_genie_none_content_raises(self) -> None:
        mock_message = Mock()
        mock_message.content = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_groq(mock_cls)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
                genie = GroqGenie()
                with pytest.raises(ValueError, match="Groq response content is None"):
                    genie.generate("z")
        finally:
            sys.modules.pop("groq", None)


class TestGroqGenieUtilities:
    """Test GroqGenie helpers."""

    def test_groq_is_available_true(self) -> None:
        with patch("chonkie.genie.groq.importutil.find_spec") as fs:
            fs.side_effect = lambda x: Mock() if x in ("pydantic", "groq") else None
            assert GroqGenie._is_available()

    def test_groq_is_available_false(self) -> None:
        with patch("chonkie.genie.groq.importutil.find_spec", return_value=None):
            assert not GroqGenie._is_available()

    def test_groq_repr(self) -> None:
        mock_cls = Mock(return_value=Mock())
        _install_fake_groq(mock_cls)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
                genie = GroqGenie(model="llama-x")
            assert repr(genie) == "GroqGenie(model=llama-x)"
        finally:
            sys.modules.pop("groq", None)
