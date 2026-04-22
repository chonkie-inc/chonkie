"""Tests for CerebrasGenie class."""

import json
import os
import sys
import types
from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie, CerebrasGenie


def _install_fake_cerebras_sdk(mock_cerebras_class: type) -> None:
    """Register minimal ``cerebras.cloud.sdk`` package tree in ``sys.modules``."""
    cerebras_pkg = types.ModuleType("cerebras")
    cloud_pkg = types.ModuleType("cerebras.cloud")
    cloud_pkg.__path__ = []
    sdk_pkg = types.ModuleType("cerebras.cloud.sdk")
    sdk_pkg.Cerebras = mock_cerebras_class
    sys.modules["cerebras"] = cerebras_pkg
    sys.modules["cerebras.cloud"] = cloud_pkg
    sys.modules["cerebras.cloud.sdk"] = sdk_pkg


class TestCerebrasGenieImportAndConstruction:
    """Test CerebrasGenie import and basic construction."""

    def test_cerebras_genie_import(self) -> None:
        assert CerebrasGenie is not None
        assert issubclass(CerebrasGenie, BaseGenie)

    def test_cerebras_genie_has_required_methods(self) -> None:
        assert hasattr(CerebrasGenie, "generate")
        assert hasattr(CerebrasGenie, "generate_batch")
        assert hasattr(CerebrasGenie, "generate_json")
        assert hasattr(CerebrasGenie, "generate_json_batch")
        assert hasattr(CerebrasGenie, "_is_available")


class TestCerebrasGenieErrorHandling:
    """Test CerebrasGenie error handling."""

    def test_cerebras_genie_missing_api_key(self) -> None:
        mock_cls = Mock()
        _install_fake_cerebras_sdk(mock_cls)
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="CerebrasGenie requires an API key"):
                    CerebrasGenie()
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)

    def test_cerebras_genie_missing_dependencies(self) -> None:
        cerebras_pkg = types.ModuleType("cerebras")
        cloud_pkg = types.ModuleType("cerebras.cloud")
        with patch.dict(
            sys.modules,
            {"cerebras": cerebras_pkg, "cerebras.cloud": cloud_pkg, "cerebras.cloud.sdk": None},
        ):
            with pytest.raises(ImportError, match="One or more of the required modules"):
                CerebrasGenie(api_key="test")


class TestCerebrasGenieBasicFunctionality:
    """Test CerebrasGenie with mocked SDK."""

    def test_cerebras_genie_initialization(self) -> None:
        mock_cls = Mock()
        mock_client = Mock()
        mock_cls.return_value = mock_client
        _install_fake_cerebras_sdk(mock_cls)
        try:
            with patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
                genie = CerebrasGenie()
            assert isinstance(genie, BaseGenie)
            mock_cls.assert_called_once_with(api_key="k")
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)

    def test_cerebras_genie_generate_text(self) -> None:
        mock_message = Mock()
        mock_message.content = "ok"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_cerebras_sdk(mock_cls)
        try:
            with patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
                genie = CerebrasGenie()
                out = genie.generate("hi")
            assert out == "ok"
            mock_client.chat.completions.create.assert_called_once()
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)

    def test_cerebras_genie_generate_json(self) -> None:
        payload = {"a": 1}
        mock_message = Mock()
        mock_message.content = json.dumps(payload)
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_cerebras_sdk(mock_cls)
        mock_schema = Mock()
        mock_schema.model_json_schema.return_value = {"type": "object"}
        try:
            with patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
                genie = CerebrasGenie()
                out = genie.generate_json("prompt", mock_schema)
            assert out == payload
            calls = mock_client.chat.completions.create.call_args_list
            assert len(calls) == 1
            assert calls[0].kwargs["response_format"] == {"type": "json_object"}
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)

    def test_cerebras_genie_none_content_raises(self) -> None:
        mock_message = Mock()
        mock_message.content = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls = Mock(return_value=mock_client)
        _install_fake_cerebras_sdk(mock_cls)
        try:
            with patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
                genie = CerebrasGenie()
                with pytest.raises(ValueError, match="Cerebras response content is None"):
                    genie.generate("x")
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)


class TestCerebrasGenieUtilities:
    """Test CerebrasGenie helpers."""

    def test_cerebras_is_available_true(self) -> None:
        with patch("chonkie.genie.cerebras.importutil.find_spec") as fs:
            fs.side_effect = lambda x: Mock() if x in ("pydantic", "cerebras.cloud.sdk") else None
            assert CerebrasGenie._is_available()

    def test_cerebras_is_available_false(self) -> None:
        with patch("chonkie.genie.cerebras.importutil.find_spec", return_value=None):
            assert not CerebrasGenie._is_available()

    def test_cerebras_repr(self) -> None:
        mock_cls = Mock(return_value=Mock())
        _install_fake_cerebras_sdk(mock_cls)
        try:
            with patch.dict(os.environ, {"CEREBRAS_API_KEY": "k"}):
                genie = CerebrasGenie(model="m")
            assert repr(genie) == "CerebrasGenie(model=m)"
        finally:
            for key in ("cerebras", "cerebras.cloud", "cerebras.cloud.sdk"):
                sys.modules.pop(key, None)
