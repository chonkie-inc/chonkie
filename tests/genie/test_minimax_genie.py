"""Tests for MiniMaxGenie class."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from chonkie import BaseGenie, MiniMaxGenie


class TestMiniMaxGenieImportAndConstruction:
    """Test MiniMaxGenie import and basic construction."""

    def test_minimax_genie_import(self) -> None:
        """Test that MiniMaxGenie can be imported."""
        assert MiniMaxGenie is not None
        assert issubclass(MiniMaxGenie, BaseGenie)

    def test_minimax_genie_has_required_methods(self) -> None:
        """Test that MiniMaxGenie has all required methods."""
        assert hasattr(MiniMaxGenie, "generate")
        assert hasattr(MiniMaxGenie, "agenerate")
        assert hasattr(MiniMaxGenie, "generate_batch")
        assert hasattr(MiniMaxGenie, "generate_json")
        assert hasattr(MiniMaxGenie, "generate_json_batch")
        assert hasattr(MiniMaxGenie, "agenerate_json")
        assert hasattr(MiniMaxGenie, "_is_available")

    def test_minimax_genie_available_models(self) -> None:
        """Test that AVAILABLE_MODELS is defined."""
        assert "MiniMax-M2.7" in MiniMaxGenie.AVAILABLE_MODELS
        assert "MiniMax-M2.7-highspeed" in MiniMaxGenie.AVAILABLE_MODELS
        assert "MiniMax-M2.5" in MiniMaxGenie.AVAILABLE_MODELS
        assert "MiniMax-M2.5-highspeed" in MiniMaxGenie.AVAILABLE_MODELS


class TestMiniMaxGenieErrorHandling:
    """Test MiniMaxGenie error handling."""

    def test_minimax_genie_missing_api_key(self) -> None:
        """Test MiniMaxGenie raises error without API key."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="MiniMaxGenie requires an API key"):
                    MiniMaxGenie()

    def test_minimax_genie_missing_dependencies(self) -> None:
        """Test MiniMaxGenie raises error without dependencies."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=False):
            with pytest.raises(
                ImportError,
                match="One or more of the required modules are not available",
            ):
                MiniMaxGenie(api_key="test")


class TestMiniMaxGenieBasicFunctionality:
    """Test MiniMaxGenie basic functionality with mocking."""

    def test_minimax_genie_initialization(self) -> None:
        """Test MiniMaxGenie can be initialized with mocked dependencies."""
        mock_openai_class = Mock()
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()

                    assert genie is not None
                    assert isinstance(genie, BaseGenie)
                    assert genie.model == "MiniMax-M2.7"
                    mock_openai_class.assert_called_once_with(
                        api_key="test_key",
                        base_url="https://api.minimax.io/v1",
                    )

    def test_minimax_genie_custom_model(self) -> None:
        """Test MiniMaxGenie with custom model name."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(model="MiniMax-M2.5-highspeed")

                    assert genie.model == "MiniMax-M2.5-highspeed"

    def test_minimax_genie_generate_text(self) -> None:
        """Test MiniMaxGenie text generation with mocked response."""
        mock_message = Mock()
        mock_message.content = "Generated response from MiniMax"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        result = genie.generate("Test prompt")

                    assert result == "Generated response from MiniMax"
                    mock_client.chat.completions.create.assert_called_once()

    def test_minimax_genie_generate_strips_think_tags(self) -> None:
        """Test that think tags are stripped from output."""
        mock_message = Mock()
        mock_message.content = "<think>Let me reason...</think>\n\nActual answer here"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        result = genie.generate("Test prompt")

                    assert "<think>" not in result
                    assert "Actual answer here" in result

    def test_minimax_genie_batch_generation(self) -> None:
        """Test MiniMaxGenie batch generation."""
        mock_responses = []
        for i in range(3):
            mock_message = Mock()
            mock_message.content = f"Response {i}"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_responses.append(mock_response)

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai_class = Mock(return_value=mock_client)

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                        results = genie.generate_batch(prompts)

                    assert len(results) == 3
                    assert results == ["Response 0", "Response 1", "Response 2"]
                    assert mock_client.chat.completions.create.call_count == 3

    def test_minimax_genie_generate_json(self) -> None:
        """Test MiniMaxGenie JSON generation with mocked response."""
        mock_message = Mock()
        mock_message.content = '{"key": "value", "count": 42}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        mock_schema = Mock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "count": {"type": "integer"},
            },
        }

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        result = genie.generate_json("Test prompt", mock_schema)

                    assert result == {"key": "value", "count": 42}
                    mock_client.chat.completions.create.assert_called_once()

    def test_minimax_genie_generate_json_strips_think(self) -> None:
        """Test that JSON generation strips think tags before parsing."""
        mock_message = Mock()
        mock_message.content = '<think>reasoning</think>{"answer": "clean"}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        mock_schema = Mock()
        mock_schema.model_json_schema.return_value = {"type": "object"}

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        result = genie.generate_json("Test prompt", mock_schema)

                    assert result == {"answer": "clean"}

    def test_minimax_genie_none_content_raises(self) -> None:
        """Test that None content raises ValueError."""
        mock_message = Mock()
        mock_message.content = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                        with pytest.raises(ValueError, match="MiniMax response content is None"):
                            genie.generate("test")


class TestMiniMaxGenieTemperature:
    """Test temperature clamping behavior."""

    def test_temperature_default(self) -> None:
        """Test default temperature is 0.7."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie()
                    assert genie.temperature == 0.7

    def test_temperature_clamped_low(self) -> None:
        """Test that temperature below 0.01 is clamped."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(temperature=0.0)
                    assert genie.temperature == 0.01

    def test_temperature_clamped_high(self) -> None:
        """Test that temperature above 1.0 is clamped."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(temperature=2.0)
                    assert genie.temperature == 1.0

    def test_temperature_passed_to_api(self) -> None:
        """Test that temperature is passed in API calls."""
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class = Mock(return_value=mock_client)

        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(temperature=0.3)
                        genie.generate("test")

                    call_kwargs = mock_client.chat.completions.create.call_args[1]
                    assert call_kwargs["temperature"] == 0.3


class TestMiniMaxGenieThinkTagStripping:
    """Test the _strip_think_tags static method."""

    def test_strip_simple_think_tags(self) -> None:
        """Test stripping a single think block."""
        text = "<think>I need to reason about this</think>The answer is 42."
        result = MiniMaxGenie._strip_think_tags(text)
        assert result == "The answer is 42."

    def test_strip_multiline_think_tags(self) -> None:
        """Test stripping multiline think blocks."""
        text = "<think>\nStep 1: Think about it\nStep 2: More thinking\n</think>\nFinal answer."
        result = MiniMaxGenie._strip_think_tags(text)
        assert result == "Final answer."

    def test_no_think_tags(self) -> None:
        """Test that text without think tags is unchanged."""
        text = "Plain text without think tags."
        result = MiniMaxGenie._strip_think_tags(text)
        assert result == "Plain text without think tags."

    def test_empty_think_tags(self) -> None:
        """Test stripping empty think blocks."""
        text = "<think></think>Content after."
        result = MiniMaxGenie._strip_think_tags(text)
        assert result == "Content after."


class TestMiniMaxGenieUtilities:
    """Test MiniMaxGenie utility methods."""

    def test_minimax_genie_is_available_true(self) -> None:
        """Test _is_available returns True when dependencies are installed."""
        with patch("chonkie.genie.minimax.importutil.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda x: Mock() if x in ["openai", "pydantic"] else None
            assert MiniMaxGenie._is_available()

    def test_minimax_genie_is_available_false_openai(self) -> None:
        """Test _is_available returns False when openai is missing."""
        with patch("chonkie.genie.minimax.importutil.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda x: Mock() if x == "pydantic" else None
            assert not MiniMaxGenie._is_available()

    def test_minimax_genie_is_available_false_pydantic(self) -> None:
        """Test _is_available returns False when pydantic is missing."""
        with patch("chonkie.genie.minimax.importutil.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = lambda x: Mock() if x == "openai" else None
            assert not MiniMaxGenie._is_available()

    def test_minimax_genie_repr(self) -> None:
        """Test MiniMaxGenie string representation."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(model="MiniMax-M2.7")
                        repr_str = repr(genie)

                    assert "MiniMaxGenie" in repr_str
                    assert "MiniMax-M2.7" in repr_str

    def test_minimax_genie_custom_base_url(self) -> None:
        """Test MiniMaxGenie with custom base URL."""
        mock_openai_class = Mock()
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", mock_openai_class):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"}):
                        genie = MiniMaxGenie(base_url="https://custom.minimax.io/v1")

                    assert genie is not None
                    mock_openai_class.assert_called_once_with(
                        api_key="test_key",
                        base_url="https://custom.minimax.io/v1",
                    )

    def test_minimax_genie_explicit_api_key(self) -> None:
        """Test MiniMaxGenie with explicit API key parameter."""
        with patch.object(MiniMaxGenie, "_is_available", return_value=True):
            with patch("chonkie.genie.minimax.OpenAI", Mock()):
                with patch("chonkie.genie.minimax.AsyncOpenAI", Mock()):
                    genie = MiniMaxGenie(api_key="explicit-key")
                    assert genie.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "MINIMAX_API_KEY" not in os.environ,
    reason="Skipping integration test — requires MINIMAX_API_KEY",
)
class TestMiniMaxGenieIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_real_generate(self) -> None:
        """Integration: generate a text response."""
        genie = MiniMaxGenie()
        result = genie.generate("What is 2 + 2? Answer with just the number.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_real_generate_batch(self) -> None:
        """Integration: batch generate responses."""
        genie = MiniMaxGenie()
        results = genie.generate_batch([
            "What is 1 + 1? Answer with just the number.",
            "What is 3 + 3? Answer with just the number.",
        ])
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_real_generate_json(self) -> None:
        """Integration: generate a JSON response."""
        from pydantic import BaseModel

        class MathResult(BaseModel):
            answer: int
            explanation: str

        genie = MiniMaxGenie()
        result = genie.generate_json(
            "What is 2 + 2? Provide the answer and a brief explanation.",
            MathResult,
        )
        assert isinstance(result, dict)
        assert "answer" in result
        assert "explanation" in result


if __name__ == "__main__":
    pytest.main()
