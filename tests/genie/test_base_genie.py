"""Tests for BaseGenie class."""

from typing import Any, List, Optional
from unittest.mock import Mock

import pytest

from chonkie import BaseGenie


class ConcreteGenie(BaseGenie):
    """Concrete implementation of BaseGenie for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None, json_responses: Optional[List[Any]] = None) -> None:
        """Initialize with predefined responses.
        
        Args:
            responses: List of text responses to return.
            json_responses: List of JSON responses to return.
            
        """
        super().__init__()
        self.responses = responses or ["test response"]
        self.json_responses = json_responses or [{"key": "value"}]
        self.call_count = 0
        self.json_call_count = 0
        self.prompts: List[str] = []
        self.json_prompts: List[str] = []
    
    def generate(self, prompt: str) -> str:
        """Generate a text response."""
        self.prompts.append(prompt)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def generate_json(self, prompt: str, schema: Any) -> Any:
        """Generate a JSON response."""
        self.json_prompts.append(prompt)
        response = self.json_responses[self.json_call_count % len(self.json_responses)]
        self.json_call_count += 1
        return response


class IncompleteGenie(BaseGenie):
    """Incomplete implementation that only implements generate."""
    
    def generate(self, prompt: str) -> str:
        """Generate a text response."""
        return f"Response to: {prompt}"


@pytest.fixture
def concrete_genie() -> ConcreteGenie:
    """Fixture providing a concrete genie implementation."""
    return ConcreteGenie()


@pytest.fixture
def multi_response_genie() -> ConcreteGenie:
    """Fixture providing a genie with multiple responses."""
    return ConcreteGenie(
        responses=["first", "second", "third"],
        json_responses=[{"id": 1}, {"id": 2}, {"id": 3}]
    )


class TestBaseGenieAbstractMethods:
    """Test BaseGenie abstract method enforcement."""
    
    def test_cannot_instantiate_base_genie(self) -> None:
        """Test that BaseGenie cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGenie()
    
    def test_concrete_implementation_works(self, concrete_genie: ConcreteGenie) -> None:
        """Test that concrete implementation can be instantiated."""
        assert isinstance(concrete_genie, BaseGenie)
        assert isinstance(concrete_genie, ConcreteGenie)
    
    def test_incomplete_implementation_works(self) -> None:
        """Test that incomplete implementation can be instantiated."""
        genie = IncompleteGenie()
        assert isinstance(genie, BaseGenie)


class TestBaseGenieGenerate:
    """Test BaseGenie text generation methods."""
    
    def test_generate_single_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test generating response for single prompt."""
        prompt = "What is the meaning of life?"
        response = concrete_genie.generate(prompt)
        
        assert response == "test response"
        assert len(concrete_genie.prompts) == 1
        assert concrete_genie.prompts[0] == prompt
        assert concrete_genie.call_count == 1
    
    def test_generate_multiple_calls(self, multi_response_genie: ConcreteGenie) -> None:
        """Test multiple generate calls with cycling responses."""
        prompts = ["First question", "Second question", "Third question", "Fourth question"]
        
        responses = [multi_response_genie.generate(prompt) for prompt in prompts]
        
        assert responses == ["first", "second", "third", "first"]  # Should cycle
        assert multi_response_genie.prompts == prompts
        assert multi_response_genie.call_count == 4
    
    def test_generate_empty_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test generating response for empty prompt."""
        response = concrete_genie.generate("")
        
        assert response == "test response"
        assert concrete_genie.prompts == [""]
    
    def test_generate_long_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test generating response for very long prompt."""
        long_prompt = "A" * 10000  # Very long prompt
        response = concrete_genie.generate(long_prompt)
        
        assert response == "test response"
        assert concrete_genie.prompts[0] == long_prompt


class TestBaseGenieBatchGenerate:
    """Test BaseGenie batch text generation methods."""
    
    def test_generate_batch_single_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test batch generation with single prompt."""
        prompts = ["What is AI?"]
        responses = concrete_genie.generate_batch(prompts)
        
        assert responses == ["test response"]
        assert concrete_genie.prompts == prompts
        assert concrete_genie.call_count == 1
    
    def test_generate_batch_multiple_prompts(self, multi_response_genie: ConcreteGenie) -> None:
        """Test batch generation with multiple prompts."""
        prompts = ["First", "Second", "Third"]
        responses = multi_response_genie.generate_batch(prompts)
        
        assert responses == ["first", "second", "third"]
        assert multi_response_genie.prompts == prompts
        assert multi_response_genie.call_count == 3
    
    def test_generate_batch_empty_list(self, concrete_genie: ConcreteGenie) -> None:
        """Test batch generation with empty prompt list."""
        responses = concrete_genie.generate_batch([])
        
        assert responses == []
        assert concrete_genie.prompts == []
        assert concrete_genie.call_count == 0
    
    def test_generate_batch_with_duplicates(self, concrete_genie: ConcreteGenie) -> None:
        """Test batch generation with duplicate prompts."""
        prompts = ["Same question", "Same question", "Different question"]
        responses = concrete_genie.generate_batch(prompts)
        
        assert len(responses) == 3
        assert all(response == "test response" for response in responses)
        assert concrete_genie.prompts == prompts
        assert concrete_genie.call_count == 3


class TestBaseGenieGenerateJson:
    """Test BaseGenie JSON generation methods."""
    
    def test_generate_json_single_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test JSON generation for single prompt."""
        prompt = "Generate JSON data"
        schema = Mock()
        response = concrete_genie.generate_json(prompt, schema)
        
        assert response == {"key": "value"}
        assert len(concrete_genie.json_prompts) == 1
        assert concrete_genie.json_prompts[0] == prompt
        assert concrete_genie.json_call_count == 1
    
    def test_generate_json_multiple_calls(self, multi_response_genie: ConcreteGenie) -> None:
        """Test multiple JSON generation calls."""
        prompts = ["First JSON", "Second JSON", "Third JSON", "Fourth JSON"]
        schema = Mock()
        
        responses = [multi_response_genie.generate_json(prompt, schema) for prompt in prompts]
        
        expected = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 1}]  # Should cycle
        assert responses == expected
        assert multi_response_genie.json_prompts == prompts
        assert multi_response_genie.json_call_count == 4
    
    def test_generate_json_default_not_implemented(self) -> None:
        """Test that default generate_json raises NotImplementedError."""
        genie = IncompleteGenie()
        
        with pytest.raises(NotImplementedError):
            genie.generate_json("test prompt", Mock())
    
    def test_generate_json_with_complex_schema(self, concrete_genie: ConcreteGenie) -> None:
        """Test JSON generation with complex schema object."""
        prompt = "Generate complex JSON"
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        response = concrete_genie.generate_json(prompt, schema)
        
        assert response == {"key": "value"}
        assert concrete_genie.json_prompts[0] == prompt


class TestBaseGenieBatchGenerateJson:
    """Test BaseGenie batch JSON generation methods."""
    
    def test_generate_json_batch_single_prompt(self, concrete_genie: ConcreteGenie) -> None:
        """Test batch JSON generation with single prompt."""
        prompts = ["Generate JSON"]
        schema = Mock()
        responses = concrete_genie.generate_json_batch(prompts, schema)
        
        assert responses == [{"key": "value"}]
        assert concrete_genie.json_prompts == prompts
        assert concrete_genie.json_call_count == 1
    
    def test_generate_json_batch_multiple_prompts(self, multi_response_genie: ConcreteGenie) -> None:
        """Test batch JSON generation with multiple prompts."""
        prompts = ["First JSON", "Second JSON", "Third JSON"]
        schema = Mock()
        responses = multi_response_genie.generate_json_batch(prompts, schema)
        
        expected = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert responses == expected
        assert multi_response_genie.json_prompts == prompts
        assert multi_response_genie.json_call_count == 3
    
    def test_generate_json_batch_empty_list(self, concrete_genie: ConcreteGenie) -> None:
        """Test batch JSON generation with empty prompt list."""
        schema = Mock()
        responses = concrete_genie.generate_json_batch([], schema)
        
        assert responses == []
        assert concrete_genie.json_prompts == []
        assert concrete_genie.json_call_count == 0
    
    def test_generate_json_batch_not_implemented(self) -> None:
        """Test batch JSON generation with incomplete implementation."""
        genie = IncompleteGenie()
        prompts = ["test1", "test2"]
        schema = Mock()
        
        with pytest.raises(NotImplementedError):
            genie.generate_json_batch(prompts, schema)


class TestBaseGenieEdgeCases:
    """Test BaseGenie edge cases and error handling."""
    
    def test_genie_state_preservation(self, concrete_genie: ConcreteGenie) -> None:
        """Test that genie maintains state across calls."""
        # Make some text generation calls
        concrete_genie.generate("text1")
        concrete_genie.generate("text2")
        
        # Make some JSON generation calls
        concrete_genie.generate_json("json1", Mock())
        concrete_genie.generate_json("json2", Mock())
        
        # Check that state is preserved correctly
        assert concrete_genie.call_count == 2
        assert concrete_genie.json_call_count == 2
        assert concrete_genie.prompts == ["text1", "text2"]
        assert concrete_genie.json_prompts == ["json1", "json2"]
    
    def test_mixed_batch_and_single_calls(self, multi_response_genie: ConcreteGenie) -> None:
        """Test mixing batch and single generation calls."""
        # Single call
        response1 = multi_response_genie.generate("single")
        
        # Batch call
        batch_responses = multi_response_genie.generate_batch(["batch1", "batch2"])
        
        # Another single call
        response2 = multi_response_genie.generate("single2")
        
        assert response1 == "first"
        assert batch_responses == ["second", "third"]
        assert response2 == "first"  # Should cycle back
        
        expected_prompts = ["single", "batch1", "batch2", "single2"]
        assert multi_response_genie.prompts == expected_prompts
        assert multi_response_genie.call_count == 4
    
    def test_unicode_and_special_characters(self, concrete_genie: ConcreteGenie) -> None:
        """Test handling of unicode and special characters in prompts."""
        unicode_prompt = "Hello 世界! 🌍 Special chars: @#$%^&*()"
        response = concrete_genie.generate(unicode_prompt)
        
        assert response == "test response"
        assert concrete_genie.prompts[0] == unicode_prompt
    
    def test_very_large_batch(self, concrete_genie: ConcreteGenie) -> None:
        """Test handling of very large batch sizes."""
        large_batch = [f"prompt_{i}" for i in range(1000)]
        responses = concrete_genie.generate_batch(large_batch)
        
        assert len(responses) == 1000
        assert all(response == "test response" for response in responses)
        assert concrete_genie.call_count == 1000
        assert concrete_genie.prompts == large_batch


class TestBaseGenieInheritance:
    """Test BaseGenie inheritance and polymorphism."""
    
    def test_polymorphism(self) -> None:
        """Test that concrete implementations work polymorphically."""
        genies: List[BaseGenie] = [
            ConcreteGenie(["response1"]),
            ConcreteGenie(["response2"]),
            IncompleteGenie()
        ]
        
        responses = []
        for genie in genies[:2]:  # Only test the ones with proper implementation
            responses.append(genie.generate("test"))
        
        assert responses == ["response1", "response2"]
    
    def test_method_resolution_order(self, concrete_genie: ConcreteGenie) -> None:
        """Test that method resolution follows expected order."""
        # Test that the concrete implementation's methods are called
        response = concrete_genie.generate("test")
        
        assert response == "test response"
        assert len(concrete_genie.prompts) == 1
        
        # Test that batch methods use the overridden generate method
        batch_responses = concrete_genie.generate_batch(["test1", "test2"])
        
        assert batch_responses == ["test response", "test response"]
        assert len(concrete_genie.prompts) == 3  # 1 + 2 from batch


class TestBaseGenieTypeHints:
    """Test BaseGenie type hints and interface compliance."""
    
    def test_return_types(self, concrete_genie: ConcreteGenie) -> None:
        """Test that methods return expected types."""
        # Test generate returns string
        text_response = concrete_genie.generate("test")
        assert isinstance(text_response, str)
        
        # Test generate_batch returns list of strings
        batch_response = concrete_genie.generate_batch(["test1", "test2"])
        assert isinstance(batch_response, list)
        assert all(isinstance(response, str) for response in batch_response)
        
        # Test generate_json returns the configured type
        json_response = concrete_genie.generate_json("test", Mock())
        assert isinstance(json_response, dict)
        
        # Test generate_json_batch returns list of configured types
        json_batch_response = concrete_genie.generate_json_batch(["test1", "test2"], Mock())
        assert isinstance(json_batch_response, list)
        assert all(isinstance(response, dict) for response in json_batch_response)
    
    def test_parameter_types(self, concrete_genie: ConcreteGenie) -> None:
        """Test that methods accept expected parameter types."""
        # All these should work without type errors
        concrete_genie.generate("string prompt")
        concrete_genie.generate_batch(["prompt1", "prompt2"])
        concrete_genie.generate_json("json prompt", Mock())
        concrete_genie.generate_json_batch(["json1", "json2"], Mock())