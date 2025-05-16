"""Claude Genie."""
import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    try:
        from anthropic import Anthropic, AsyncAnthropic
    except ImportError:
        Anthropic = Any  # type: ignore
        AsyncAnthropic = Any  # type: ignore

    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = Any  # type: ignore


class ClaudeGenie(BaseGenie):
    """Anthropic's Claude Genie.
    
    Supports Anthropic's Claude models such as:
    - Claude 3 Opus
    - Claude 3 Sonnet
    - Claude 3 Haiku
    - And other models from the Claude family
    """

    def __init__(self,
                 model: str = "claude-3-opus-20240229",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """Initialize the ClaudeGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use. Defaults to env var ANTHROPIC_API_KEY.
            temperature (float): The temperature to use for generation. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 1024.
        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ClaudeGenie requires an API key. Either pass the `api_key` parameter or set the `ANTHROPIC_API_KEY` in your environment.")

        # Initialize the client and model
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        content = response.content[0].text
        if content is None:
            raise ValueError("Claude response content is None")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        # Construct a prompt that asks for a JSON response according to the schema
        schema_json = schema.schema_json()
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON that conforms to this schema:\n{schema_json}"
        
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": json_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system="Respond with valid, parseable JSON only"
        )
        
        content = response.content[0].text
        if content is None:
            raise ValueError("Claude response content is None")
        
        # Clean the response to extract just the JSON part
        content = self._extract_json(content)
        
        try:
            return dict(json.loads(content))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}, content: {content}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text which might contain markdown code blocks or other text."""
        # Try to extract JSON from markdown code blocks
        import re
        json_code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_code_block_pattern, text)
        if match:
            return match.group(1).strip()
        
        # If no code block is found, try to find content that looks like JSON
        # (starts with { and ends with })
        json_pattern = r"(\{[\s\S]*\})"
        match = re.search(json_pattern, text)
        if match:
            return match.group(1).strip()
        
        # If all else fails, return the original text
        return text.strip()

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environment."""
        if (importlib.util.find_spec("pydantic") is not None
                and importlib.util.find_spec("anthropic") is not None):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global Anthropic, BaseModel
            from anthropic import Anthropic
            from pydantic import BaseModel
        else:
            raise ImportError("One or more of the required modules are not available: [pydantic, anthropic]", 
                             "Please install the dependencies via `pip install chonkie[claude]`")

    def __repr__(self) -> str:
        """Return a string representation of the ClaudeGenie instance."""
        return f"ClaudeGenie(model={self.model})" 