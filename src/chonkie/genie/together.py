"""TogetherAI Genie."""
import importlib.util as importutil
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseGenie

if TYPE_CHECKING:
    try:
        from together import Together
    except ImportError:
        Together = Any  # type: ignore

    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = Any  # type: ignore


class TogetherGenie(BaseGenie):
    """TogetherAI's Genie.
    
    Supports TogetherAI's language models such as:
    - Llama 3
    - Mistral
    - Mixtral
    - And other models available on the TogetherAI platform
    """

    def __init__(self,
                 model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """Initialize the TogetherGenie class.

        Args:
            model (str): The model to use.
            api_key (Optional[str]): The API key to use. Defaults to env var TOGETHER_API_KEY.
            temperature (float): The temperature to use for generation. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 1024.
        """
        super().__init__()

        # Lazily import the dependencies
        self._import_dependencies()

        # Initialize the API key
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TogetherGenie requires an API key. Either pass the `api_key` parameter or set the `TOGETHER_API_KEY` in your environment.")

        # Initialize the client and model
        self.client = Together(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("TogetherAI response content is None")
        return content

    def generate_json(self, prompt: str, schema: "BaseModel") -> Dict[str, Any]:
        """Generate a JSON response based on the given prompt and schema."""
        # Construct a prompt that asks for a JSON response according to the schema
        json_prompt = f"{prompt}\n\nPlease format your response as a valid JSON object according to this schema: {schema.schema_json()}"
        
        # Set response format to JSON to help ensure proper formatting
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": json_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("TogetherAI response content is None")
        
        try:
            return dict(json.loads(content))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def _is_available(self) -> bool:
        """Check if all the dependencies are available in the environement."""
        if (importutil.find_spec("pydantic") is not None
                and importutil.find_spec("together") is not None):
            return True
        return False

    def _import_dependencies(self) -> None:
        """Import all the required dependencies."""
        if self._is_available():
            global Together, BaseModel
            from together import Together
            from pydantic import BaseModel
        else:
            raise ImportError("One or more of the required modules are not available: [pydantic, together]", 
                             "Please install the dependencies via `pip install chonkie[together]`")

    def __repr__(self) -> str:
        """Return a string representation of the TogetherGenie instance."""
        return f"TogetherGenie(model={self.model})" 